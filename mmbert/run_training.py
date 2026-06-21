#!/usr/bin/env python
"""
Prepare the VQA-Med-2019 dataset and launch MMBERT training.

The trainer in vqamed2019/train.py expects:
    traindf.csv, valdf.csv, testdf.csv
    train_images/, val_images/, test_images/

This wrapper converts the JSONL + pickle files into that structure before
launching training. Paths default to Kaggle locations when running on Kaggle
and to repository-relative locations otherwise.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from path_config import MMBERT_DIR, add_path_args, resolve_path_args


TRAIN_JSONL = "mmf_data/dev_unseen.jsonl"
EVAL_JSONL = "mmf_data/dev_seen.jsonl"
DEFAULT_CATEGORIES = ["modality", "plane", "organ", "abnormality"]


def normalize_question(text):
    return " ".join(str(text).strip().lower().split())


def infer_category(question):
    q = normalize_question(question)

    modality_terms = [
        "modality",
        "kind of image",
        "kind of scan",
        "type of image",
        "type of scan",
        "imaging technique",
        "mri",
        "ct",
        "xray",
        "x-ray",
        "ultrasound",
        "pet",
        "fluoroscopic",
        "angiogram",
        "contrast",
        "t1 weighted",
        "t2 weighted",
    ]
    plane_terms = ["plane", "view", "orientation", "axial", "coronal", "sagittal", "ap view", "pa view", "lateral"]
    organ_terms = ["organ", "body part", "anatomical", "anatomy", "where is", "what is shown", "what part", "which part"]
    abnormality_terms = ["abnormal", "abnormality", "disease", "diagnosis", "finding", "lesion", "mass", "tumor", "fracture", "present", "seen in"]

    if any(term in q for term in plane_terms):
        return "plane"
    if any(term in q for term in organ_terms):
        return "organ"
    if any(term in q for term in abnormality_terms):
        return "abnormality"
    if any(term in q for term in modality_terms):
        return "modality"
    return "abnormality"


def load_answer_lookup(source_dir):
    lookup = {}
    for filename in ("data_dictionary.pkl", "data_dictionary_val.pkl"):
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required answer dictionary not found: {path}")

        with path.open("rb") as f:
            records = pickle.load(f)

        for record in records:
            for image_id, qa in record.items():
                question = normalize_question(qa.get("ques", ""))
                answer = str(qa.get("ans", "")).strip().lower()
                if image_id and question and answer:
                    lookup[(str(image_id), question)] = answer

    return lookup


def resolve_image(source_dir, row):
    mmf_dir = source_dir / "mmf_data"
    image_ref = Path(str(row.get("img") or row["id"]))
    candidates = [
        mmf_dir / f"{image_ref}.jpg",
        mmf_dir / image_ref,
        source_dir / f"{image_ref}.jpg",
        source_dir / image_ref,
        source_dir / "resized-images-train" / f"{row['id']}.jpg",
        source_dir / "resized-images-val" / f"{row['id']}.jpg",
        mmf_dir / "Resize_images" / f"{row['id']}.jpg",
        mmf_dir / "Resize_images_val" / f"{row['id']}.jpg",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON in {path}:{line_number}: {exc}") from exc
    return rows


def prepare_split(source_dir, output_dir, mode, rel_jsonl, image_folder, answer_lookup):
    jsonl_path = source_dir / rel_jsonl
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Required split file not found: {jsonl_path}")

    dst_img_dir = output_dir / image_folder
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing_answers = 0
    missing_images = 0

    for row in load_jsonl(jsonl_path):
        image_id = str(row["id"]).strip()
        question = str(row["text"]).strip()
        answer = answer_lookup.get((image_id, normalize_question(question)))
        image_path = resolve_image(source_dir, row)

        if not answer:
            missing_answers += 1
            continue
        if not image_path:
            missing_images += 1
            continue

        dst_path = dst_img_dir / f"{image_id}.jpg"
        if not dst_path.exists():
            shutil.copy2(image_path, dst_path)

        records.append(
            {
                "img_id": image_id,
                "category": infer_category(question),
                "question": question,
                "answer": answer,
                "mode": mode,
            }
        )

    if missing_answers or missing_images:
        print(
            f"  [WARNING] {mode}: skipped {missing_answers} rows without answers "
            f"and {missing_images} rows without images"
        )

    return pd.DataFrame(records)


def split_eval_df(df):
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    midpoint = max(1, len(shuffled) // 2)

    val_df = shuffled.iloc[:midpoint].copy().reset_index(drop=True)
    test_df = shuffled.iloc[midpoint:].copy().reset_index(drop=True)
    val_df["mode"] = "val"
    test_df["mode"] = "test"

    return val_df, test_df


def prepare_dataset(args):
    source_dir = args.dataset_root
    output_dir = args.converted_data_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source data dir not found: {source_dir}")

    required_csvs = [output_dir / "traindf.csv", output_dir / "valdf.csv", output_dir / "testdf.csv"]
    if not args.rebuild_data and all(path.exists() for path in required_csvs):
        print(f"Using existing converted dataset: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    answer_lookup = load_answer_lookup(source_dir)

    train_df = prepare_split(source_dir, output_dir, "train", TRAIN_JSONL, "train_images", answer_lookup)
    eval_df = prepare_split(source_dir, output_dir, "val", EVAL_JSONL, "val_images", answer_lookup)
    val_df, test_df = split_eval_df(eval_df)

    test_img_dir = output_dir / "test_images"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    for image_id in test_df["img_id"].unique():
        src = output_dir / "val_images" / f"{image_id}.jpg"
        dst = test_img_dir / f"{image_id}.jpg"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    for mode, df in split_dfs.items():
        csv_name = "traindf.csv" if mode == "train" else f"{mode}df.csv"
        df.to_csv(output_dir / csv_name, index=False)

    combined = pd.concat(split_dfs.values(), ignore_index=True)
    combined.to_csv(output_dir / "data.csv", index=False)

    print("=" * 60)
    print("  Converted VQA-Med-2019 dataset")
    print("=" * 60)
    print(f"  Source : {source_dir}")
    print(f"  Output : {output_dir}")
    for mode, df in split_dfs.items():
        counts = Counter(df["category"])
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {mode:<5} : {len(df):>5} rows ({summary})")
    print("=" * 60 + "\n")


def run_category(args, category):
    run_name = f"{args.run_prefix}_{category.lower()}" if category else f"{args.run_prefix}_all"
    train_script = MMBERT_DIR / "vqamed2019" / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir",
        str(args.converted_data_dir),
        "--save_dir",
        str(args.checkpoint_dir),
        "--output_dir",
        str(args.output_dir),
        "--cache_dir",
        str(args.cache_dir),
        "--run_name",
        run_name,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--num_vis",
        str(args.num_vis),
        "--hidden_size",
        str(args.hidden_size),
        "--patience",
        str(args.patience),
    ]

    if args.smoothing is not None:
        cmd.extend(["--smoothing", str(args.smoothing)])
    if category:
        cmd.extend(["--category", category])
    if args.mixed_precision:
        cmd.append("--mixed_precision")
    if args.wandb:
        cmd.append("--wandb")
    if args.allow_directml:
        cmd.append("--allow_directml")

    print("=" * 60)
    print(f"  Training category: {category or 'all'}")
    print(f"  Run name         : {run_name}")
    print(f"  Checkpoint       : {run_name}_acc.pt")
    print("=" * 60)
    print(f"  Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[WARNING] Training failed for category '{category or 'all'}' (exit code {result.returncode})")
    else:
        print(f"\n[OK] Finished category '{category or 'all'}'\n")

    return result.returncode, run_name


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare VQA-Med-2019 data and train MMBERT.")
    add_path_args(parser)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES, help="Categories to train, or 'all' for one model.")
    parser.add_argument("--run_prefix", default="test_run", help="Prefix used for checkpoint run names.")
    parser.add_argument("--rebuild_data", action="store_true", default=False, help="Rebuild converted data even if CSVs already exist.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_vis", type=int, default=36, required=False)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--allow_directml", action="store_true", default=False, help="Allow DirectML as a local fallback when CUDA is unavailable.")
    return parser


def main():
    parser = build_parser()
    args = resolve_path_args(parser.parse_args())
    categories = [None if category.lower() == "all" else category.lower() for category in args.categories]

    prepare_dataset(args)

    print(f"Data directory       : {args.converted_data_dir}")
    print(f"Checkpoint directory : {args.checkpoint_dir}")
    print(f"Output directory     : {args.output_dir}")
    print(f"Cache directory      : {args.cache_dir}")
    print(f"Categories to train  : {[c or 'all' for c in categories]}\n")

    summary = {}
    for category in categories:
        rc, run_name = run_category(args, category)
        ckpt_path = args.checkpoint_dir / f"{run_name}_acc.pt"
        exists = "checkpoint saved" if ckpt_path.exists() else "no checkpoint"
        summary[category or "all"] = ("OK" if rc == 0 else f"FAILED ({rc})", exists)

    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for category, (status, exists) in summary.items():
        print(f"  {category:<12} : {status:<12} ({exists})")
    print("=" * 60)


if __name__ == "__main__":
    main()
