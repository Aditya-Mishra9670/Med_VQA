#!/usr/bin/env python
"""Sample rows from converted VQA-Med data and run MMBERT prediction."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

from path_config import MMBERT_DIR, add_path_args, resolve_path_args


VALID_CATEGORIES = {"modality", "plane", "organ", "abnormality"}


def load_random_items(csv_path: Path, img_dir: Path, categories: set[str], n: int, seed: int | None):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header[:3] != ["img_id", "category", "question"]:
            f.seek(0)
            reader = csv.reader(f)

        for row in reader:
            if len(row) < 3:
                continue
            image_id, category, question = row[0].strip(), row[1].strip().lower(), row[2].strip()
            if category not in categories:
                continue
            image_path = img_dir / f"{image_id}.jpg"
            rows.append((image_path, question, category))

    if seed is not None:
        random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def build_parser():
    parser = argparse.ArgumentParser(description="Run MMBERT prediction on random converted-data samples.")
    add_path_args(parser, dataset_root=False)
    parser.add_argument("--test_csv", type=Path, default=None, help="CSV to sample from. Defaults to converted_data_dir/testdf.csv.")
    parser.add_argument("--test_img_dir", type=Path, default=None, help="Image directory. Defaults to converted_data_dir/test_images.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random sample seed.")
    parser.add_argument("--categories", nargs="+", default=["modality", "plane", "organ"], help="Categories to sample.")
    parser.add_argument("--run_prefix", default="test_run", help="Checkpoint prefix used during training.")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_vis", type=int, default=36)
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to output_dir/prediction_results.png.")
    parser.add_argument("--allow_directml", action="store_true", default=False, help="Allow DirectML when CUDA is unavailable.")
    return parser


def main():
    args = resolve_path_args(build_parser().parse_args())

    categories = {category.lower() for category in args.categories}
    invalid = categories - VALID_CATEGORIES
    if invalid:
        raise ValueError(f"Invalid categories: {sorted(invalid)}. Valid categories: {sorted(VALID_CATEGORIES)}")

    test_csv = args.test_csv.expanduser() if args.test_csv else args.converted_data_dir / "testdf.csv"
    test_img_dir = args.test_img_dir.expanduser() if args.test_img_dir else args.converted_data_dir / "test_images"
    output = args.output.expanduser() if args.output else args.output_dir / "prediction_results.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")

    items = load_random_items(test_csv, test_img_dir, categories, args.num_samples, args.seed)
    if not items:
        raise RuntimeError(f"No matching rows found in {test_csv} for categories {sorted(categories)}")

    for image_path, _, category in items:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        ckpt_path = args.checkpoint_dir / f"{args.run_prefix}_{category}_acc.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for category '{category}': {ckpt_path}")

    predict_script = MMBERT_DIR / "vqamed2019" / "predict.py"
    item_args = [f"{image_path}::{question}::{category}" for image_path, question, category in items]
    cmd = [
        sys.executable,
        str(predict_script),
        "--items",
        *item_args,
        "--weights_dir",
        str(args.checkpoint_dir),
        "--data_dir",
        str(args.converted_data_dir),
        "--cache_dir",
        str(args.cache_dir),
        "--run_prefix",
        args.run_prefix,
        "--hidden_size",
        str(args.hidden_size),
        "--num_vis",
        str(args.num_vis),
        "--output",
        str(output),
    ]
    if args.allow_directml:
        cmd.append("--allow_directml")

    print("=" * 60)
    print("  Med-VQA Prediction")
    print("=" * 60)
    for i, (image_path, question, category) in enumerate(items, 1):
        print(f"  [{i}] Image    : {image_path}")
        print(f"      Category : {category}")
        print(f"      Question : {question}")
    print(f"  Data dir    : {args.converted_data_dir}")
    print(f"  Weights dir : {args.checkpoint_dir}")
    print(f"  Output      : {output}")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
