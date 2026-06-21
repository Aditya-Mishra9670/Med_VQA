#!/usr/bin/env python
"""Prepare a pipe-delimited VQA-Med split into the MMBERT CSV/image layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from path_config import add_path_args, resolve_path_args


def load_answers(path):
    """Load rows formatted as synpic_id|category|question|answer."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                print(f"  [SKIP] bad answer line: {line}")
                continue
            rows.append(
                {
                    "img_id": parts[0].strip(),
                    "category": parts[1].strip(),
                    "question": parts[2].strip(),
                    "answer": parts[3].strip(),
                }
            )
    return pd.DataFrame(rows)


def load_questions(path):
    """Load rows formatted as synpic_id|question."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                print(f"  [SKIP] bad question line: {line}")
                continue
            rows.append({"img_id": parts[0].strip(), "question": parts[1].strip()})
    return pd.DataFrame(rows)


def copy_images(image_names, src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name in image_names:
        candidates = [name, f"{name}.jpg", f"{name}.png"]
        copied = False
        for candidate in candidates:
            src = src_dir / candidate
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
                copied = True
                break
        if not copied:
            missing += 1
    if missing:
        print(f"  [WARNING] {missing} images not found in source folder")


def default_test_images_dir(dataset_root):
    return dataset_root / "VQAMed2019Test" / "VQAMed2019_Test_Images" / "VQAMed2019_Test_Images"


def default_answers_txt(dataset_root):
    return dataset_root / "VQAMed2019Test" / "VQAMed2019_Test_Questions_w_Ref_Answers.txt"


def default_questions_txt(dataset_root):
    return dataset_root / "VQAMed2019Test" / "VQAMed2019_Test_Questions.txt"


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare pipe-delimited VQA-Med data for MMBERT.")
    add_path_args(parser)
    parser.add_argument("--images_dir", type=Path, default=None, help="Source image directory.")
    parser.add_argument("--answers_txt", type=Path, default=None, help="Pipe-delimited question/answer file.")
    parser.add_argument("--questions_txt", type=Path, default=None, help="Pipe-delimited question file.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = resolve_path_args(parser.parse_args())

    images_dir = args.images_dir or default_test_images_dir(args.dataset_root)
    answers_txt = args.answers_txt or default_answers_txt(args.dataset_root)
    questions_txt = args.questions_txt or default_questions_txt(args.dataset_root)
    output_dir = args.converted_data_dir

    for path, label in ((images_dir, "images_dir"), (answers_txt, "answers_txt"), (questions_txt, "questions_txt")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if round(args.train_ratio + args.val_ratio + args.test_ratio, 6) != 1.0:
        raise ValueError("--train_ratio + --val_ratio + --test_ratio must equal 1.0")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading answers file...")
    ans_df = load_answers(answers_txt)
    print(f"  {len(ans_df)} rows loaded")

    print("Loading questions file...")
    q_df = load_questions(questions_txt)
    print(f"  {len(q_df)} rows loaded")

    q_only = q_df[~q_df["img_id"].isin(ans_df["img_id"])].copy()
    q_only["category"] = ""
    q_only["answer"] = ""

    full_df = pd.concat([ans_df, q_only], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["img_id", "question"])
    print(f"\nTotal unique QA pairs: {len(full_df)}")

    unique_images = full_df["img_id"].unique()
    print(f"Unique images       : {len(unique_images)}")

    train_imgs, temp_imgs = train_test_split(
        unique_images,
        test_size=(args.val_ratio + args.test_ratio),
        random_state=args.seed,
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
        random_state=args.seed,
    )

    mode_map = {img: "train" for img in train_imgs}
    mode_map.update({img: "val" for img in val_imgs})
    mode_map.update({img: "test" for img in test_imgs})
    full_df["mode"] = full_df["img_id"].map(mode_map)

    full_df.to_csv(output_dir / "data.csv", index=False)
    full_df[full_df["mode"] == "train"].reset_index(drop=True).to_csv(output_dir / "traindf.csv", index=False)
    full_df[full_df["mode"] == "val"].reset_index(drop=True).to_csv(output_dir / "valdf.csv", index=False)
    full_df[full_df["mode"] == "test"].reset_index(drop=True).to_csv(output_dir / "testdf.csv", index=False)

    train_count = (full_df["mode"] == "train").sum()
    val_count = (full_df["mode"] == "val").sum()
    test_count = (full_df["mode"] == "test").sum()

    print("\nSplit sizes:")
    print(f"  Train : {train_count} rows  ({len(train_imgs)} images)")
    print(f"  Val   : {val_count} rows  ({len(val_imgs)} images)")
    print(f"  Test  : {test_count} rows  ({len(test_imgs)} images)")

    print("\nCopying images...")
    copy_images(train_imgs, images_dir, output_dir / "train_images")
    print("  train_images/ done")
    copy_images(val_imgs, images_dir, output_dir / "val_images")
    print("  val_images/   done")
    copy_images(test_imgs, images_dir, output_dir / "test_images")
    print("  test_images/  done")

    print("\n" + "=" * 55)
    print("DONE! Output structure:")
    print(f"  {output_dir}/")
    print(f"  data.csv         (combined, {len(full_df)} rows, has 'mode' column)")
    print(f"  traindf.csv      ({train_count} rows)")
    print(f"  valdf.csv        ({val_count} rows)")
    print(f"  testdf.csv       ({test_count} rows)")
    print("  train_images/")
    print("  val_images/")
    print("  test_images/")
    print("\nCSV columns: img_id | category | question | answer | mode")
    print("=" * 55)


if __name__ == "__main__":
    main()
