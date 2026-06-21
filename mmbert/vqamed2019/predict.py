#!/usr/bin/env python
"""Run inference on one or more images with per-category MMBERT models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MMBERT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MMBERT_DIR))

from path_config import configure_huggingface_cache, default_checkpoint_dir, default_converted_data_dir, default_hf_cache_dir, default_output_dir, select_device

configure_huggingface_cache(default_hf_cache_dir())

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from utils import Model, load_bert_tokenizer, load_data


VALID_CATEGORIES = {"modality", "plane", "organ", "abnormality"}


def build_ans_maps_for_category(category, args):
    train_df, val_df, test_df = load_data(args)

    train_df = train_df[train_df["category"] == category].reset_index(drop=True)
    val_df = val_df[val_df["category"] == category].reset_index(drop=True)
    test_df = test_df[test_df["category"] == category].reset_index(drop=True)

    train_df = train_df[~train_df["answer"].isin(["yes", "no"])].reset_index(drop=True)
    val_df = val_df[~val_df["answer"].isin(["yes", "no"])].reset_index(drop=True)
    test_df = test_df[~test_df["answer"].isin(["yes", "no"])].reset_index(drop=True)

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    ans2idx = {ans: idx for idx, ans in enumerate(df["answer"].unique())}
    idx2ans = {idx: ans for ans, idx in ans2idx.items()}
    return ans2idx, idx2ans


def load_model(weights_path: Path, num_classes: int, args, device):
    model = Model(args)
    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return tfm(img).unsqueeze(0), img


def preprocess_question(question, tokenizer, max_len=28):
    encoding = tokenizer(
        question,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["token_type_ids"], encoding["attention_mask"]


@torch.no_grad()
def predict(image_path: Path, question: str, model, tokenizer, idx2ans, args, device):
    img_tensor, img_rgb = preprocess_image(image_path)
    q_ids, seg_ids, attn_mask = preprocess_question(question, tokenizer, args.max_position_embeddings)

    img_tensor = img_tensor.to(device)
    q_ids = q_ids.to(device)
    seg_ids = seg_ids.to(device)
    attn_mask = attn_mask.to(device)

    logits, _, _ = model(img_tensor, q_ids, seg_ids, attn_mask)

    probs = logits.softmax(1).squeeze(0).cpu().numpy()
    k = min(5, len(idx2ans))
    topk_idx = probs.argsort()[::-1][:k]
    topk_ans = [idx2ans[i] for i in topk_idx]
    topk_probs = probs[topk_idx]

    return topk_ans[0], topk_probs[0], topk_ans, topk_probs, img_rgb


def visualize_grid(results, save_path: Path, show: bool = False):
    n = len(results)
    row_height = 3.2

    fig = plt.figure(figsize=(16, row_height * n), constrained_layout=False)
    fig.patch.set_facecolor("#1e1e2e")

    gs = fig.add_gridspec(
        nrows=n,
        ncols=2,
        width_ratios=[1, 1.6],
        hspace=1.0,
        wspace=0.3,
        left=0.06,
        right=0.97,
        top=1 - (0.4 / (row_height * n)),
        bottom=0.02,
    )

    for i, r in enumerate(results):
        image_path = r["image_path"]
        question = r["question"]
        category = r["category"]
        pred_ans = r["pred_ans"]
        pred_conf = r["pred_conf"]
        top5_ans = r["top5_ans"]
        top5_probs = r["top5_probs"]
        img_rgb = r["img_rgb"]

        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(img_rgb, aspect="equal")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)
        ax_img.set_title(f"[{i + 1}] {image_path.name} ({category})", color="white", fontsize=10, pad=4)
        ax_img.text(
            0.5,
            -0.18,
            f"Q: {question}",
            transform=ax_img.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#cdd6f4",
            style="italic",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#313244", alpha=0.85),
        )

        ax_bar = fig.add_subplot(gs[i, 1])
        ax_bar.set_facecolor("#181825")
        colors = ["#a6e3a1" if k == 0 else "#89b4fa" for k in range(len(top5_ans))]
        bars = ax_bar.barh(range(len(top5_ans)), top5_probs * 100, color=colors, height=0.55, edgecolor="none")

        for bar, prob in zip(bars, top5_probs):
            ax_bar.text(
                bar.get_width() + max(top5_probs * 100) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%",
                va="center",
                color="white",
                fontsize=8,
            )

        ax_bar.set_yticks(range(len(top5_ans)))
        ax_bar.set_yticklabels(top5_ans, color="white", fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Confidence (%)", color="#cdd6f4", fontsize=8)
        ax_bar.set_xlim(0, max(top5_probs * 100) * 1.30)
        ax_bar.tick_params(colors="white", labelsize=8)
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#45475a")
        ax_bar.set_title(
            f"[{i + 1}] Answer: {pred_ans.upper()} ({pred_conf * 100:.1f}%)",
            color="#a6e3a1",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )

        if i == 0:
            legend_handles = [
                mpatches.Patch(color="#a6e3a1", label="Top prediction"),
                mpatches.Patch(color="#89b4fa", label="Other candidates"),
            ]
            ax_bar.legend(
                handles=legend_handles,
                loc="lower right",
                facecolor="#313244",
                edgecolor="none",
                labelcolor="white",
                fontsize=7,
            )

    fig.suptitle("Med-VQA - MMBERT Predictions", color="#cdd6f4", fontsize=15, y=1 - 0.1 / (row_height * n))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_summary_table(results):
    print("\n" + "=" * 85)
    print(f"{'#':<3} {'Image':<20} {'Category':<12} {'Question':<25} {'Answer':<15} {'Conf':>6}")
    print("-" * 85)
    for i, r in enumerate(results, 1):
        img_name = r["image_path"].name
        q = r["question"]
        q_disp = (q[:22] + "...") if len(q) > 25 else q
        print(f"{i:<3} {img_name:<20} {r['category']:<12} {q_disp:<25} {r['pred_ans']:<15} {r['pred_conf'] * 100:>5.1f}%")
    print("=" * 85)


def parse_item(item: str):
    parts = item.split("::")
    if len(parts) != 3:
        raise ValueError(f"Invalid --items entry (expected 'image_path::question::category'): {item}")

    image_path, question, category = (p.strip() for p in parts)
    category = category.lower()
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Valid categories: {sorted(VALID_CATEGORIES)}")

    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path, question, category


def build_parser():
    parser = argparse.ArgumentParser(description="Med-VQA multi-image prediction with per-category MMBERT models.")
    parser.add_argument("--items", type=str, nargs="+", required=True, help='Entries formatted as "image_path::question::category".')
    parser.add_argument("--weights_dir", "--checkpoint_dir", dest="weights_dir", type=Path, default=default_checkpoint_dir(), help="Directory containing {run_prefix}_{category}_acc.pt checkpoints.")
    parser.add_argument("--data_dir", "--converted_data_dir", dest="data_dir", type=Path, default=default_converted_data_dir(), help="Converted data directory used for training.")
    parser.add_argument("--output", type=Path, default=default_output_dir() / "prediction_results.png", help="Output image path.")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Cache directory.")
    parser.add_argument("--run_prefix", default="test_run", help="Checkpoint prefix used during training.")
    parser.add_argument("--allow_directml", action="store_true", default=False, help="Allow DirectML when CUDA is unavailable.")
    parser.add_argument("--show", action="store_true", default=False, help="Show matplotlib window after saving.")

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_vis", type=int, default=36)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_position_embeddings", type=int, default=28)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--smoothing", type=float, default=None)
    parser.add_argument("--train_pct", type=float, default=1.0)
    parser.add_argument("--valid_pct", type=float, default=1.0)
    parser.add_argument("--test_pct", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    return parser


def main():
    args = build_parser().parse_args()
    args.data_dir = str(args.data_dir.expanduser())
    args.weights_dir = args.weights_dir.expanduser()
    args.output = args.output.expanduser()
    cache_dir = args.cache_dir.expanduser() if args.cache_dir else None
    configure_huggingface_cache(default_hf_cache_dir(cache_dir))

    parsed_items = [parse_item(item) for item in args.items]
    device = select_device(args.allow_directml)
    print(f"Using device: {device}")

    tokenizer = load_bert_tokenizer()
    category_cache = {}

    def get_category_resources(category):
        if category in category_cache:
            return category_cache[category]

        print(f"\nBuilding answer vocabulary for category: {category} ...")
        args.category = category
        ans2idx, idx2ans = build_ans_maps_for_category(category, args)
        num_classes = len(ans2idx)
        print(f"  {num_classes} unique answers found for {category}")

        weights_path = args.weights_dir / f"{args.run_prefix}_{category}_acc.pt"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for category '{category}': {weights_path}")

        print(f"Loading weights from: {weights_path}")
        model = load_model(weights_path, num_classes, args, device)
        print("  Model loaded")

        category_cache[category] = (model, idx2ans)
        return model, idx2ans

    results = []
    for idx, (image_path, question, category) in enumerate(parsed_items, 1):
        print(f"\n[{idx}/{len(parsed_items)}] Image: {image_path} | Category: {category} | Question: {question}")
        model, idx2ans = get_category_resources(category)
        pred_ans, pred_conf, top5_ans, top5_probs, img_rgb = predict(
            image_path, question, model, tokenizer, idx2ans, args, device
        )

        print(f"  Predicted answer: {pred_ans} ({pred_conf * 100:.2f}%)")
        for rank, (ans, prob) in enumerate(zip(top5_ans, top5_probs), 1):
            print(f"      {rank}. {ans:<30} {prob * 100:.2f}%")

        results.append(
            {
                "image_path": image_path,
                "question": question,
                "category": category,
                "pred_ans": pred_ans,
                "pred_conf": pred_conf,
                "top5_ans": top5_ans,
                "top5_probs": top5_probs,
                "img_rgb": img_rgb,
            }
        )

    print_summary_table(results)
    visualize_grid(results, args.output, show=args.show)


if __name__ == "__main__":
    main()
