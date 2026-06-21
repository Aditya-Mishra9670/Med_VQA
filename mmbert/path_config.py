"""Shared path and environment helpers for MMBERT scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional


MMBERT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MMBERT_DIR.parent
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists() and KAGGLE_WORKING.exists()


def find_kaggle_dataset_root() -> Optional[Path]:
    if not KAGGLE_INPUT.exists():
        return None

    candidates = sorted(KAGGLE_INPUT.glob("**/vqa-med-2019"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def default_dataset_root() -> Path:
    env_path = os.environ.get("MEDVQA_DATASET_ROOT")
    if env_path:
        return Path(env_path).expanduser()

    if is_kaggle():
        kaggle_root = find_kaggle_dataset_root()
        if kaggle_root is not None:
            return kaggle_root
        return KAGGLE_INPUT

    return PROJECT_ROOT / "med-vqa-data" / "vqa-med-2019"


def default_converted_data_dir() -> Path:
    if is_kaggle():
        return KAGGLE_WORKING / "mmbert" / "data" / "vqa-med-2019-converted"
    return MMBERT_DIR / "data" / "vqa-med-2019-converted"


def default_checkpoint_dir() -> Path:
    if is_kaggle():
        return KAGGLE_WORKING / "mmbert" / "checkpoints"
    return MMBERT_DIR / "checkpoints"


def default_output_dir() -> Path:
    if is_kaggle():
        return KAGGLE_WORKING / "mmbert" / "outputs"
    return MMBERT_DIR / "outputs"


def default_cache_dir() -> Path:
    if is_kaggle():
        return KAGGLE_WORKING / "cache"
    return PROJECT_ROOT / ".cache"


def default_hf_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    if is_kaggle():
        return KAGGLE_WORKING / "hf_cache"
    return (cache_dir or default_cache_dir()) / "huggingface"


def add_path_args(parser: argparse.ArgumentParser, *, dataset_root: bool = True) -> None:
    if dataset_root:
        parser.add_argument(
            "--dataset_root",
            type=Path,
            default=None,
            help="Raw VQA-Med-2019 dataset root. Defaults to local med-vqa-data/vqa-med-2019 or auto-detected /kaggle/input/**/vqa-med-2019.",
        )
    parser.add_argument(
        "--converted_data_dir",
        type=Path,
        default=None,
        help="Converted MMBERT CSV/image dataset directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for generated outputs and predictions.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Cache directory for runtime files.",
    )


def resolve_path_args(args: argparse.Namespace, *, create: bool = True) -> argparse.Namespace:
    if hasattr(args, "dataset_root"):
        args.dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else default_dataset_root()

    args.converted_data_dir = (
        Path(args.converted_data_dir).expanduser()
        if args.converted_data_dir
        else default_converted_data_dir()
    )
    args.checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser()
        if args.checkpoint_dir
        else default_checkpoint_dir()
    )
    args.output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir()
    args.cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else default_cache_dir()
    args.hf_cache_dir = default_hf_cache_dir(args.cache_dir)

    if create:
        for path in (args.converted_data_dir, args.checkpoint_dir, args.output_dir, args.cache_dir, args.hf_cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    configure_huggingface_cache(args.hf_cache_dir)
    return args


def configure_huggingface_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)


def select_device(allow_directml: bool = False):
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")

    if allow_directml:
        try:
            import torch_directml

            return torch_directml.device()
        except ImportError:
            pass

    return torch.device("cpu")
