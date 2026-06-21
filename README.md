# Med-VQA

This repository contains medical visual question answering experiments for the ImageCLEF / VQA-Med 2019 dataset. The current reproducible path focuses on the MMBERT pipeline while preserving the original model behavior, vocabulary generation, label mappings, and evaluation metrics.

## MMBERT Pipeline

The MMBERT workflow has three steps:

1. Prepare VQA-Med data into the CSV/image layout expected by `mmbert/vqamed2019/train.py`.
2. Train one model per category, or a single combined model.
3. Run inference with saved category checkpoints.

Shared path defaults live in `mmbert/path_config.py`. The scripts auto-detect Kaggle and otherwise use repository-relative local paths.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or Kaggle, activate the environment with the equivalent shell command before installing requirements.

## Dataset Layout

Local default raw data path:

```text
med-vqa-data/vqa-med-2019/
```

Local default converted MMBERT path:

```text
mmbert/data/vqa-med-2019-converted/
```

Converted data must contain:

```text
traindf.csv
valdf.csv
testdf.csv
train_images/
val_images/
test_images/
```

Use explicit paths when your data is elsewhere:

```bash
python mmbert/prepare_dataset.py --dataset_root /path/to/vqa-med-2019 --converted_data_dir /path/to/converted
```

## Kaggle

Upload the raw dataset as a Kaggle input dataset, then run from the notebook working directory:

```bash
pip install -r requirements.txt
python mmbert/run_training.py --dataset_root /kaggle/input/<dataset-name>/vqa-med-2019 --epochs 2 --batch_size 4 --num_workers 0
```

Kaggle defaults:

```text
converted data : /kaggle/working/mmbert/data/vqa-med-2019-converted
checkpoints    : /kaggle/working/mmbert/checkpoints
outputs        : /kaggle/working/mmbert/outputs
HF cache       : /kaggle/working/hf_cache
```

## Training

Prepare data and train the default category models:

```bash
python mmbert/run_training.py --epochs 2 --batch_size 4 --num_workers 0
```

Train a single combined model:

```bash
python mmbert/run_training.py --categories all --run_prefix mmbert --epochs 2
```

Call the lower-level trainer directly:

```bash
python mmbert/vqamed2019/train.py --data_dir mmbert/data/vqa-med-2019-converted --save_dir mmbert/checkpoints --run_name test_run_modality --category modality --num_vis 36
```

Enable W&B explicitly with `--wandb`. CPU is used when CUDA is unavailable; `--allow_directml` enables DirectML as a Windows fallback if `torch-directml` is installed.

## Inference

Sample examples from `testdf.csv` and run prediction:

```bash
python mmbert/run_predict.py --num_samples 5 --run_prefix test_run
```

Run direct prediction:

```bash
python mmbert/vqamed2019/predict.py --items "path/to/image.jpg::what modality is shown?::modality" --checkpoint_dir mmbert/checkpoints --converted_data_dir mmbert/data/vqa-med-2019-converted
```

Prediction plots are written to `mmbert/outputs/` by default.

## Outputs

Generated outputs are ignored by Git:

```text
mmbert/checkpoints/
mmbert/outputs/
.cache/
hf_cache/
kaggle_outputs/
wandb/
```

## Troubleshooting

If a script cannot find the dataset, pass `--dataset_root` and `--converted_data_dir` explicitly.

If Hugging Face downloads repeatedly, verify `HF_HOME` and `TRANSFORMERS_CACHE` point to `.cache/huggingface` locally or `/kaggle/working/hf_cache` on Kaggle.

If checkpoints are missing during prediction, train the relevant category first or pass the matching `--run_prefix`.
