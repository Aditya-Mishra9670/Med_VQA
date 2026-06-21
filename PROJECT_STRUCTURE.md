# Project Structure

```text
.
  README.md
  DATASET_SETUP.md
  PROJECT_STRUCTURE.md
  requirements.txt
  mmbert/
  hierarchical/
  mmf/
  dataset_plots/
```

## Main Directories

`mmbert/` contains the reproducible MMBERT pipeline:

```text
mmbert/path_config.py
mmbert/prepare_dataset.py
mmbert/run_training.py
mmbert/run_predict.py
mmbert/eval.py
mmbert/vqamed2019/train.py
mmbert/vqamed2019/predict.py
mmbert/vqamed2019/utils.py
```

`hierarchical/` contains the hierarchical question-image co-attention baseline.

`mmf/` contains the MMF-based baselines and vendored MMF code used by the original project.

`dataset_plots/` contains static dataset analysis images referenced by the original README.

## Generated Folders

These folders are generated locally or on Kaggle and are ignored by Git:

```text
mmbert/data/
mmbert/checkpoints/
mmbert/outputs/
logs/
runs/
outputs/
wandb/
.cache/
hf_cache/
kaggle_outputs/
```

## Tracked Source vs Artifacts

Python source, documentation, and configuration should be tracked. Python bytecode, model checkpoints, runtime logs, converted datasets, local notebooks caches, and Hugging Face caches should remain untracked.
