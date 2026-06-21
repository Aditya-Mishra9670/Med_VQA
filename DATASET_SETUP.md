# Dataset Setup

## Local Setup

Place raw VQA-Med 2019 data here, or pass the path explicitly:

```text
med-vqa-data/vqa-med-2019/
```

Then convert it for MMBERT:

```bash
python mmbert/prepare_dataset.py --dataset_root med-vqa-data/vqa-med-2019 --converted_data_dir mmbert/data/vqa-med-2019-converted
```

The converted layout is:

```text
mmbert/data/vqa-med-2019-converted/
  traindf.csv
  valdf.csv
  testdf.csv
  data.csv
  train_images/
  val_images/
  test_images/
```

## Kaggle Upload

Create a Kaggle dataset containing the VQA-Med 2019 folder, for example:

```text
vqa-med-2019/
  mmf_data/
  data_dictionary.pkl
  data_dictionary_val.pkl
```

In a Kaggle notebook, mount it as an input and run:

```bash
python mmbert/run_training.py --dataset_root /kaggle/input/<dataset-name>/vqa-med-2019 --rebuild_data
```

If your Kaggle input uses a different folder name, point `--dataset_root` to the folder containing the required files.

## Path Defaults

Local defaults:

```text
dataset root        : med-vqa-data/vqa-med-2019
converted data      : mmbert/data/vqa-med-2019-converted
checkpoints         : mmbert/checkpoints
outputs             : mmbert/outputs
Hugging Face cache  : .cache/huggingface
```

Kaggle defaults:

```text
converted data      : /kaggle/working/mmbert/data/vqa-med-2019-converted
checkpoints         : /kaggle/working/mmbert/checkpoints
outputs             : /kaggle/working/mmbert/outputs
Hugging Face cache  : /kaggle/working/hf_cache
```

## Notes

The conversion scripts preserve the original train/validation/test logic used by the MMBERT training code. Use `--rebuild_data` only when the source dataset or conversion target should be regenerated.
