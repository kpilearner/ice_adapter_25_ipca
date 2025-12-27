# ICEdit Training Repository

This repository contains the training code for ICEdit, a model for image editing based on text instructions. It utilizes conditional generation to perform instructional image edits.

This codebase is based heavily on the [OminiControl](https://github.com/Yuanshi9815/OminiControl) repository. We thank the authors for their work and contributions to the field!

## Setup and Installation

```bash
# Create a new conda environment
conda create -n train python=3.10
conda activate train

# Install requirements
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code directory
  - `train/`: Training modules
    - `train.py`: Main training script
    - `data.py`: Dataset classes for handling different data formats
    - `model.py`: Model definition using Flux pipeline
    - `callbacks.py`: Training callbacks for logging and checkpointing
  - `flux/`: Flux model implementation
- `assets/`: Asset files
- `parquet/`: Parquet data files
- `requirements.txt`: Dependency list

## Datasets

Download training datasets (part of OmniEdit) to the `parquet/` directory. You can use the provided scripts `parquet/prepare.sh`.

```bash
cd parquet
bash prepare.sh
```

## Training

```bash
bash train/script/train.sh
```

You can modify the training configuration in `train/config/normal_lora.yaml`. 

### Paired translation (e.g., VIS→IR)

For paired, deterministic translation tasks, metrics like PSNR/SSIM typically benefit from (1) focusing the loss on the masked region and (2) using a conservative optimizer setup. A ready-to-start config is provided at `train/train/config/paired_translation_lora.yaml`:

```bash
bash train/train/script/train.sh paired_translation_lora.yaml 41353
```

### Local CSV + image pairs (Kontext-like format)

If you prefer not to build Parquet datasets, you can train from a local metadata CSV that points to paired images (similar to Kontext examples). Use `train/train/config/paired_csv_lora.yaml` and prepare a CSV like:

```csv
kontext_images,image,prompt
vis/0001.png,ir/0001.png,convert to infrared image.
vis/0002.png,ir/0002.png,convert to infrared image.
```

Then run:

```bash
bash train/train/script/train.sh paired_csv_lora.yaml 41353
```

### Using trained adapters at inference

If you train with `train.adapter.enabled: true`, checkpoints saved by the training callback also include:

- `prompt_adapter.pt` (pooled MLP adapter, if enabled)
- `text_token_adapter.pt` (token adapter, if enabled)
- `adapter_config.json`

You can load them in inference scripts via `--adapter-dir /path/to/ckpt_dir`.



## MoE-LoRA Training
