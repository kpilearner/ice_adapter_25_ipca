# Repository Guidelines

## Project Structure & Module Organization
This repo is split between inference utilities and a nested training repo.

- `scripts/`: Inference and tooling (`inference.py`, `inference_folder.py`, `gradio_demo.py`, `make_paired_csv.py`).
- `run_inference.sh`, `run_inference_folder.sh`: Example launchers that wire paths, prompts, and output locations.
- `train/`: Training code and assets.
  - `train/src/`: Core training modules (`train/`, `flux/`).
  - `train/train/config/`: YAML configs (e.g., `normal_lora.yaml`, `paired_csv_lora.yaml`).
  - `train/train/script/`: Training entrypoints (`train.sh`, `train_moe.sh`).
  - `train/parquet/`: Dataset preparation scripts and data storage.
  - `train/assets/`: Sample assets and references.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs inference/demo dependencies.
- `./run_inference.sh` runs single-image inference after you set `FLUX_PATH`, `LORA_PATH`, and `ADAPTER_DIR` in the script.
- `./run_inference_folder.sh` runs batch inference for a folder; adjust `INPUT_DIR`, `OUTPUT_DIR`, and sizes.
- `python scripts/gradio_demo.py --port 7860` launches the Gradio UI demo.
- `python scripts/make_paired_csv.py --vis-dir data/vis --ir-dir data/ir --out train/parquet/metadata.csv --base-path train` builds paired CSV metadata.
- `pip install -r train/requirements.txt` installs training dependencies.
- `bash train/train/script/train.sh` trains using `train/train/config/normal_lora.yaml`.
- `bash train/train/script/train.sh paired_csv_lora.yaml 41353` runs paired CSV training.
- `bash train/parquet/prepare.sh` fetches/organizes parquet datasets.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/vars, `UpperCamelCase` for classes.
- CLI flags use `--kebab-case` as shown in `scripts/`.
- Configs live under `train/train/config/` and use descriptive suffixes like `*_lora.yaml`.
- Shell scripts follow `set -euo pipefail` and keep paths configurable at the top.

## Testing Guidelines
No automated test suite is checked in. Validate changes by running inference on a small sample or a short training run. If you add tests, place them in a `tests/` or `train/tests/` directory and document the run command.

## Commit & Pull Request Guidelines
Git history is minimal with simple messages like “first commit” and “Initial commit of forked ICDEdit project.” Keep commit messages short, sentence-case, and action-oriented (e.g., “Add paired CSV helper”). PRs should describe the goal, list commands run, and call out config or dataset changes. Avoid committing large model weights; reference local paths instead.
