# Repository Guidelines

## Project Structure & Module Organization

- `scripts/`: Primary entrypoints for inference and demos (`inference.py`, `gradio_demo.py`, and MoE variants).
- `assets/`: Small sample images for smoke-testing (`assets/girl.png`, etc.).
- `docs/images/`: Documentation images referenced by `README.md`.
- `train/`: Training-related notes and a separate dependency set (`train/README.md`, `train/requirements.txt`).
- `requirements.txt`: Runtime/inference dependencies for the main scripts.

## Build, Test, and Development Commands

- Install (inference): `pip install -r requirements.txt`
- Run single-image inference:
  - `python scripts/inference.py --image assets/girl.png --instruction "Make her hair dark green." --seed 42`
  - Use local weights: add `--flux-path /path/to/flux.1-fill-dev --lora-path /path/to/ICEdit-normal-LoRA`
- Run Gradio demo: `python scripts/gradio_demo.py --port 7860`
- MoE variants: `python scripts/inference_moe.py ...` and `python scripts/gradio_demo_moe.py ...`

## Coding Style & Naming Conventions

- Language: Python (recommended: 3.10 to match `README.md`).
- Indentation: 4 spaces; prefer explicit imports and small, readable functions.
- Naming: `snake_case` for variables/functions, `PascalCase` for classes, and CLI flags in `kebab-case` (via `argparse`).
- Keep CLI arguments backward compatible; prefer adding new flags over changing defaults.

## Testing Guidelines

- No dedicated automated test suite in this repo. Do a quick smoke check by running `scripts/inference.py` on an image in `assets/`.
- Reproducibility: set `--seed`. Many scripts assume input width `512` (they may auto-resize).

## Commit & Pull Request Guidelines

- Commit messages in history are short and imperative (e.g., “update readme”, “release training code”), sometimes bilingual; follow the same style.
- PRs should include: what changed, how it was validated (exact command), and any user-facing screenshots (e.g., Gradio UI or before/after results) when applicable.

## Security & Configuration Tips

- Do not commit model weights, datasets, or generated outputs. Use `--flux-path`/`--lora-path` for local checkpoints and add large artifacts to `.gitignore` if needed.
