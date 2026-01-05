#!/usr/bin/env bash

# Stage1 inference for a folder of images using CSV prompts.
# Usage: ./run_inference_stage1_folder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------
# Required configuration
# -----------------------

METADATA_PATH="/root/autodl-tmp/MSRS/metadata.csv"
INPUT_DIR="/root/autodl-tmp/MSRS/test/vi"
BASE_PATH=""   # set to "" to use INPUT_DIR for CSV path resolution
IMAGE_KEY="kontext_images"
CAPTION_KEY="prompt"
OUTPUT_DIR="./stage1_folder_outputs"

FLUX_PATH="/root/autodl-tmp/qyt/hfd_model"
LORA_PATH="/root/autodl-tmp/qyt_adapter/ICEdit_adapter_kontext/ICEdit_adapter/train/runs/20251215-131532/ckpt/10000/pytorch_lora_weights.safetensors"
ADAPTER_DIR="/root/autodl-tmp/qyt_adapter/ICEdit_adapter_kontext/ICEdit_adapter/train/runs/20251215-131532/ckpt/10000"   # set to "" to disable

PROMPT_PREFIX="A diptych with two side-by-side images of the same scene. The left image is visible light. The right image is the corresponding infrared thermal image. Instruction: "
INSTRUCTION="Transform the visible image into its infrared thermal counterpart while preserving scene structure and objects."

# -----------------------
# Optional configuration
# -----------------------
SEED=42
SEED_MODE="fixed"          # fixed | incremental | hash
GUIDANCE_SCALE=50
STEPS=28
RESIZE_MODE="square"      # square | width
SIZE=512
RECURSIVE="--recursive"   # set to "" to disable
SKIP_EXISTING="--skip-existing"
DEFAULT_CAPTION=""        # fallback when caption column is empty
CAPTION_PREFIX=""         # prefix for each caption
USE_INSTRUCTION_WHEN_EMPTY=""  # set to "--use-instruction-when-empty" if needed
DEBUG_ADAPTER=""  # set to "" to disable
CPU_OFFLOAD=""            # set to "--enable-model-cpu-offload" if needed
MAX_IMAGES=""             # e.g. "--max-images 10"

if [ -z "${ADAPTER_DIR}" ]; then
  ADAPTER_ARGS=()
else
  ADAPTER_ARGS=(--adapter-dir "${ADAPTER_DIR}")
fi

if [ -z "${BASE_PATH}" ]; then
  BASE_PATH_ARGS=()
else
  BASE_PATH_ARGS=(--base-path "${BASE_PATH}")
fi

python scripts/inference_stage1_folder.py \
  --metadata-path "${METADATA_PATH}" \
  --input-dir "${INPUT_DIR}" \
  "${BASE_PATH_ARGS[@]}" \
  --image-key "${IMAGE_KEY}" \
  --caption-key "${CAPTION_KEY}" \
  --default-caption "${DEFAULT_CAPTION}" \
  --caption-prefix "${CAPTION_PREFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RECURSIVE} \
  ${SKIP_EXISTING} \
  ${MAX_IMAGES} \
  --resize-mode "${RESIZE_MODE}" \
  --size "${SIZE}" \
  --flux-path "${FLUX_PATH}" \
  --lora-path "${LORA_PATH}" \
  "${ADAPTER_ARGS[@]}" \
  --prompt-prefix "${PROMPT_PREFIX}" \
  --instruction "${INSTRUCTION}" \
  --seed "${SEED}" \
  --seed-mode "${SEED_MODE}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --num-inference-steps "${STEPS}" \
  ${USE_INSTRUCTION_WHEN_EMPTY} \
  ${DEBUG_ADAPTER} \
  ${CPU_OFFLOAD}
