#!/usr/bin/env bash

# Batch inference for a folder of images.
# Usage: ./run_inference_folder.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------
# Required configuration
# -----------------------

INPUT_DIR="/root/autodl-tmp/MSRS/test/vi"
OUTPUT_DIR="./msrs_adapter_128lora_kontype_1w_gra50_28steps"

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
DEBUG_ADAPTER=""  # set to "" to disable
CPU_OFFLOAD=""            # set to "--enable-model-cpu-offload" if needed
MAX_IMAGES=""             # e.g. "--max-images 10"

if [ -z "${ADAPTER_DIR}" ]; then
  ADAPTER_ARGS=()
else
  ADAPTER_ARGS=(--adapter-dir "${ADAPTER_DIR}")
fi

python scripts/inference_folder.py \
  --input-dir "${INPUT_DIR}" \
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
  ${DEBUG_ADAPTER} \
  ${CPU_OFFLOAD}
