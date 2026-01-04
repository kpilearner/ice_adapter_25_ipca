#!/usr/bin/env bash

# Batch inference using dual LoRA weights (stage-1 + phys).
# Usage: ./run_inference_from_csv_dual_lora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------
# Required configuration
# -----------------------

METADATA_PATH="/root/autodl-tmp/MSRS/metadata.csv"
BASE_PATH="/root/autodl-tmp/MSRS"
IMAGE_KEY="kontext_images"
CAPTION_KEY="prompt"

OUTPUT_DIR="./msrs_dual_lora_infer"

FLUX_PATH="/root/autodl-tmp/model/hfd_model"
LORA_PATH_1="/root/autodl-tmp/runs_msrs_text_stage1/ckpt/10000"
LORA_PATH_2="/root/autodl-tmp/runs_msrs_phys_stage2/ckpt/10000/lora_phys"
LORA_NAME_1="default"
LORA_NAME_2="phys"
LORA_SCALE_1=1.0
LORA_SCALE_2=1.0

ADAPTER_DIR="/root/autodl-tmp/runs_msrs_text_stage1/ckpt/10000"   # set to "" to disable

PROMPT_PREFIX="A diptych with two side-by-side images of the same scene. The left image is visible light. The right image is the corresponding infrared thermal image. Instruction: "
INSTRUCTION="Transform the visible image into its infrared thermal counterpart while preserving scene structure and objects."

# -----------------------
# Optional configuration
# -----------------------
SEED=42
SEED_MODE="hash"          # fixed | incremental | hash
GUIDANCE_SCALE=50
STEPS=28
RESIZE_MODE="square"      # square | width
SIZE=512
SKIP_EXISTING="--skip-existing"
DEBUG_ADAPTER=""          # set to "" to disable
CPU_OFFLOAD=""            # set to "--enable-model-cpu-offload" if needed
MAX_IMAGES=""             # e.g. "--max-images 10"
DEFAULT_CAPTION=""        # fallback when caption column is empty
CAPTION_PREFIX=""         # prefix for each caption
USE_INSTRUCTION_WHEN_EMPTY=""  # set to "--use-instruction-when-empty" to reuse instruction if caption empty

if [ -z "${ADAPTER_DIR}" ]; then
  ADAPTER_ARGS=()
else
  ADAPTER_ARGS=(--adapter-dir "${ADAPTER_DIR}")
fi

python scripts/inference_folder_from_csv_dual_lora.py \
  --metadata-path "${METADATA_PATH}" \
  --base-path "${BASE_PATH}" \
  --image-key "${IMAGE_KEY}" \
  --caption-key "${CAPTION_KEY}" \
  --default-caption "${DEFAULT_CAPTION}" \
  --caption-prefix "${CAPTION_PREFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  ${MAX_IMAGES} \
  ${SKIP_EXISTING} \
  --resize-mode "${RESIZE_MODE}" \
  --size "${SIZE}" \
  --flux-path "${FLUX_PATH}" \
  --lora-path-1 "${LORA_PATH_1}" \
  --lora-path-2 "${LORA_PATH_2}" \
  --lora-name-1 "${LORA_NAME_1}" \
  --lora-name-2 "${LORA_NAME_2}" \
  --lora-scale-1 "${LORA_SCALE_1}" \
  --lora-scale-2 "${LORA_SCALE_2}" \
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
