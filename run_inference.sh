#!/usr/bin/env bash

# ICEdit inference launcher (LoRA + optional adapter-dir).
# Usage: ./run_inference.sh

set -euo pipefail

# Resolve repo root as the directory of this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------
# Required configuration
# -----------------------

# Input image (visible/light image).
IMAGE_PATH="assets/girl.png"

# Prompt template (should match what you used for training).
PROMPT_PREFIX="A diptych with two side-by-side images of the same scene. The left image is visible light. The right image is the corresponding infrared thermal image. Instruction: "

# Instruction text (the fixed training instruction). You may set this to "" if you
# already appended the full instruction into PROMPT_PREFIX.
INSTRUCTION="Transform the visible image into its infrared thermal counterpart while preserving scene structure and objects."

# Flux base model path or HF id.
FLUX_PATH="/root/autodl-tmp/qyt/hfd_model"

# LoRA weights path (either a directory saved by diffusers, or a specific file if supported).
LORA_PATH="/root/autodl-tmp/qyt_adapter/ice_adapter_2/ICEdit_adapter/train/runs/20251214-001544/ckpt/2"

# Adapter checkpoint directory (should contain `text_token_adapter.pt` and/or `prompt_adapter.pt`).
ADAPTER_DIR="/root/autodl-tmp/qyt_adapter/ice_adapter_2/ICEdit_adapter/train/runs/20251214-001544/ckpt/2"

# Output directory.
OUTPUT_DIR="./outputs"

# Reproducibility.
SEED=42

# -----------------------
# Optional flags
# -----------------------
ENABLE_CPU_OFFLOAD=""          # set to "--enable-model-cpu-offload" if needed
DEBUG_ADAPTER="--debug-adapter" # set to "" to disable

# -----------------------
# Checks
# -----------------------
if [ ! -f "$IMAGE_PATH" ]; then
  echo "Error: image file not found: $IMAGE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "ICEdit Inference Config"
echo "=================================="
echo "Image:        $IMAGE_PATH"
echo "PromptPrefix: $PROMPT_PREFIX"
echo "Instruction:  $INSTRUCTION"
echo "Flux:         $FLUX_PATH"
echo "LoRA:         $LORA_PATH"
echo "AdapterDir:   $ADAPTER_DIR"
echo "OutputDir:    $OUTPUT_DIR"
echo "Seed:         $SEED"
echo "=================================="
echo

python scripts/inference.py \
  --image "$IMAGE_PATH" \
  --prompt-prefix "$PROMPT_PREFIX" \
  --instruction "$INSTRUCTION" \
  --flux-path "$FLUX_PATH" \
  --lora-path "$LORA_PATH" \
  --adapter-dir "$ADAPTER_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  $DEBUG_ADAPTER \
  $ENABLE_CPU_OFFLOAD

echo
echo "Done. Results saved to: $OUTPUT_DIR"

