#!/usr/bin/env bash

# usage: ./train_thermal_trainable.sh [CONFIG_FILE] [PORT]
# example: ./train_thermal_trainable.sh paired_csv_lora_thermal_trainable.yaml 41353

set -euo pipefail

CONFIG_FILE="${1:-paired_csv_lora_thermal_trainable.yaml}"
PORT="${2:-41353}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find `<repo>/train` by walking upwards until `src/` is found.
TRAIN_ROOT="${SCRIPT_DIR}"
while [ "${TRAIN_ROOT}" != "/" ] && [ ! -d "${TRAIN_ROOT}/src" ]; do
  TRAIN_ROOT="$(dirname "${TRAIN_ROOT}")"
done
if [ ! -d "${TRAIN_ROOT}/src" ]; then
  echo "Error: could not locate training root containing ./src (starting from ${SCRIPT_DIR})." >&2
  exit 1
fi

CONFIG_DIR="${TRAIN_ROOT}/train/config"                    # <repo>/train/train/config
CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"

export XFL_CONFIG="${CONFIG_PATH}"
echo "Using config: ${XFL_CONFIG}"

export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${TRAIN_ROOT}"

python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("src")
if spec is None:
    raise SystemExit("Error: Python cannot find top-level module `src`. Check PYTHONPATH and working directory.")
print("Python module check OK: src found")
PY

accelerate launch --main_process_port "${PORT}" -m src.train.train
