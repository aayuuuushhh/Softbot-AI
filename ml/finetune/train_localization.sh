#!/usr/bin/env bash
# Stage 1: building localization on hackathon train subset
set -euo pipefail
FINETUNE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FINETUNE_DIR/../pytorch-xview2"

CONFIG_ARGS=()
if [[ -n "${FINETUNE_CONFIG:-}" ]]; then
  CONFIG_ARGS=(--config "$FINETUNE_CONFIG")
fi
eval "$(python3 "$FINETUNE_DIR/load_config.py" "${CONFIG_ARGS[@]}" localization)"

DATA_DIR="${DATA_DIR:-/data/train_subset}"
RESULTS_DIR="${RESULTS_DIR:-/results/loc}"

mkdir -p "$RESULTS_DIR/checkpoints"

python main.py \
  --exec_mode train \
  --type pre \
  --data "$DATA_DIR" \
  --results "$RESULTS_DIR" \
  --encoder "$ENCODER" \
  --loss_str ce+dice \
  --deep_supervision \
  --gpus 1 \
  --num_workers "${NUM_WORKERS:-4}" \
  --batch_size "$BATCH_SIZE" \
  --val_batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Localization training done -> $RESULTS_DIR"
