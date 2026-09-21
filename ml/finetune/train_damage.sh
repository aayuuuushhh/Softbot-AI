#!/usr/bin/env bash
# Stage 2: damage classification fine-tune (siamese U-Net)
set -euo pipefail
FINETUNE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FINETUNE_DIR/../pytorch-xview2"

CONFIG_ARGS=()
if [[ -n "${FINETUNE_CONFIG:-}" ]]; then
  CONFIG_ARGS=(--config "$FINETUNE_CONFIG")
fi
_CKPT_PRE_OVERRIDE="${CKPT_PRE:-}"
eval "$(python3 "$FINETUNE_DIR/load_config.py" "${CONFIG_ARGS[@]}" damage)"
if [[ -n "$_CKPT_PRE_OVERRIDE" ]]; then
  CKPT_PRE="$_CKPT_PRE_OVERRIDE"
fi

DATA_DIR="${DATA_DIR:-/data/train_subset}"
RESULTS_DIR="${RESULTS_DIR:-/results/dmg}"
CKPT_PRE="${CKPT_PRE:-/results/loc/checkpoints/best.ckpt}"

resolve_ckpt() {
  local ckpt="$1"
  local dir
  dir="$(dirname "$ckpt")"
  if [[ -f "$ckpt" ]]; then
    echo "$ckpt"
    return 0
  fi
  if [[ -f "$dir/last.ckpt" ]]; then
    echo "$dir/last.ckpt"
    return 0
  fi
  local latest
  latest="$(ls -t "$dir"/*.ckpt 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" ]]; then
    echo "$latest"
    return 0
  fi
  return 1
}

if ! CKPT_PRE="$(resolve_ckpt "$CKPT_PRE")"; then
  echo "Missing localization checkpoint: $CKPT_PRE"
  echo "Run train_localization.sh first."
  exit 1
fi
echo "Using localization checkpoint: $CKPT_PRE"

mkdir -p "$RESULTS_DIR/checkpoints"

python main.py \
  --exec_mode train \
  --type post \
  --dmg_model siamese \
  --data "$DATA_DIR" \
  --results "$RESULTS_DIR" \
  --encoder "$ENCODER" \
  --loss_str focal+dice \
  --ckpt_pre "$CKPT_PRE" \
  --attention \
  --deep_supervision \
  --gpus 1 \
  --num_workers "${NUM_WORKERS:-4}" \
  --batch_size "$BATCH_SIZE" \
  --val_batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Damage fine-tune done -> $RESULTS_DIR"
