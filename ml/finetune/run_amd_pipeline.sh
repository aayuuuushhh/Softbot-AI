#!/usr/bin/env bash
# Bootstrap AMD GPU instance and run full fine-tune pipeline
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
FINETUNE_DIR="$REPO_ROOT/ml/finetune"

echo "=== ROCm GPU check ==="
rocm-smi || true
python3 -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')" || echo "WARN: torch not importable"

if [[ ! -d "$REPO_ROOT/ml/pytorch-xview2" ]]; then
  echo "Missing ml/pytorch-xview2 — clone michal2409/xView2 first."
  exit 1
fi

echo "=== Patch upstream xView2 for subset training ==="
python3 "$FINETUNE_DIR/patch_pytorch_xview2.py"

export DATA_DIR="${DATA_DIR:-/data/train_subset}"
export RESULTS_ROOT="${RESULTS_ROOT:-/results}"
INDEX_OUT="$REPO_ROOT/ml/pytorch-xview2/utils/index.csv"
export XVIEW2_INDEX_CSV="$INDEX_OUT"

echo "=== Generate index.csv for $DATA_DIR ==="
python3 "$REPO_ROOT/scripts/generate_subset_index.py" \
  --data-dir "$DATA_DIR" \
  --out "$INDEX_OUT"

echo "=== CPU dataset smoke test ==="
python3 "$REPO_ROOT/scripts/test_pytorch_dataset.py" --data-dir "$DATA_DIR"

mkdir -p "$RESULTS_ROOT"
cd "$REPO_ROOT/ml/pytorch-xview2"

echo "=== Stage 1: localization ==="
DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/loc" \
  bash "$FINETUNE_DIR/train_localization.sh"

echo "=== Stage 2: damage fine-tune ==="
DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/dmg" \
  CKPT_PRE="$RESULTS_ROOT/loc/checkpoints/best.ckpt" \
  bash "$FINETUNE_DIR/train_damage.sh"

echo "=== Eval on test set ==="
eval "$(python3 "$FINETUNE_DIR/load_config.py" data)"
python main.py --exec_mode eval --type post \
  --ckpt "$RESULTS_ROOT/dmg/checkpoints/best.ckpt" \
  --data "${TEST_DIR:-/data/test}" --results "$RESULTS_ROOT/eval" \
  --gpus 1 --val_batch_size 4

echo "Done. Checkpoints:"
ls -la "$RESULTS_ROOT/dmg/checkpoints/" || true
