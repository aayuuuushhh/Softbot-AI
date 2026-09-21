#!/usr/bin/env bash
# DisasterIQ fine-tune pipeline for Kaggle Notebooks (CUDA GPU)
# Usage (from repo root on Kaggle):
#   bash ml/finetune/run_kaggle_pipeline.sh
#   bash ml/finetune/run_kaggle_pipeline.sh --stage loc   # localization only
#   bash ml/finetune/run_kaggle_pipeline.sh --stage dmg   # damage only (needs loc ckpt)
set -euo pipefail

STAGE="all"
if [[ "${1:-}" == "--stage" ]]; then
  STAGE="${2:-all}"
  shift 2 || true
elif [[ "${1:-}" == "--prep-only" ]]; then
  STAGE="prep"
  shift
elif [[ "${1:-}" == "--train-only" ]]; then
  STAGE="train"
  shift
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
FINETUNE_DIR="$REPO_ROOT/ml/finetune"
WORKING="${KAGGLE_WORKING:-/kaggle/working}"
INPUT_ROOT="${KAGGLE_INPUT:-/kaggle/input}"

export FINETUNE_CONFIG="${FINETUNE_CONFIG:-$FINETUNE_DIR/config_subset_kaggle.yaml}"

require_gpu() {
  python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: CUDA not available. Kaggle: Settings → Accelerator → GPU, restart, re-run." >&2
    exit 1
  }
}

find_train_subset() {
  local candidate
  for candidate in \
    "$WORKING/data/train_subset" \
    "$INPUT_ROOT/disasteriq-train-subset" \
    "$INPUT_ROOT"/disasteriq-train-subset/train_subset \
    "$INPUT_ROOT"/*/train_subset \
    "$INPUT_ROOT"/*/*/*/train_subset \
    "$INPUT_ROOT"/*/
  do
    [[ -d "$candidate/images" && -d "$candidate/targets" ]] || continue
    echo "$candidate"
    return 0
  done
  # Fallback: walk /kaggle/input for any train_subset layout
  python3 - <<'PY'
from pathlib import Path
root = Path("/kaggle/input")
for images in root.rglob("images"):
    parent = images.parent
    if (parent / "targets").is_dir():
        print(parent)
        break
PY
}

echo "=== CUDA GPU check ==="
if [[ "$STAGE" != "prep" ]]; then
  python3 -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
  require_gpu
else
  python3 -c "import torch; print('cuda (prep-only):', torch.cuda.is_available())" || true
fi

if [[ ! -d "$REPO_ROOT/ml/pytorch-xview2" ]]; then
  echo "Missing ml/pytorch-xview2 — clone michal2409/xView2 first."
  exit 1
fi

echo "=== Stage training data (merge train_subset + tier3 if both attached) ==="
DEST_SUBSET="$WORKING/data/train_subset"
if [[ -d "$WORKING/data/combined_subset/images" ]]; then
  DEST_SUBSET="$WORKING/data/combined_subset"
else
  python3 "$REPO_ROOT/scripts/stage_kaggle_data.py" \
    --input-root "$INPUT_ROOT" \
    --dest "$WORKING/data/train_subset" \
    --combined-dest "$WORKING/data/combined_subset" || {
    SRC_SUBSET="$(find_train_subset)"
    if [[ -z "$SRC_SUBSET" || ! -d "$SRC_SUBSET/images" ]]; then
      echo "ERROR: Could not find train_subset under $INPUT_ROOT" >&2
      exit 1
    fi
    mkdir -p "$WORKING/data"
    rm -rf "$DEST_SUBSET"
    cp -a "$SRC_SUBSET" "$DEST_SUBSET"
  }
  if [[ -d "$WORKING/data/combined_subset/images" ]]; then
    DEST_SUBSET="$WORKING/data/combined_subset"
  fi
fi
export DATA_DIR="$DEST_SUBSET"

ensure_xview2_layout() {
  local root="$1"
  [[ -d "$root/images" ]] || return 0
  python3 "$FINETUNE_DIR/data_layout.py" "$root"
}
ensure_xview2_layout "$DEST_SUBSET"

echo "=== Patch upstream xView2 for subset training ==="
python3 "$FINETUNE_DIR/patch_pytorch_xview2.py"

INDEX_OUT="$REPO_ROOT/ml/pytorch-xview2/utils/index.csv"
export XVIEW2_INDEX_CSV="$INDEX_OUT"

echo "=== Generate index.csv for $DATA_DIR ==="
# Do not trust a pre-existing index.csv (xView2 clone often has full-train ~8k rows).
# Validate max(idx) against this data_dir; regenerate when stale.
NEED_INDEX=1
N_PRE=$(find "$DATA_DIR/images" -name '*pre*' 2>/dev/null | wc -l | tr -d ' ')
if [[ -f "$INDEX_OUT" ]] && [[ "$(wc -l < "$INDEX_OUT")" -gt 1 ]] && [[ "${N_PRE:-0}" -gt 0 ]]; then
  MAX_IDX=$(python3 -c "import pandas as pd; df=pd.read_csv('$INDEX_OUT'); print(int(df['idx'].max()) if len(df) else -1)")
  if [[ "$MAX_IDX" -ge 0 ]] && [[ "$MAX_IDX" -lt "$N_PRE" ]]; then
    echo "index.csv OK (max_idx=$MAX_IDX < n_pre=$N_PRE) — keeping"
    NEED_INDEX=0
  else
    echo "Stale index.csv (max_idx=$MAX_IDX, n_pre=$N_PRE) — regenerating"
    rm -f "$INDEX_OUT"
  fi
fi
if [[ "$NEED_INDEX" -eq 1 ]]; then
  python3 "$REPO_ROOT/scripts/generate_subset_index.py" \
    --data-dir "$DATA_DIR" \
    --out "$INDEX_OUT"
fi

if [[ "$STAGE" != "train" ]]; then
  echo "=== CPU dataset smoke test ==="
  python3 "$REPO_ROOT/scripts/test_pytorch_dataset.py" --data-dir "$DATA_DIR"
fi

eval "$(python3 "$FINETUNE_DIR/load_config.py" --config "$FINETUNE_CONFIG" data)"
mkdir -p "$RESULTS_ROOT"

if [[ "$STAGE" == "prep" ]]; then
  echo "Prep complete (no training)."
  exit 0
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "loc" ]]; then
  echo "=== Stage 1: localization ==="
  DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/loc" \
    FINETUNE_CONFIG="$FINETUNE_CONFIG" \
    bash "$FINETUNE_DIR/train_localization.sh"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "dmg" ]]; then
  echo "=== Stage 2: damage fine-tune ==="
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
  LOC_CKPT="$RESULTS_ROOT/loc/checkpoints/best.ckpt"
  if ! LOC_CKPT="$(resolve_ckpt "$LOC_CKPT")"; then
    echo "ERROR: Missing localization checkpoint under $RESULTS_ROOT/loc/checkpoints" >&2
    exit 1
  fi
  echo "Using localization checkpoint: $LOC_CKPT"
  DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/dmg" \
    CKPT_PRE="$LOC_CKPT" \
    FINETUNE_CONFIG="$FINETUNE_CONFIG" \
    bash "$FINETUNE_DIR/train_damage.sh"
fi

export_damage_ckpt() {
  local dmg_dir="$RESULTS_ROOT/dmg/checkpoints"
  local out="$WORKING/damage_best.ckpt"
  for candidate in "$dmg_dir/best.ckpt" "$dmg_dir/last.ckpt" "$dmg_dir"/*.ckpt; do
    [[ -f "$candidate" ]] || continue
    cp -f "$candidate" "$out"
    echo "Exported checkpoint -> $out (from $(basename "$candidate"))"
    ls -la "$out"
    return 0
  done
  echo "WARN: No damage checkpoint under $dmg_dir"
  return 1
}

if [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "dmg" ]]; then
  export_damage_ckpt || true
fi

echo "Done."
