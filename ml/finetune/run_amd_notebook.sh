#!/usr/bin/env bash
# DisasterIQ fine-tune on an AMD Instinct GPU (ROCm) — hackathon notebook.
#
#   bash ml/finetune/run_amd_notebook.sh              # evidence + data + loc + dmg + eval
#   bash ml/finetune/run_amd_notebook.sh --stage prep # everything except training
#   bash ml/finetune/run_amd_notebook.sh --stage dmg  # damage only (needs a loc ckpt)
#
# Data (pick one, checked in this order):
#   DATA_DIR=/path/to/train_subset   already-staged images/ + targets/
#   DATA_URL=https://.../subset.tar.gz  any direct link to the subset archive
#   KAGGLE_USERNAME + KAGGLE_KEY + KAGGLE_DATASET=<owner>/<slug>
#
# Everything it does is written to $WORK_ROOT/amd_evidence/ as it goes. That
# directory is the artifact the judges need: a project that does not
# demonstrate AMD compute usage is disqualified, and "we intended to" is not a
# demonstration. Commit it (docs/AMD_COMPUTE.md explains how).
set -euo pipefail

STAGE="all"
if [[ "${1:-}" == "--stage" ]]; then
  STAGE="${2:-all}"
  shift 2 || true
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
FINETUNE_DIR="$REPO_ROOT/ml/finetune"
export WORK_ROOT="${WORK_ROOT:-$HOME/disasteriq-work}"
EVIDENCE_DIR="$WORK_ROOT/amd_evidence"

export FINETUNE_CONFIG="${FINETUNE_CONFIG:-$FINETUNE_DIR/config_subset_amd.yaml}"

mkdir -p "$WORK_ROOT/data" "$EVIDENCE_DIR"

# ---------------------------------------------------------------- AMD evidence
# Capture the hardware *before* anything can fail. A crashed training run on a
# real MI300X still proves AMD compute; a perfect run with no record proves
# nothing.
echo "=== AMD GPU / ROCm check ==="
{
  echo "# Captured $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo
  echo "## rocm-smi"
  rocm-smi 2>&1 || echo "(rocm-smi unavailable)"
  echo
  echo "## rocminfo (agent names)"
  rocminfo 2>/dev/null | grep -iE "Name:|Marketing" | head -20 || echo "(rocminfo unavailable)"
} | tee "$EVIDENCE_DIR/rocm-smi.txt"

python3 - <<'PY' | tee "$EVIDENCE_DIR/torch_device.txt"
import json, platform
import torch

# On ROCm, torch keeps the CUDA API surface — torch.cuda talks to HIP. A device
# name like "AMD Instinct MI300X" (gfx942) is what makes this AMD compute.
info = {
    "torch": torch.__version__,
    "hip": getattr(torch.version, "hip", None),
    "cuda": getattr(torch.version, "cuda", None),
    "gpu_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "devices": [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else [],
    "python": platform.python_version(),
}
print(json.dumps(info, indent=2))
PY

if [[ "$STAGE" != "prep" ]]; then
  python3 - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(
        "ERROR: no GPU visible to torch. In the AMD hackathon notebook this "
        "usually means the kernel started before the GPU attached — restart the "
        "kernel and re-run. Verify with `rocm-smi` in a terminal cell first."
    )
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}")
if "AMD" not in name and "Instinct" not in name and "gfx" not in name.lower():
    print(f"WARNING: {name!r} does not look like an AMD Instinct part.", file=sys.stderr)
    print("Training here will NOT satisfy the AMD compute requirement.", file=sys.stderr)
PY
fi

# ------------------------------------------------------------------- upstream
if [[ ! -d "$REPO_ROOT/ml/pytorch-xview2" ]]; then
  echo "=== Clone upstream xView2 (michal2409) ==="
  git clone --depth 1 https://github.com/michal2409/xView2.git "$REPO_ROOT/ml/pytorch-xview2"
fi

# ----------------------------------------------------------------------- data
DEST_SUBSET="$WORK_ROOT/data/train_subset"

if [[ -n "${DATA_DIR:-}" && -d "${DATA_DIR:-}/images" ]]; then
  echo "=== Using pre-staged DATA_DIR=$DATA_DIR ==="
  DEST_SUBSET="$DATA_DIR"
elif [[ -d "$DEST_SUBSET/images" ]]; then
  echo "=== Reusing staged subset at $DEST_SUBSET ==="
elif [[ -n "${DATA_URL:-}" ]]; then
  echo "=== Downloading subset from DATA_URL ==="
  ARCHIVE="$WORK_ROOT/data/subset_archive"
  curl -fL --retry 3 -o "$ARCHIVE" "$DATA_URL"
  mkdir -p "$DEST_SUBSET"
  case "$DATA_URL" in
    *.zip) unzip -q -o "$ARCHIVE" -d "$DEST_SUBSET" ;;
    *)     tar -xf "$ARCHIVE" -C "$DEST_SUBSET" ;;
  esac
  rm -f "$ARCHIVE"
elif [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" && -n "${KAGGLE_DATASET:-}" ]]; then
  echo "=== Downloading $KAGGLE_DATASET via Kaggle API ==="
  pip install -q kaggle
  mkdir -p "$DEST_SUBSET"
  kaggle datasets download -d "$KAGGLE_DATASET" -p "$WORK_ROOT/data" --unzip
  # The archive may unpack a level deep; find the images/ + targets/ root.
  FOUND="$(python3 - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["WORK_ROOT"]) / "data"
for images in root.rglob("images"):
    if (images.parent / "targets").is_dir():
        print(images.parent)
        break
PY
)"
  [[ -n "$FOUND" ]] && DEST_SUBSET="$FOUND"
else
  cat >&2 <<'MSG'
ERROR: no training data. Provide exactly one of:
  DATA_DIR=/path/to/train_subset        (already on disk: images/ + targets/)
  DATA_URL=https://.../subset.tar.gz    (direct link — simplest in a notebook)
  KAGGLE_USERNAME=... KAGGLE_KEY=... KAGGLE_DATASET=owner/slug
MSG
  exit 1
fi

if [[ ! -d "$DEST_SUBSET/images" || ! -d "$DEST_SUBSET/targets" ]]; then
  echo "ERROR: $DEST_SUBSET has no images/ + targets/ — wrong archive layout?" >&2
  exit 1
fi

export DATA_DIR="$DEST_SUBSET"
echo "Training data: $DATA_DIR ($(find "$DATA_DIR/images" -name '*pre*' | wc -l) pre images)"

# Disjoint train/holdout split. Without this, validation F1 is train-set
# leakage and the number you would put on a slide is a lie.
python3 "$FINETUNE_DIR/data_layout.py" "$DATA_DIR"

echo "=== Patch upstream xView2 for subset training ==="
python3 "$FINETUNE_DIR/patch_pytorch_xview2.py"

INDEX_OUT="$REPO_ROOT/ml/pytorch-xview2/utils/index.csv"
export XVIEW2_INDEX_CSV="$INDEX_OUT"
rm -f "$INDEX_OUT"
python3 "$REPO_ROOT/scripts/generate_subset_index.py" --data-dir "$DATA_DIR" --out "$INDEX_OUT"

echo "=== CPU dataset smoke test ==="
python3 "$REPO_ROOT/scripts/test_pytorch_dataset.py" --data-dir "$DATA_DIR"

eval "$(python3 "$FINETUNE_DIR/load_config.py" --config "$FINETUNE_CONFIG" data)"
mkdir -p "$RESULTS_ROOT"

if [[ "$STAGE" == "prep" ]]; then
  echo "Prep complete — data staged, no training. Evidence: $EVIDENCE_DIR"
  exit 0
fi

# ------------------------------------------------------------------- training
if [[ "$STAGE" == "all" || "$STAGE" == "loc" ]]; then
  echo "=== Stage 1: localization (AMD GPU) ==="
  DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/loc" FINETUNE_CONFIG="$FINETUNE_CONFIG" \
    bash "$FINETUNE_DIR/train_localization.sh" 2>&1 | tee "$EVIDENCE_DIR/train_loc.log"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "dmg" ]]; then
  echo "=== Stage 2: damage fine-tune (AMD GPU) ==="
  DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/dmg" FINETUNE_CONFIG="$FINETUNE_CONFIG" \
    bash "$FINETUNE_DIR/train_damage.sh" 2>&1 | tee "$EVIDENCE_DIR/train_dmg.log"
fi

# The app loads exactly this path (settings.pytorch_checkpoint_path).
BEST="$RESULTS_ROOT/dmg/checkpoints/best.ckpt"
if [[ -f "$BEST" ]]; then
  mkdir -p "$REPO_ROOT/ml/checkpoints"
  cp "$BEST" "$REPO_ROOT/ml/checkpoints/damage_best.ckpt"
  sha256sum "$REPO_ROOT/ml/checkpoints/damage_best.ckpt" | tee "$EVIDENCE_DIR/checkpoint_sha256.txt"
  echo "Checkpoint -> ml/checkpoints/damage_best.ckpt"
  echo "Serve it with:  INFERENCE_MODE=pytorch"
else
  echo "WARNING: no best.ckpt at $BEST — training did not complete." >&2
fi

echo
echo "AMD evidence written to: $EVIDENCE_DIR"
ls -la "$EVIDENCE_DIR"
