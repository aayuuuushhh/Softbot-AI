#!/usr/bin/env bash
# Pack everything the training box needs into one archive, and nothing it doesn't.
#
#   bash scripts/make_training_bundle.sh                    # -> data/xbd_training_bundle.tar
#   bash scripts/make_training_bundle.sh /media/usb         # write it somewhere else
#
# Contents: the chips, the three manifests, and the code needed to train. Not the 3.3 GB of
# raw 1024x1024 tiles - chip extraction already happened, and the training box never reads
# them. Not the Nepal GIS layers either; those are for scoring, which happens back here.
#
# The manifests store chip paths relative to the repo root, so they resolve on any machine
# as long as the bundle is unpacked at the root of a clone of this repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$REPO_ROOT/data}"
OUT="$OUT_DIR/xbd_training_bundle.tar"

cd "$REPO_ROOT"

for required in data/processed/manifest.csv data/processed/chips; do
  if [ ! -e "$required" ]; then
    echo "missing $required - run: python scripts/run_pipeline.py --prepare" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

echo "==> packing"
tar -cf "$OUT" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' \
  data/processed/chips \
  data/processed/manifest.csv \
  data/processed/manifest_test.csv \
  data/processed/manifest_holdout.csv \
  ml/ config/paths.yaml

echo "==> checksum"
# Written with the bare filename, not the absolute path: `sha256sum -c` on the other machine
# looks for whatever path is recorded here, and this box's paths do not exist over there.
( cd "$OUT_DIR" && sha256sum "$(basename "$OUT")" | tee "$(basename "$OUT").sha256" )

cat <<EOF

==> $(du -h "$OUT" | cut -f1)  $OUT

On the training machine:

  git clone <this repo> && cd softbot
  tar -xf xbd_training_bundle.tar
  sha256sum -c xbd_training_bundle.tar.sha256     # before unpacking, ideally

  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -r ml/requirements.txt

  # smoke test first - 3 epochs, head only, a few minutes. Confirm macro-F1 moves off zero.
  .venv/bin/python -m ml.train --freeze-backbone --epochs 3

  # then the real run
  .venv/bin/python -m ml.train --epochs 8

Bring back two files:
  ml/checkpoints/siamese_resnet18.pt
  ml/checkpoints/siamese_resnet18.metrics.json

Then here:  python -m ml.evaluate
EOF
