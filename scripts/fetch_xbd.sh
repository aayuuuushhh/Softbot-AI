#!/usr/bin/env bash
# Fetch xBD tile pairs (images + polygon labels) from the aryananand/xBD HuggingFace mirror.
#
#   bash scripts/fetch_xbd.sh                       # the four events we use
#   bash scripts/fetch_xbd.sh hurricane-matthew     # add another event
#
# Why not huggingface-cli / snapshot_download: that repo holds 22,396 loose files, and the
# HF tree API stalls indefinitely trying to list them. This fetches by direct URL instead.
#
# Why -P 6: HuggingFace answers 429 above roughly this. Wider is not faster, it just fails.
#
# Resumable - already-present, non-empty files are skipped. Safe to re-run after an interrupt.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/raw/xbd"
BASE="https://huggingface.co/datasets/aryananand/xBD/resolve/main"
PARALLEL="${XBD_PARALLEL:-6}"
MAX_INDEX="${XBD_MAX_INDEX:-800}"

EVENTS=("$@")
if [ ${#EVENTS[@]} -eq 0 ]; then
  EVENTS=(hurricane-michael palu-tsunami santa-rosa-wildfire mexico-earthquake)
fi

mkdir -p "$DEST"/{train,test}/{images,labels}
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- probe: which tile indices exist, in which split ------------------------
cat > "$WORK/probe.sh" <<'PROBE'
#!/bin/bash
code=$(curl -sS -m 20 -o /dev/null -w "%{http_code}" \
  "$BASE/$2/labels/$(printf '%s_%08d' "$1" "$3")_post_disaster.json")
case "$code" in 200|302|307) echo "$1 $2 $3";; esac
PROBE

# --- fetch: pre+post, image+label, for one tile -----------------------------
cat > "$WORK/fetch.sh" <<'FETCH'
#!/bin/bash
stem=$(printf '%s_%08d' "$1" "$3")
for phase in pre post; do
  for spec in labels:json images:png; do
    dir=${spec%%:*}; ext=${spec##*:}
    out="$DEST/$2/$dir/${stem}_${phase}_disaster.$ext"
    [ -s "$out" ] && continue
    curl -sSL -m 240 -f --retry 6 --retry-delay 3 --retry-all-errors \
      -o "$out" "$BASE/$2/$dir/${stem}_${phase}_disaster.$ext" || rm -f "$out"
  done
done
FETCH

chmod +x "$WORK/probe.sh" "$WORK/fetch.sh"
export BASE DEST

for event in "${EVENTS[@]}"; do
  echo "==> $event: probing 0..$MAX_INDEX"
  for split in train test; do
    for i in $(seq 0 "$MAX_INDEX"); do echo "$event $split $i"; done
  done | xargs -P 24 -n 3 "$WORK/probe.sh" >> "$WORK/found.txt"

  n=$(grep -c "^$event " "$WORK/found.txt" || true)
  echo "==> $event: $n tiles found"
done

total=$(wc -l < "$WORK/found.txt")
echo "==> downloading $total tile pairs ($((total * 4)) files) at -P $PARALLEL"
xargs -P "$PARALLEL" -n 3 "$WORK/fetch.sh" < "$WORK/found.txt"

# Re-run once: anything the rate limiter dropped gets picked up, existing files are skipped.
echo "==> sweeping for gaps"
xargs -P 4 -n 3 "$WORK/fetch.sh" < "$WORK/found.txt"

echo "==> verifying"
python3 - "$DEST" <<'VERIFY'
import glob, json, sys
from PIL import Image
root = sys.argv[1]
bad = []
for p in glob.glob(f"{root}/*/images/*.png"):
    try: Image.open(p).verify()
    except Exception: bad.append(p)
for p in glob.glob(f"{root}/*/labels/*.json"):
    try: json.load(open(p))
    except Exception: bad.append(p)
imgs = len(glob.glob(f"{root}/*/images/*.png"))
lbls = len(glob.glob(f"{root}/*/labels/*.json"))
print(f"    {imgs} images, {lbls} labels, {len(bad)} corrupt")
for p in bad:
    print(f"    CORRUPT {p}")
if bad:
    print("    delete those and re-run this script to repair")
    sys.exit(1)
VERIFY

echo "==> done. Next: python scripts/train_satellite.py --xbd-root data/raw/xbd"
