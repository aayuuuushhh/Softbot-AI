#!/usr/bin/env bash
# Fetches everything that can be fetched without a login.
#
# OSM layers come from the Overpass API via scripts/fetch_osm.py - no 200 MB Geofabrik
# extract, no GDAL. xBD still needs manual registration; see data/README.md.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/raw data/nepal

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

echo "[1/2] OpenStreetMap: ward boundaries, critical facilities, building footprints"
"$PY" scripts/fetch_osm.py --all

echo
echo "[2/2] WorldPop population raster (Nepal, constrained, 100 m)"
if [[ -s data/nepal/worldpop.tif ]]; then
  echo "  already have data/nepal/worldpop.tif"
else
  curl -fL --progress-bar \
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/NPL/npl_ppp_2020_UNadj_constrained.tif" \
    -o data/nepal/worldpop.tif
fi

echo
echo "Done. Check what is present:  $PY scripts/run_pipeline.py --check"
echo
echo "Remaining manual step - xBD imagery + damage labels -> data/raw/xbd/"
echo "  https://xview2.org/dataset   (CC BY-NC-SA 4.0, non-commercial)"
