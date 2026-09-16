# AI Disaster Damage Assessment & Response System

A first-pass disaster assessment tool for Nepal. It takes pre- and post-disaster satellite
imagery, detects buildings, classifies damage severity with computer vision, and produces a
ward-level **Response Priority Score** (damage + population density + critical infrastructure +
accessibility) so emergency responders know where to go first.

Every AI result is reviewable and correctable by a human official before it informs a real
decision. The AI does the first pass; people decide.

```
pre/post imagery -> buildings -> damage classes -> ward priority -> dashboard
                                                          ^
                                           human review corrects the AI
```

**Stack:** Python 3.11 · PyTorch · FastAPI · GeoPandas/rasterio · React + Vite + Leaflet

---

## Quick start

The dashboard runs on synthetic demo data out of the box — no imagery or model needed.

```bash
# 1. Backend
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.app.main:app --reload     # http://localhost:8000/docs

# 2. Frontend (new terminal)
cd frontend && npm install && npm run dev                     # http://localhost:5173
```

Open <http://localhost:5173>. You get a Kathmandu Valley AOI with ~1,200 buildings across 8
wards, ranked by response priority. Click a ward to focus it; click a building to review and
correct the AI's call.

```bash
.venv/bin/python -m pytest              # 44 tests
.venv/bin/python scripts/run_pipeline.py --check    # what data is present, what is missing
```

## Current state

| Piece | Status |
|---|---|
| API contract, ward aggregation, priority scoring | **done**, unit-tested |
| Dashboard: map, ward cards, review drawer, export | **done** |
| Real Nepal GIS — 32 Kathmandu wards, 150k OSM buildings, 2.3k facilities, WorldPop | **done** |
| Heuristic damage classifier (fallback path) | **done**, unit-tested |
| Siamese CNN: model, dataset, training loop, inference | **written**, needs xBD data to train |
| Damage from real imagery | not yet — simulated from a shaking field; `mode=model` returns 501 |

The dashboard labels **two** things separately in the header, because they are separately
real: `geometry_source` (`OSM + WORLDPOP` vs `SYNTHETIC MAP`) and `mode` (`SIMULATED DAMAGE`
vs `HEURISTIC`/`MODEL`). Right now the map, the wards, the buildings, the facilities and the
population are real Kathmandu data; only the damage is simulated. Un-wired inference modes
return **501** rather than silently serving synthetic data — nobody should demo mock numbers
believing they came from a model.

Damage is simulated as a **shaking field** that decays with distance from an epicentre, not as
per-ward random numbers. That matters: the ward ranking emerges from geography, so moving the
epicentre re-ranks the wards. There is a test for exactly that.

```bash
# Move the earthquake and watch the priority list change
curl -X POST "localhost:8000/api/assessments/demo?epicentre_lon=85.29&epicentre_lat=27.69&radius_km=2"
```

## Getting real data in

```bash
bash scripts/download_data.sh                  # OSM + WorldPop (already fetched in this repo)
.venv/bin/python scripts/run_pipeline.py --prepare    # extract chips, print class balance
.venv/bin/python -m ml.train --epochs 8               # train the siamese classifier
.venv/bin/python scripts/run_pipeline.py --assess \
    --pre data/raw/pre.tif --post data/raw/post.tif \
    --footprints data/nepal/buildings.geojson --mode model
```

See **[data/README.md](./data/README.md)** for every dataset, its source and its license.

## Layout

```
backend/    FastAPI: routers, ward aggregation, priority scoring, exports, tests
ml/         siamese damage classifier: dataset, model, train, infer
frontend/   React + Leaflet dashboard
config/     priority.yaml (weights, bands, colors), paths.yaml
scripts/    download_data.sh, run_pipeline.py
data/       gitignored - see data/README.md
```

## Tuning the priority score

Everything lives in [`config/priority.yaml`](./config/priority.yaml) — component weights,
per-class damage weights, facility importance, band thresholds, map colors. Edit it and:

```bash
curl -X POST localhost:8000/api/config/reload
```

No restart. Raise the hospital weight and watch the ward ranking shift.

## Documentation

📄 **[PROJECT_PLAN.md](./PROJECT_PLAN.md)** — scope, datasets, model design, the priority-score
formula, API contract, hour-by-hour build schedule, risks, and the demo script.

## Data licenses

xBD / xView2 imagery and labels are **CC BY-NC-SA 4.0 (non-commercial)**. OpenStreetMap is
**ODbL**. WorldPop is CC BY 4.0. Basemap imagery © Esri, Maxar, Earthstar Geographics.

**WorldPop caveat:** its constrained raster over-concentrates population — a zonal sum over
Kathmandu reads ~3× the census figure, and its hottest cell implies 4.7 M people/km². The
ranking is unaffected (population is normalized against the AOI maximum), but absolute
"affected population" is anchored to the 2021 census via `population.calibration` in
`config/priority.yaml`. Details in [data/README.md](./data/README.md).

## Limitations

Single disaster type (earthquake). Imagery-dependent — cloud cover, off-nadir angles and
poor pre/post co-registration all degrade results. Not validated against ground truth in Nepal.
A first-pass triage aid, not an authoritative damage survey.
