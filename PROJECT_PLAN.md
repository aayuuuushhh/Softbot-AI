# AI Disaster Damage Assessment & Response System — Build Plan

> Read this before writing any code. It fixes scope, data, models, the API contract, the scoring
> formula, and the build order so the sprint is spent building, not deciding.

**Stack (locked):** Python 3.11 · PyTorch · FastAPI · GeoPandas/rasterio/shapely · React + Vite + Leaflet
**Timeline (locked):** 48–72h hackathon sprint — a working end-to-end demo beats a better model.

---

## 1. Problem statement & MVP scope

When an earthquake hits Nepal, responders must decide where to send limited teams *before*
ground surveys finish. This system takes **pre- and post-disaster imagery**, detects buildings,
classifies damage, and produces a **ward-level priority map** so responders can answer one
question: *where do we go first?*

It is a **first-pass assessment tool**, not a replacement for human teams. Every AI result is
reviewable and correctable by an official before it informs a decision (human-in-the-loop).

### Locked choices
| Decision | Choice | Why |
|---|---|---|
| Disaster type | **Earthquake** | Best dataset coverage (xBD), directly relevant to Nepal 2015 |
| Demo AOI | A ward-level area in **Kathmandu Valley** | Matches the brief's "Ward 8" narrative |
| Fallback AOI | An xBD earthquake AOI (e.g. Mexico City / Nepal Flooding tiles) | If Nepal pre/post pairs can't be sourced in time |
| Aggregation unit | **Ward** | Matches how Nepali local government actually dispatches |

### In scope
pre/post tile pair → building polygons → 4-class damage → ward aggregation → priority score →
interactive map + ward cards → reviewer override → export.

### Out of scope (say no to these during the sprint)
Live satellite tasking · multi-disaster models · training a foundation model from scratch ·
authentication / multi-tenancy · mobile app · real-time streaming · SAR / multispectral fusion.

### Success criteria
- [ ] End-to-end run on a laptop in **< 60s per tile pair**.
- [ ] Map renders buildings colored by the brief's legend (Red / Orange / Yellow / Blue).
- [ ] Ward panel reproduces the brief's card verbatim in format:

  > **Ward 8** — 127 potentially damaged buildings · 34 severely damaged
  > Estimated affected population: 2,400 · 3 critical facilities nearby
  > **Priority: Very High**

- [ ] A human can reclassify one building in the UI and the ward score updates live.
- [ ] Results export as GeoJSON + CSV.

---

## 2. Data plan

**Settle this first.** More hackathons die on data hunting than on modelling.

### Primary: xBD / xView2
- Source: <https://xview2.org/dataset> (registration required) — mirrors also on AWS Open Data.
- Content: paired **pre** and **post** 1024×1024 RGB GeoTIFFs + building polygons with 4 damage
  labels: `no-damage`, `minor-damage`, `major-damage`, `destroyed`.
- License: CC BY-NC-SA 4.0 — **non-commercial**; fine for a hackathon, state it on the slide.
- Maps directly onto the brief's categories:

  | xBD label | Our class | Map color |
  |---|---|---|
  | `no-damage` | Undamaged | (unfilled / grey) |
  | `minor-damage` | Suspected damage | **Yellow** |
  | `major-damage` | Moderate damage | **Orange** |
  | `destroyed` | Severe damage | **Red** |

  Critical infrastructure is an **overlay**, not a damage class → **Blue**.

### Nepal overlays
| Dataset | Gives us | Source | Join key |
|---|---|---|---|
| **OpenStreetMap / HOT export** | building footprints, hospitals, schools, bridges, road network | <https://export.hotosm.org>, Geofabrik Nepal extract | spatial (within ward polygon) |
| **Microsoft / Google Open Buildings** | footprint fallback where OSM is sparse | MS Building Footprints, Google Open Buildings v3 | spatial |
| **WorldPop / HRSL** | population per 100m cell → affected population | <https://www.worldpop.org> (NPL, constrained) | zonal sum over ward / building buffer |
| **Nepal ward boundaries** | aggregation unit + ward names | Nepal federal admin GeoJSON (OCHA HDX) | `ward_id` |

### Layout
```
data/
  raw/         # untouched downloads (xBD zips, OSM pbf, WorldPop tif)
  interim/     # reprojected, tiled, cleaned
  processed/   # model-ready chips + manifests
  nepal/       # wards.geojson, facilities.geojson, roads.geojson, worldpop.tif
```
All gitignored. `data/README.md` holds the exact download commands so anyone can rebuild it.

### Preprocessing steps
1. Reproject everything to **EPSG:4326** (store a metric CRS, e.g. EPSG:32645 / UTM 45N, for area & distance math).
2. Verify **pre/post co-registration** — check a few corner features; note a manual pixel offset if needed.
3. Tile large rasters to 1024×1024 with overlap.
4. Extract per-building **chips**: polygon bbox + fixed padding (e.g. 10px), resized to 128×128, from both pre and post.
5. Print **class balance** — xBD is dominated by `no-damage`; you need this number for loss weighting.

---

## 3. Architecture

```
pre/post GeoTIFF ─▶ preprocess ─▶ building footprints ─▶ damage classifier
                                        │                       │
                                 (OSM/MS  or  U-Net)      (siamese CNN)
                                        └──────┬────────────────┘
                                   per-building GeoJSON (damage + confidence)
                                               │
                        ward aggregation + priority scoring (pop, infra, access)
                                               │
                                 FastAPI  ──▶  React + Leaflet dashboard
                                               │
                                  reviewer overrides ──▶ SQLite ──▶ re-score
```

Two decoupled stages, so stage 2 still demos even if stage 1 is swapped out.

### Stage 1 — Building localization
- **Sprint default: skip the model.** Use existing OSM / Open Buildings polygons. Zero training
  time, and for Nepal these footprints are genuinely good.
- **Stretch:** U-Net via `segmentation_models_pytorch` (ResNet34 encoder, ImageNet weights) on
  xBD pre-images. Only attempt this if stage 2 is already green.

### Stage 2 — Damage classification (where the sprint's training time goes)
- **Siamese ResNet18**: one shared ImageNet-pretrained encoder runs on the pre-chip and the
  post-chip; concatenate the two feature vectors (and optionally their difference) → MLP head →
  **4 classes**.
- Loss: **class-weighted cross-entropy** (weights ∝ 1/frequency) — otherwise it predicts
  `no-damage` for everything and scores 80% accuracy while being useless.
- Augmentation: flips, 90° rotations, mild color jitter — applied **identically** to pre and post.
- Report **per-class F1 and a confusion matrix**, not overall accuracy. Judges notice.
- Train on a subset if time is short; a 30-minute run on balanced chips beats an unfinished 6-hour run.

### Fallback (build this *before* you trust training)
Image-difference heuristic per polygon: brightness delta, texture/edge-density delta, and
structural-similarity drop between pre and post chips → thresholds → severity class. It is
mediocre, it is honest, and it guarantees the map is never empty at demo time.

---

## 4. Response Priority Score

The brief's core idea: *don't show all 500 damaged buildings — show where to go first.*
Computed per ward (and a simplified per-building version for map coloring).

```
damage_score     = Σ(wᵢ × count_classᵢ) / max_ward_damage
                   where w: destroyed 1.0, major 0.6, minor 0.3, none 0.0

population_score = affected_population / max_ward_population

infra_score      = min(1, Σ facility_weight within radius) 
                   where hospital 1.0, bridge 0.7, school 0.5

access_penalty   = normalized( road disruption + distance from staging area )   # 0 = easy, 1 = cut off

priority = 0.40·damage_score
         + 0.30·population_score
         + 0.20·infra_score
         + 0.10·access_penalty
```

Bands: `≥0.75` **Very High** · `≥0.50` **High** · `≥0.25` **Medium** · else **Low**.

Map legend (exactly as the brief specifies):
- 🔴 **Red** — severe damage + high population
- 🟠 **Orange** — moderate damage
- 🟡 **Yellow** — suspected damage
- 🔵 **Blue** — hospitals, schools, bridges and other critical infrastructure nearby

All weights and thresholds live in **`config/priority.yaml`** — one file, tunable live during the
demo. Being able to say "watch what happens if we weight hospitals higher" is a strong moment.

`affected_population`: zonal sum of WorldPop over damaged-building buffers within the ward,
clipped to ward area. Document the assumption — judges will ask how you got "2,400".

---

## 5. Repo layout & API contract

```
backend/
  app/
    main.py                  # FastAPI app
    routers/{assess,buildings,wards,export}.py
    services/detect.py       # footprints (OSM or model)
    services/classify.py     # siamese inference
    services/score.py        # priority scoring
    models.py                # SQLAlchemy: Assessment, Building, Ward, Override
    schemas.py               # Pydantic
ml/
  datasets/xbd.py            # chip extraction + Dataset
  models/siamese.py
  train.py  infer.py  evaluate.py
  notebooks/
frontend/
  src/components/Map/        # Leaflet, GeoJSON layers, legend
  src/components/WardPanel/  # the priority cards
  src/components/ReviewDrawer/  # human-in-the-loop override
  src/api/client.ts
config/    priority.yaml, paths.yaml
scripts/   download_data.sh, run_pipeline.py
data/      (gitignored)
```

### Endpoints
Agree this contract in **hour 1** — it's what lets ML, backend and frontend work in parallel.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/assess` | Submit a pre/post pair (paths or upload) → `{ "assessment_id": "..." , "status": "running" }` |
| `GET` | `/api/assessments/{id}` | Job status + summary counts |
| `GET` | `/api/assessments/{id}/buildings` | GeoJSON `FeatureCollection`; each feature has `damage_class`, `confidence`, `reviewed_by_human`, `ward_id` |
| `GET` | `/api/assessments/{id}/wards` | Ward cards: `{ ward_id, name, damaged, severe, affected_population, critical_facilities, priority_score, priority_band }` |
| `PATCH` | `/api/buildings/{id}` | `{ "damage_class": "destroyed" }` → override, sets `reviewed_by_human=true`, re-scores the ward |
| `GET` | `/api/assessments/{id}/export?format=geojson\|csv` | Responder-ready download |

Building feature example:
```json
{
  "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [[[85.32, 27.71], "..."]] },
  "properties": {
    "id": "b_00417", "ward_id": "ktm-08",
    "damage_class": "major-damage", "confidence": 0.81,
    "reviewed_by_human": false
  }
}
```

Storage: **SQLite** via SQLAlchemy. Don't stand up Postgres/PostGIS in a 48h sprint — do the
spatial work in GeoPandas and persist results.

---

## 6. Hour-by-hour sprint schedule

Every phase has a **done-gate**. If a gate isn't met, cut scope — do not slide the schedule.

### H0–H8 — Foundation
- Repo scaffold, `requirements.txt` / `uv`, Vite app running.
- Download xBD sample + Nepal wards/facilities/WorldPop.
- Agree the GeoJSON contract (§5) and commit `schemas.py` + a **mock endpoint returning fake data**.
- **Gate:** one pre/post tile pair and ward boundaries visible on a Leaflet map; frontend reads the mock API.

### H8–H24 — Model
- Chip extraction + class-balance stats.
- Siamese training run → checkpoint + confusion matrix.
- **Build the heuristic fallback first**, then train. The fallback is the insurance policy.
- **Gate:** `infer.py` turns a tile pair into per-building classes via *either* path.

### H24–H40 — Integration
- Wire FastAPI to the real pipeline; swap the frontend off mock data.
- Ward aggregation + priority scoring from `config/priority.yaml`.
- Map colored by class; ward panel showing the brief's card.
- **Gate:** click a ward → real numbers → real priority band.

### H40–H56 — Human-in-the-loop & polish
- Override UI (select building → reclassify → ward re-scores live).
- Export endpoint, legend, empty/loading/error states, mobile-ish layout.
- **Gate:** the "human corrects the AI" demo beat works end to end.

### H56–H72 — Demo
- **Freeze code.** Rehearse the run at least twice, on the demo machine, offline.
- README with screenshots, architecture diagram, limitations slide.
- **Gate:** the whole demo runs from a cold start without a terminal error.

### Parallel split (2–3 people)
- **ML:** data prep, chips, training, `infer.py`
- **Backend/GIS:** FastAPI, ward joins, WorldPop zonal stats, scoring, export
- **Frontend:** Leaflet map, ward panel, review drawer

The GeoJSON contract from H0–H8 is what keeps these three from blocking each other. Backend
serves mock data until the model lands; frontend never waits.

---

## 7. Risks & mitigations

| Risk | Signal | Mitigation |
|---|---|---|
| Nepal pre/post imagery unavailable | Nothing usable by H8 | Demo on an xBD AOI, keep Nepal wards/pop as the overlay story |
| Model doesn't converge | Val F1 flat after 2 epochs | Ship the heuristic path; present the model as trained-but-limited with honest metrics |
| Pre/post misaligned | Buildings offset between tiles | Co-registration check in H0–H8; apply a manual pixel shift |
| CRS / ward-join mismatch | Buildings land in the wrong ward or nowhere | Validate joins during H0–H8, never at H40 |
| Class imbalance | 80% accuracy, all `no-damage` | Class weights + per-class F1 as the reported metric |
| Slow inference on full tiles | Demo hangs | Cap polygons per demo tile, precompute the demo assessment, cache results |
| Laptop dies / no wifi at venue | — | Everything runs offline from local files; no live downloads in the demo path |

---

## 8. Demo script

Four beats, ~4 minutes:

1. **The problem (30s).** After the 2015 earthquake, officials had limited teams and no fast way
   to know which wards needed them most.
2. **Run it (60s).** Feed the pre/post pair. Buildings appear, colored by damage.
3. **The actual value (90s).** "500 damaged buildings isn't an answer. *This* is." Open the ward
   panel — Ward 8, 127 damaged, 34 severe, 2,400 people, 3 critical facilities, **Very High**.
   Optionally re-weight hospitals in `priority.yaml` and show the ranking shift.
4. **Human-in-the-loop (60s).** An official reviews a building the AI called `destroyed`,
   corrects it, and the ward score updates. "The AI does the first pass. People decide."

   *Rehearsal note:* scores are normalized against the worst ward in the AOI, so correcting a
   building inside the top-ranked ward moves its counts but not its score (it is still the
   maximum). For a visible ranking shift, make the correction in the **second- or third-ranked
   ward** instead.

Close on limitations honestly: single disaster type, imagery-dependent, non-commercial dataset,
not validated against ground truth in Nepal. Stating these builds more credibility than hiding them.

---

## 9. Week-1 checklist (mapped to the brief)

- [ ] **1. Study existing research** — Google Research / UN damage-assessment approach; note what
      transfers to Nepal (§1, §3)
- [ ] **2. Define the MVP** — one disaster type (earthquake); outputs = damaged buildings, severity,
      affected areas, priority (§1)
- [ ] **3. Collect and prepare data** — xBD + OSM + WorldPop + ward boundaries downloaded,
      reprojected, tiled, chipped (§2)
- [ ] **4. Design the AI pipeline** — architecture, model choices, GIS stack fixed (§3, §4)
- [ ] **5. Build the initial prototype structure** — repo, env, base map/dashboard, training &
      processing pipeline scaffolded (§5, §6 H0–H8)

**Week-1 goal:** MVP defined, datasets prepared, architecture finalized, dashboard/model pipeline
ready for development.
