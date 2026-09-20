# CLAUDE.md — Uddhar

Project instructions for Claude Code. Read this fully before writing any code.

---

## 1. What this is

**Uddhar** turns a before/after image pair of a disaster-affected area into three things:

1. **What was damaged** — building-level structural damage, classified by severity.
2. **What each zone needs** — shelter, water, food, medical priority, computed from damage and population.
3. **What never arrived** — computed need minus confirmed delivery, per zone, with days elapsed.

It runs offline, on CPU, on a single laptop.

### The core insight

Every relief logistics system in existence starts with a **request**. Someone fills a form, calls a number, submits a demand. Aid therefore flows toward whoever can reach a form — which is precisely why the most isolated and worst-affected communities receive least.

Uddhar does not start with a request. It starts with a **computation**. Need is derived from imagery, objectively, before anyone asks, for communities that cannot ask.

That inversion is the product. Everything in this repo serves it.

> Aid in Nepal flows toward whoever can make a phone call. Uddhar computes need from imagery, so the villages that can't call are still in the queue.

### Operating context

Anchoring event: the 26 August 2026 Bhote Koshi / Trishuli flood. ~1,050 confirmed dead, thousands missing, 1.6 million affected across Rasuwa, Nuwakot, Dhading and Gorkha. Roughly 200 telecom towers down between Nepal Telecom and Ncell. Bridges destroyed, bailey bridges installed as replacements. Thirty-seven holding centres established. Twenty containers of relief supplies stranded at Tribhuvan International Airport long enough for an MP to photograph them and raise it in parliament.

Secondary reference: the November 2023 Jajarkot earthquake, where the government's Initial Rapid Assessment reported over 4,000 damaged homes at the 48-hour mark against a verified final figure of 28,769.

**Design for that environment.** District EOC, generator power, no uplink, a mid-range laptop, an officer under time pressure who is not a GIS analyst.

### What Uddhar explicitly does NOT do

State these limits in the UI and in any generated report. They are not weaknesses; naming them is what makes the rest credible.

- Does not clear customs or move physical goods.
- Does not dispatch or assign rescue teams to zones. It ranks; humans assign.
- Does not accuse anyone of diversion. It reports arithmetic: dispatched minus received.
- Does not work through monsoon cloud. Optical imagery only; SAR is out of scope.
- Does not replace field assessment. It produces a first-pass picture in the hours before field data exists.

---

## 2. Non-negotiable constraints

These are architectural commitments. Do not violate them for convenience.

| Constraint | Rule |
| --- | --- |
| **Offline-first** | The full pipeline must run with the network interface down. No cloud inference, no remote API in the critical path. ~200 towers were down in the anchoring event; a system needing connectivity is a system that fails during the event it was built for. |
| **CPU only** | No CUDA, no GPU dependency, no model requiring accelerated inference. Target: full pipeline under 20 s on 4 cores for a 4096×4096 pair. |
| **Deterministic core** | Damage classification, need computation and gap arithmetic must be reproducible. Same input, same output, every time. No LLM anywhere in these paths. |
| **Traceable numbers** | Every figure surfaced to a user must be expandable to the arithmetic that produced it. `171 affected = 38 destroyed × 4.5 persons/household`. If a number cannot be explained, it will be ignored by the person who needs it. |
| **Local storage** | SQLite on disk. No Postgres, no cloud DB, no ORM requiring a server. The delivery ledger must be writable on a disconnected machine and syncable later. |
| **Degrade loudly** | Poor co-registration, cloud cover, missing baseline, absent road data — each must produce an explicit warning and a partial result, never a silently confident wrong answer. |

---

## 3. Architecture

```
INPUT            pre.tif, post.tif  [+ optional: ward.geojson, roads.geojson, dem.tif]
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
PERCEPTION       S1  co-register        FFT phase correlation, sub-pixel
                 S2  normalise          slope-aspect banded illumination correction
                 S3  detect change      multi-channel difference + morphology
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
REASONING        S4  extract            connected components → building footprints
                 S5  classify           minor / major / destroyed + confidence
                 S6  reachability       road-corridor change → reachable/blocked/unknown
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
DECISION         S7  zone + need        aggregate to wards, compute requirements
                 S8  gap engine         need − confirmed delivery, ranked
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
OUTPUT           damage overlay · zone table · need table · undelivered list
                 GeoJSON export (BIPAD-aligned) · PDF field report
```

Stages S1–S8 are pure functions over their inputs. Each writes a serialisable artefact to the run directory so any stage can be inspected, re-run or replaced in isolation.

---

## 4. Phase 1 features

Four features. Nothing else ships in Phase 1. Section 9 lists what is deliberately excluded.

### F1 — Damage detection

**Input:** `pre.tif`, `post.tif`, optional `dem.tif`, optional `buildings.geojson`
**Output:** `structures.geojson` — one feature per detected damaged structure, with severity and confidence.

#### S1 · Co-registration

Two acquisitions are never pixel-aligned, and uncorrected misregistration is the single largest source of false damage.

- Compute cross-power spectrum of the two frames; locate the correlation peak in the Fourier domain.
- Use `skimage.registration.phase_cross_correlation` with `upsample_factor=10` for sub-pixel refinement.
- Apply the translation to the post frame; crop both to the overlapping extent.
- **Confidence** = peak height / mean of surrounding correlation surface. Below `0.15` → emit `LOW_REGISTRATION_CONFIDENCE` and continue with a warning banner.
- Shift exceeding 15% of the smaller image dimension → abort with `REGISTRATION_FAILED`. Do not produce damage output from a bad alignment.

Chosen over SIFT/ORB + RANSAC deliberately: post-disaster scenes destroy the keypoints feature matching depends on. Phase correlation degrades gracefully where feature matching fails outright.

#### S2 · Slope-aspect illumination normalisation

**This is the Nepal-specific contribution. Treat it as a primary objective, not preprocessing.**

Sun elevation and azimuth differ between acquisitions. On flat terrain, global histogram equalisation suffices — that is what established systems use. On a 35–45° Nepali hillside it does not: an entire slope face changes brightness between passes, and a global normaliser reads that as structural change across hundreds of buildings.

Implementation:
1. If a DEM is supplied, compute slope and aspect per pixel (`richdem` or a Horn-method gradient over the elevation array).
2. Bin aspect into 8 bands of 45°. Optionally sub-bin slope into `<15°`, `15–30°`, `>30°`.
3. Apply CLAHE to the luminance channel **independently within each band**, so north- and south-facing terraces are normalised against their own populations.
4. Blend across band boundaries with a small feather (3–5 px) to avoid seam artefacts.
5. No DEM supplied → fall back to global CLAHE and set `terrain_correction: false` in the run manifest. The UI must show this, because results on steep terrain are then unreliable.

**Required validation:** run against two clear-sky acquisitions of an *undamaged* Nepali hillside at different sun angles. Banded normalisation should report near-zero change where global CLAHE reports substantial false change. Keep this as a regression test — it is the evidence for the whole approach.

#### S3 · Change detection

Difference the normalised frames across three channels:

| Channel | Computation | Captures |
| --- | --- | --- |
| Intensity | absolute luminance difference | gross brightness change |
| Texture | local standard deviation (7×7 window) delta | roughening from rubble |
| Edge | Canny density in 15×15 window, delta | loss of structural line coherence |

Structural collapse produces a characteristic *joint* signature across all three. Benign change (vehicles, vegetation movement, water level) typically triggers one channel only. Combine as a weighted sum, weights `[0.3, 0.4, 0.3]`, tunable in config.

Threshold adaptively (Otsu per 512×512 tile, floored at a global percentile to prevent tiles of pure noise producing spurious detections). Then morphological opening with a 3×3 kernel to remove speckle, followed by closing with 5×5 to consolidate fragmented collapse regions.

#### S4 · Building extraction

- `skimage.measure.label` on the change mask → discrete candidate structures.
- Filter by physical plausibility for rural Nepali building stock:
  - area: `6 m²` to `400 m²` (convert via pixel ground resolution — read from GeoTIFF, do not assume)
  - solidity `> 0.4`
  - major/minor axis ratio `< 6.0`
- Where `buildings.geojson` (OSM or Google Open Buildings) is supplied, associate components with known footprints by maximum IoU. Where absent — the common case in rural Karnali and upper Rasuwa — the component itself defines the structure.

#### S5 · Severity classification

Feature vector per structure:

```
change_magnitude_mean        mean combined change score within footprint
change_extent_ratio          changed pixels / total footprint pixels
edge_coherence_loss          pre edge density − post edge density
texture_entropy_delta        Shannon entropy shift within footprint
debris_signature             mean change in a 10 m annulus around the footprint
footprint_area_m2
```

Classifier: gradient-boosted trees (`sklearn.ensemble.HistGradientBoostingClassifier`). Not a neural network — CPU-only, fast, inspectable, and `feature_importances_` gives a real explanation to surface in the UI.

Classes: `minor`, `major`, `destroyed`. Emit `predict_proba` as confidence; below `0.55` mark `low_confidence: true` and render distinctly.

Train on xBD, collapsing its four classes to three (`no-damage` is dropped since change detection already gates on it). Prioritise `mexico-earthquake` and any masonry-dominant events; xBD's timber-frame American stock is a known domain gap. Target the same evidence standard as published work: balanced classes, ≥500 examples per class.

#### S6 · Reachability flag

**Minimal by design.** Not routing, not travel-time estimation, not a weighted graph. One flag per zone.

1. Buffer OSM road centrelines to corridor polygons (8 m default, configurable).
2. Compute change statistics along each corridor segment.
3. Mark a segment `blocked` when a contiguous high-magnitude, high-texture change region intersects the corridor **transversely** (perpendicular extent exceeding the corridor width) — the signature of a landslide or debris deposit crossing the road, as distinct from change running parallel to it.
4. Build a `networkx` graph of road segments. Remove blocked edges. Test connectivity from designated staging nodes.
5. Emit per zone: `reachable` | `blocked` | `unknown` (`unknown` when no road data covers the zone).

**Why this matters and why it cannot be cut:** "Zone 7 received nothing in six days" is a complaint. "Zone 7 received nothing in six days *and its access road is blocked*" is a diagnosis — helicopter problem, not paperwork problem. The flag is what turns F3 from an accusation into a diagnostic.

---

### F2 — Need computation

**Input:** `structures.geojson`, `zones.geojson` (ward boundaries), population raster
**Output:** `needs.json` — per zone, with full derivation attached to every figure.

#### S7 · Zone aggregation and requirement derivation

Snap the analysis grid to real ward boundaries wherever they exist. A zone that maps to an actual ward maps to an accountable authority who can be contacted; an arbitrary grid cell does not. Fall back to a fixed grid only where boundaries are unavailable, and label it as such.

Per zone, aggregate structure counts by severity, then derive:

```
affected_people   = (destroyed × hh_size) + (major × hh_size × 0.7)
shelter_units     = ceil(affected_people / persons_per_shelter)
water_litres_day  = affected_people × 15
food_person_days  = affected_people × planning_horizon_days
medical_priority  = f(affected_people, destroyed_count, reachability)
```

**Constants must be sourced, not invented.** Record the source for each in `config/standards.yaml`:

- `15 L/person/day` — Sphere Handbook minimum water quantity for survival needs.
- Covered living space per person — Sphere minimum, currently `3.5 m²`; derive `persons_per_shelter` from the shelter unit spec in use rather than guessing.
- `hh_size` — per-district average household size from the 2015 Nepal Earthquake Open Data Portal (`eq2015.npc.gov.np`) or CBS census. Do **not** hardcode a single national figure; Rasuwa and Kathmandu differ materially.
- `0.7` occupancy factor for `major` damage — a stated planning assumption, flagged as such in the UI, not presented as measured.

**Every derived figure carries its arithmetic.** The JSON must include a `derivation` string per field:

```json
{
  "zone_id": "rasuwa-uttargaya-4",
  "affected_people": {
    "value": 171,
    "derivation": "38 destroyed x 4.5 persons/household",
    "assumptions": ["hh_size=4.5 (CBS, Rasuwa district)"]
  }
}
```

The UI must render this on click. A number a coordinator cannot interrogate is a number they will not act on.

Vulnerability weighting is optional in Phase 1. If included, weight by known demographic composition — this encodes an existing humanitarian prioritisation principle rather than inventing policy. If the data is unavailable, omit it rather than approximating.

---

### F3 — The undelivered list

**This is the feature that distinguishes Uddhar.** F1 and F2 make it possible; F3 is what it was all for.

**Input:** `needs.json`, delivery ledger (SQLite)
**Output:** `gaps.json` — ranked, with reachability context.

#### Delivery ledger

Two tables. Offline-first: writes must succeed on a disconnected machine and sync later.

```sql
CREATE TABLE consignment (
  id              TEXT PRIMARY KEY,   -- UUID, generated client-side
  agency          TEXT NOT NULL,      -- 'NDRRMA', 'CARE', 'UNICEF', 'Nepal Army'
  commodity       TEXT NOT NULL,      -- 'shelter_kit' | 'water_purification' | 'food' | 'medical'
  quantity        REAL NOT NULL,
  unit            TEXT NOT NULL,
  destination_zone TEXT NOT NULL,
  dispatched_at   TEXT NOT NULL,      -- ISO 8601
  created_offline INTEGER DEFAULT 1,
  synced_at       TEXT
);

CREATE TABLE handover (
  id            TEXT PRIMARY KEY,
  consignment_id TEXT NOT NULL REFERENCES consignment(id),
  event_type    TEXT NOT NULL,        -- 'dispatched' | 'in_transit' | 'received'
  occurred_at   TEXT NOT NULL,
  latitude      REAL,
  longitude     REAL,
  recipient     TEXT,                 -- ward official, camp manager
  recorded_by   TEXT NOT NULL,
  notes         TEXT,
  created_offline INTEGER DEFAULT 1,
  synced_at     TEXT
);
```

Sync is last-write-wins on `id`. Conflicts are logged, never silently resolved.

#### Gap engine

Per zone, per commodity:

```
confirmed_delivered = Σ quantity WHERE handover.event_type = 'received'
dispatched_total    = Σ quantity WHERE handover.event_type = 'dispatched'
in_transit_unconfirmed = dispatched_total − confirmed_delivered
gap                 = max(0, need − confirmed_delivered)
gap_ratio           = gap / need
days_since_delivery = now − max(handover.occurred_at WHERE received)
priority_score      = gap_ratio × log(1 + days_since_delivery)
```

Three output lists:

| List | Definition | Why it matters |
| --- | --- | --- |
| **Unmet** | `gap > 0`, sorted by `priority_score` desc | Six days with nothing outranks one day with nothing. |
| **Duplicated** | ≥2 agencies delivered the same commodity to the same zone within 72 h | Real, common, and invisible without a shared ledger. |
| **In transit, unconfirmed** | `in_transit_unconfirmed > 0`, sorted by days since dispatch | Dispatched but never confirmed received. This is where aid actually disappears. |

Every row carries the reachability flag from S6.

**Framing is a hard requirement.** Never label a row "diverted", "missing", "lost", or anything implying wrongdoing. The column is `unconfirmed`. The system reports arithmetic; a human interprets it. Any language beyond that is both unsupportable and a liability.

#### Data provenance — be honest about this

F1 and F2 run entirely on data Uddhar generates. F3 requires delivery records that do not currently exist in a usable shared form. For demonstration, seed the ledger from a CSV of realistic consignments modelled on the actual August 2026 response — four agencies, no shared view, some confirmed, some not.

State this plainly in the UI and the pitch: *"In deployment this is populated by agencies at handover; for this demo we've seeded it."* An honest seed beats a fabricated integration, and the insight being demonstrated is that **need and delivery can be compared at all** — nobody currently computes need objectively, so the comparison has never before been possible. The ledger is the easy half.

---

### F4 — Outputs

- **Damage overlay** on post-event imagery, severity-coded, with per-structure drill-down.
- **Zone table** — counts, need, reachability, gap.
- **Undelivered list** — the three lists above.
- **GeoJSON export** — BIPAD-aligned attributes (palika, ward, damage counts, need, gap) for ingestion into NDRRMA's existing portal. Not a rival portal; a missing layer for the one that exists.
- **PDF field report** — printable, one page per zone for the top five.
- **Low-bandwidth text summary** — under one printed page, top five zones with coordinates, counts and gaps. Readable on a low-end phone, dictatable over a radio handset, printable and physically carried. With ~200 towers down, the lowest-bandwidth output is often the only one that reaches the person who needs it.

---

## 5. Repository layout

```
uddhar/
  config/
    pipeline.yaml         thresholds, kernel sizes, channel weights
    standards.yaml        Sphere constants, household sizes, WITH SOURCES
  core/
    registration.py       S1
    illumination.py       S2  (slope-aspect banding)
    change.py             S3
    extraction.py         S4
    severity.py           S5  (+ model artefact)
    reachability.py       S6
    zoning.py             S7  aggregation
    needs.py              S7  requirement derivation
    gaps.py               S8  gap engine
  ledger/
    schema.sql
    store.py              SQLite access, offline-safe writes
    sync.py               last-write-wins reconciliation
  api/
    main.py               FastAPI
    routes/
  web/                    React + MapLibre dashboard
  reports/
    pdf.py
    geojson.py            BIPAD-aligned export
    textbrief.py          low-bandwidth summary
  tests/
    fixtures/             including the undamaged-hillside regression pair
  data/
    demo/                 seeded ledger CSV, demo image pairs
```

---

## 6. Stack

| Layer | Choice | Reason |
| --- | --- | --- |
| Imaging | NumPy, SciPy, OpenCV, scikit-image | Mature, CPU-efficient, no GPU |
| Geospatial | rasterio, Shapely, GeoPandas, NetworkX | Raster IO, geometry, road graph |
| Terrain | richdem or Horn-method gradients | Slope and aspect from DEM |
| Classifier | scikit-learn HistGradientBoostingClassifier | Fast on CPU, inspectable importances |
| Storage | SQLite | Offline-first, file-based, no server |
| API | FastAPI | Same code path online and offline |
| Web | React, MapLibre GL, Tailwind | Map-centred; offline tile caching |
| Reports | ReportLab | PDF generation |
| Packaging | Docker + a plain `pip install -e .` path | Must be installable without Docker on a field laptop |

**Do not add:** PyTorch, TensorFlow, any cloud SDK, any service requiring network at runtime.

---

## 7. Conventions

- Python 3.11+, type hints on all public functions, `ruff` + `black`.
- Every core stage is a pure function: explicit inputs, returns an artefact, writes nothing global.
- Every stage writes its intermediate to `runs/{run_id}/{stage}.{ext}` for inspection and replay.
- Config lives in YAML, never hardcoded. Magic numbers in code are a bug.
- Logging: structured, with `run_id`. Every warning carries a stable machine code (`LOW_REGISTRATION_CONFIDENCE`, `NO_DEM_SUPPLIED`, `NO_ROAD_DATA`) so the UI can render it.
- Errors are values, not exceptions, across stage boundaries. Each stage returns `(result, warnings[])`.
- All timestamps ISO 8601 UTC. All coordinates WGS84 (EPSG:4326) at API boundaries; work in an appropriate projected CRS internally and convert on output.

---

## 8. Testing and validation

Validation is a deliverable, not an afterthought. A triage system with unstated accuracy is not deployable.

**Unit** — each stage against fixtures. Deterministic: identical input must produce byte-identical output.

**Regression, slope-shadow** — the undamaged-hillside pair at two sun angles. Banded normalisation must report near-zero change where global CLAHE reports substantial false change. **This test failing means the core contribution is broken.**

**Benchmark** — held-out xBD tiles. Report per-class precision, recall and F1 for minor/major/destroyed. Position against published xView2 baselines.

**Zone ranking** — the decisive metric, because the product is the ranking, not the per-building label. Report rank correlation against ground-truth zone ordering, and top-5 recall (proportion of genuinely worst-affected zones appearing in the predicted top five). A pipeline that mislabels individual structures but orders zones correctly remains operationally valuable. Ground truth: ward-aggregated damage grades from `eq2015.npc.gov.np`, covering 762,106 surveyed buildings — real Government of Nepal data.

**Gap engine** — property tests. Gap never negative. Confirmed delivery never exceeds dispatched without an explicit over-delivery flag. Zone with no need never appears in the unmet list.

**Performance** — full pipeline under 20 s, 4 cores, 4096×4096 pair. Assert in CI.

Publish failure modes including negative results. A project that characterises where it breaks on real Nepali terrain is more credible than one showing clean results on American suburbs.

---

## 9. Out of scope for Phase 1

Do not build these. If a request arrives that implies one, say so and point here.

| Excluded | Why |
| --- | --- |
| Rescue team assignment | Ranking is decision support; assignment is decision making. Crossing that line collapses the accountability argument. Rank zones; humans assign. |
| Full routing / travel-time estimation | S6 emits a flag, not a route. Routing is Phase 2. |
| T+20min ShakeMap prediction layer | Strong idea, earthquake-only. The anchoring event is a flood; adding it fragments the story. Phase 2. |
| SAR / monsoon flood mapping | Different sensor, different processing chain entirely. Separate project. |
| Deep learning models | Breaks the CPU-only and offline constraints, which are the deployment argument. |
| LLM in the decision path | Nothing generative touches damage classification, need computation or gap arithmetic. If a narrative brief is added later, the model phrases pipeline facts and injects no entities of its own. |
| Agent orchestration | If any agentic surface is added, it is an MCP server wrapping existing deterministic tools. Nothing more. |
| Multi-agent deliberation | Theatre. Deterministic answers already exist. |
| Public accusation features | The system reports `unconfirmed`. It does not allege. |

---

## 10. The demo path

Build toward this sequence. It is what will be remembered.

1. Load a before/after pair. Overlay appears, severity-coded.
2. Zones populate and rank.
3. Need table fills — **not** "severe damage" but `171 people · 38 shelter units · 2,565 L/day`.
4. A coordinator clicks a figure; the derivation expands.
5. Delivery ledger loads. Three zones turn red: **need computed, nothing confirmed received, six days elapsed, road blocked.**

Step 5 is the entire pitch in one screen. Every engineering decision should be checked against whether it gets there.

Closing line for the slide:

> Nobody could check whether aid reached the right place, because nobody had an independent measure of what the right amount was. Uddhar computes it from imagery — so the comparison becomes possible for the first time.