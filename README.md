# Uddhar

Multi-modal disaster imagery → web-informed resource allocation.

Uddhar turns pre/post-event satellite pairs and crowd-sourced ground photos into a
geotagged damage map, computes what each ward needs, and routes people, food, water and
medical supplies across a road-network graph, adapting its planning numbers from
historical disaster-response evidence it finds on the web, and citing every source.

See `CLAUDE.md` for the full architecture and constraints.

---

## Status

| Milestone | Scope | State |
|---|---|---|
| 1 | Config, MongoDB schema, FastAPI skeleton, demo seed | **done** |
| 2 | Perception S1/S2/S3: change detection, ground classification, fusion | **done** |
| 3 | Reasoning S4/S5/S6: needs with derivations, PyG graph routing, inventory ledger | **done** |
| 4 | Agentic routing S7/S8: web research, cited allocation, PDF + radio report | **done** |
| 5 | Next.js + MapLibre dashboard, Docker | **done** |
| - | Train `satellite_change.pt` / `ground_damage.pt` | **code ready, not yet trained** |

Until the models are trained, perception runs on the deterministic `stub` backends.
Everything downstream works the same either way.

## Quick start (local)

```bash
bash scripts/setup_env.sh && conda activate uddhar   # or any venv: see setup_env.sh
cp .env.example .env        # ANTHROPIC_API_KEY enables web-grounded allocation

python scripts/make_synthetic_pair.py --out data/demo    # synthetic Rasuwa scene
python scripts/seed_demo.py --reset                      # event, wards, roads, stock
uvicorn api.main:app --reload --port 8765

cd web && npm install && npm run dev                     # http://localhost:3000
```

`python scripts/smoke_e2e.py` drives the whole pipeline over real HTTP (analysis, ground
uploads, needs, allocation, approval, PDF). It needs a freshly seeded event.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build
docker compose run --rm api python scripts/make_synthetic_pair.py --out data/demo
docker compose run --rm api python scripts/seed_demo.py --reset
```

Dashboard on :3000, API on :8765. With an NVIDIA GPU add
`-f docker-compose.gpu.yml` (CUDA 12.4 torch wheels). Trained weights are mounted from
`./models`, not baked into the image.

## The pipeline

| Stage | Where | What it does |
|---|---|---|
| S1 | `core/vision/inference.py` | Siamese change detection on the pre/post pair (stub or `SiameseChangeNet`) |
| S2 | `core/vision/inference.py` | Ground photo → severity (stub or `GroundDamageNet`) |
| S3 | `core/vision/fusion.py` | Per-ward fusion. Ground photos decide **severity**; overhead imagery decides **extent** |
| S4 | `core/needs.py` | Affected people → doctors, rescuers, rice, water, trauma kits, each with its arithmetic |
| S5 | `core/spatial/graph_network.py` | PyG road graph. Blocked roads removed; roads cut by detected damage auto-blocked |
| S6 | `core/inventory.py` | Atomic ledger: reserve → in transit → delivered, or cancelled |
| S7 | `agents/researcher.py` | Claude + web search finds precedents; only URLs actually retrieved can be cited |
| S8 | `agents/allocator.py`, `core/allocation.py` | Agent adapts parameters from cited precedents; a deterministic optimiser plans the dispatches |

## Training the models (later)

Both trainers write to the exact paths the API loads, in the exact checkpoint format it
accepts. That round-trip is covered by `tests/test_training.py`.

```bash
# Satellite (S1): xBD, ~3.3 GB. Events and hold-out rationale in scripts/fetch_xbd.sh.
bash scripts/fetch_xbd.sh
python scripts/train_satellite.py --xbd-root data/raw/xbd --epochs 30 --amp
python scripts/eval_xbd.py --xbd-root data/raw/xbd        # exit 0 only if destroyed F1 > 0.80

# Ground (S2): photos in data/training/ground/{none,minor,major,destroyed}/
python scripts/train_ground.py --data data/training/ground --epochs 20

# Then switch the API over:
UDDHAR_SAT_BACKEND=torch UDDHAR_GROUND_BACKEND=torch uvicorn api.main:app --port 8765
```

* **Satellite.** `SiameseChangeNet` segments every pixel into background / none / minor /
  major / destroyed. The best epoch is picked by *building-level* F1 on `destroyed`, the
  CLAUDE.md gate. `mexico-earthquake` is held out of training by default, so `eval_xbd.py`
  reports an honest seismic-transfer number alongside the test split. Quote that one for Nepal.
* **Ground.** No labelled ground-photo set ships with the repo. The docstring in
  `scripts/train_ground.py` lists candidate sources. Nepali masonry is the domain gap
  that matters.
* A 6 GB GPU fits the defaults (512 px crops, batch 4, fp16). `--max-tiles 20
  --epochs 1` makes a quick smoke run on CPU.

## Design notes worth knowing

**Backends degrade, they don't crash.** If a `.pt` file is missing, the `torch` backend
logs a warning and falls back to `stub`. A fresh clone runs end to end with no weights.

**The LLM never writes a dispatch order.** The agent proposes *parameter adjustments*
(for example patients per doctor per day), each citing retrieved precedents. Code rejects
any adjustment that is unknown, out of bounds or uncited. A deterministic optimiser then
plans the orders, and `check_invariants` re-verifies stock, demand and routes. Every run
is stored in `agent_runs`: queries, report, precedents, accepted and rejected proposals,
and parameters. See `GET /api/agent-runs/{id}`.

**Citations are structural.** `DispatchOrder.citations` has `min_length=1`. Web research
may only cite URLs its search tool actually returned; fabricated sources are dropped in
code. Without an API key, allocation runs on the cited baseline in `core/standards.py`
(Sphere, WHO EMT, Nepal Census 2021) and says so.

**Every number explains itself.** Needs carry `{value, formula, assumptions, sources}`.
In the dashboard, click any figure. Planning assumptions are labelled as such, never
presented as measurements.

**Blocked is blocked.** Blocked roads are absent from the PyG graph, and blocked nodes
lose every incident road. Air routes exist only when `air_transport_verified` is set on
the event. Satellite detections that cross a road corridor *between* wards auto-block it
(`source: cv`). An operator's manual status is never overridden.

**Stock can't be oversold.** Reservations are single conditional MongoDB updates.
Multi-item reservations roll back on partial failure, and dispatch status changes are
compare-and-set. A concurrent-reservation test pins this down.

## Layout

```
core/config.py        settings; picks the vision backend and resolves the device
core/schemas.py       pydantic models: the wire format and the storage format
core/standards.py     baseline planning parameters, each with its source
core/needs.py         S4 needs with derivations
core/allocation.py    S8 deterministic optimiser + invariant checks
core/inventory.py     S6 ledger state machine
core/report.py        PDF situation report and radio-length summary
core/vision/          S1-S3 perception; datasets.py + metrics.py for training
core/spatial/         S5 PyTorch Geometric road graph
agents/               S7 researcher, S8 allocation agent
api/                  FastAPI app and routers
scripts/              setup, seed, smoke test, fetch_xbd, train_*, eval_xbd
tests/                pytest; DB-backed tests skip when MongoDB is down
web/                  Next.js 16 + MapLibre dashboard
```

## Testing

```bash
pytest -q                    # unit, property (Hypothesis), training, and API tests
python scripts/smoke_e2e.py  # full pipeline over HTTP against a fresh seed
python scripts/check_env.py  # preflight: CUDA, version pairing, MongoDB, API key
cd web && npx tsc --noEmit && npx eslint
```
