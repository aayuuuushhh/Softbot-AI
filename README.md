# Uddhar

Multi-modal disaster imagery → web-informed resource allocation.

Uddhar turns pre/post-event satellite pairs and crowd-sourced ground photos into a
geotagged damage map, then routes people, food, water and medical supplies across a
road-network graph using an agent that grounds its decisions in historical disaster
response data — and cites its sources.

See `CLAUDE.md` for the full architecture and constraints.

---

## Status

| Milestone | Scope | State |
|---|---|---|
| 1 | Config, MongoDB schema, FastAPI skeleton, demo seed | **done** |
| 2 | Perception S1/S2/S3 — change detection, ground classification, fusion | next |
| 3 | Reasoning S4/S5/S6 — needs, PyG graph routing, inventory ledger | |
| 4 | Agentic routing S7/S8 — web research, allocation, PDF report | |
| 5 | Next.js + MapLibre dashboard, Docker | |

Endpoints belonging to a future milestone return **501**, not a stub result.

## Quick start

```bash
# one-time: creates the `uddhar` conda env, installs torch 2.6.0+cu124, runs preflight
bash scripts/setup_env.sh
conda activate uddhar

cp .env.example .env        # add ANTHROPIC_API_KEY and TAVILY_API_KEY when you reach M4

# MongoDB: on this machine it already runs as a systemd service (`systemctl
# status mongod`). Otherwise: bash scripts/run_mongo.sh
python scripts/seed_demo.py --reset

uvicorn api.main:app --reload --port 8765
```

Then:

```bash
curl localhost:8765/health          # confirms cuda: true and the active backends
open http://localhost:8765/docs
```

Port 8000 and 8010 are already taken on this machine by other processes, hence 8765.

## Layout

```
core/config.py     settings; picks the vision backend and resolves the device
core/schemas.py    pydantic models — the wire format and the storage format
core/db.py         motor client, collection names, indexes
core/vision/       S1/S2/S3 perception            (milestone 2)
core/spatial/      S5 PyTorch Geometric routing   (milestone 3)
agents/            S7/S8 research + allocation    (milestone 4)
api/               FastAPI app and routers
scripts/           check_env, setup_env, run_mongo, seed_demo, train_*
tests/             pytest; DB tests skip when MongoDB is down
web/               Next.js dashboard              (milestone 5)
```

## Design notes worth knowing

**Backends degrade, they don't crash.** `UDDHAR_SAT_BACKEND` / `UDDHAR_GROUND_BACKEND`
select `stub` (deterministic OpenCV, no weights) or `torch`. If a `.pt` file is missing,
`torch` logs a loud warning and falls back to `stub`. A fresh clone runs end-to-end with
no trained weights.

**Citations are structural.** `DispatchOrder.citations` has `min_length=1` and
`Citation.source_url` must be a real http(s) URL. An uncited dispatch order cannot be
constructed, so the agent cannot emit one.

**Blocked roads are removed, not penalised.** Routing in milestone 3 deletes
`status: "blocked"` edges from the graph unless air transport is verified for the event,
so no caller can accidentally route relief through a landslide.

**6 GB VRAM is the binding constraint.** 512 px tiles, small batches, AMP throughout.
`scripts/check_env.py` warns if the torch/torchvision pair is mismatched or CUDA is absent.

## Testing

```bash
pytest -q                    # DB-backed tests skip if MongoDB is not running
python scripts/check_env.py  # preflight: CUDA, version pairing, MongoDB, API keys
```
