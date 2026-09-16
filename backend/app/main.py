"""AI Disaster Damage Assessment & Response System - API.

    pre/post imagery -> buildings -> damage classes -> ward priority -> dashboard
                                                             ^
                                              human review corrects the AI

See PROJECT_PLAN.md for scope, the priority formula and the build schedule.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import priority_config, reload_config
from .routers import assess, buildings, export, wards
from .schemas import Legend, LegendEntry

app = FastAPI(
    title="AI Disaster Damage Assessment & Response System",
    description="First-pass earthquake damage assessment and response prioritisation for Nepal.",
    version="0.1.0",
)

# The Vite dev server. Tighten before anything leaves a laptop.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assess.router)
app.include_router(buildings.router)
app.include_router(wards.router)
app.include_router(export.router)

meta = APIRouter(prefix="/api", tags=["meta"])

DAMAGE_LABELS = {
    "destroyed": "Severe damage",
    "major-damage": "Moderate damage",
    "minor-damage": "Suspected damage",
    "no-damage": "Undamaged",
}


@meta.get("/health")
def health() -> dict:
    return {"status": "ok"}


@meta.get("/legend", response_model=Legend)
def legend() -> Legend:
    """Colors and labels, served from config/priority.yaml so the map legend and the
    scoring weights can never drift apart."""
    cfg = priority_config()
    return Legend(
        damage=[
            LegendEntry(key=key, label=DAMAGE_LABELS.get(key, key), color=color)
            for key, color in cfg["damage_colors"].items()
        ],
        priority=[
            LegendEntry(key=b["name"], label=b["name"], color=b["color"]) for b in cfg["bands"]
        ],
        critical_infrastructure=LegendEntry(
            key="critical_infrastructure",
            label="Critical infrastructure nearby",
            color=cfg["critical_infrastructure_color"],
        ),
    )


@meta.get("/config/priority")
def get_priority_config() -> dict:
    return priority_config()


@meta.post("/config/reload")
def post_reload_config() -> dict:
    """Re-read config/priority.yaml without restarting.

    This is the demo beat: change the hospital weight, reload, watch the ranking shift.
    """
    reload_config()
    return {"status": "reloaded", "weights": priority_config()["weights"]}


app.include_router(meta)


@app.get("/")
def root() -> dict:
    return {
        "name": "AI Disaster Damage Assessment & Response System",
        "docs": "/docs",
        "health": "/api/health",
    }
