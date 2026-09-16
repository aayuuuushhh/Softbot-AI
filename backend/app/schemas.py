"""The API contract.

Agreed in hour 1 of the sprint. ML, backend and frontend all code against this, which is what
lets the three tracks run in parallel before the model exists.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DamageClass(str, Enum):
    """xBD's four labels. See PROJECT_PLAN.md section 2 for the mapping to map colors."""

    NO_DAMAGE = "no-damage"
    MINOR = "minor-damage"
    MAJOR = "major-damage"
    DESTROYED = "destroyed"


class AssessmentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class InferenceMode(str, Enum):
    """Which stage-2 path produced the classes."""

    MODEL = "model"          # siamese CNN
    HEURISTIC = "heuristic"  # image-difference fallback
    MOCK = "mock"            # synthetic demo data


# --- Requests -------------------------------------------------------------


class AssessRequest(BaseModel):
    pre_image: str = Field(..., description="Path to the pre-disaster raster")
    post_image: str = Field(..., description="Path to the post-disaster raster")
    aoi_name: str | None = Field(None, description="Human-readable area name for the dashboard")
    mode: InferenceMode = InferenceMode.MOCK


class BuildingOverride(BaseModel):
    """Human-in-the-loop correction. Triggers a ward re-score."""

    damage_class: DamageClass
    note: str | None = None


# --- Responses ------------------------------------------------------------


class BuildingProperties(BaseModel):
    id: str
    ward_id: str
    damage_class: DamageClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    reviewed_by_human: bool = False
    review_note: str | None = None


class Feature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict[str, Any]
    properties: BuildingProperties


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature]


class PriorityBreakdown(BaseModel):
    """The four terms behind the score, so the dashboard can show *why* a ward ranks high."""

    damage: float
    population: float
    infrastructure: float
    accessibility: float


class WardCard(BaseModel):
    """Reproduces the brief's ward card:

    Ward 8: 127 potentially damaged buildings / 34 severely damaged /
    Estimated affected population: 2,400 / 3 critical facilities nearby / Priority: Very High
    """

    ward_id: str
    ward_name: str
    total_buildings: int
    damaged_buildings: int
    severely_damaged: int
    damage_breakdown: dict[DamageClass, int]
    affected_population: int
    critical_facilities: int      # headline types only - see headline_facility_types in config
    facility_breakdown: dict[str, int] = {}   # every type within range, with counts
    facility_types: list[str] = []            # distinct types present, for chips
    priority_score: float
    priority_band: str
    priority_color: str
    breakdown: PriorityBreakdown
    reviewed_count: int = 0


class GeometrySource(str, Enum):
    """Where the wards, buildings and facilities came from - independent of where the
    *damage classes* came from (`InferenceMode`). Real geometry with simulated damage is a
    legitimate intermediate state, and the dashboard should be able to say so precisely."""

    OSM = "osm"              # real OpenStreetMap layers + WorldPop
    SYNTHETIC = "synthetic"   # the generated demo grid


class AssessmentSummary(BaseModel):
    assessment_id: str
    status: AssessmentStatus
    mode: InferenceMode
    geometry_source: GeometrySource = GeometrySource.SYNTHETIC
    aoi_name: str | None = None
    created_at: str
    total_buildings: int = 0
    damaged_buildings: int = 0
    severely_damaged: int = 0
    affected_population: int = 0
    wards: int = 0
    error: str | None = None


class LegendEntry(BaseModel):
    key: str
    label: str
    color: str


class Legend(BaseModel):
    damage: list[LegendEntry]
    priority: list[LegendEntry]
    critical_infrastructure: LegendEntry
