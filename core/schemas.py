"""Pydantic models shared by the database layer, the API and the agents.

These are the wire format *and* the storage format. MongoDB `_id` values are
surfaced as plain strings via `PyObjectId`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)


def _to_str(v: Any) -> Any:
    return str(v) if v is not None else v


PyObjectId = Annotated[str, BeforeValidator(_to_str)]


def id_field() -> Any:
    """Accept MongoDB's `_id` on the way in, emit a plain `id` on the way out."""
    return Field(
        validation_alias=AliasChoices("_id", "id"),
        serialization_alias="id",
    )


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Severity(StrEnum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    DESTROYED = "destroyed"


SEVERITY_ORDER: dict[str, int] = {
    Severity.NONE.value: 0,
    Severity.MINOR.value: 1,
    Severity.MAJOR.value: 2,
    Severity.DESTROYED.value: 3,
}


class ResourceKind(StrEnum):
    PERSONNEL = "personnel"
    FOOD = "food"
    WATER = "water"
    MEDICAL = "medical"


class EdgeStatus(StrEnum):
    OPEN = "open"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class TransportMode(StrEnum):
    ROAD = "road"
    AIR = "air"


class NodeKind(StrEnum):
    HOLDING_CENTER = "holding_center"
    STAGING_NODE = "staging_node"
    ZONE = "zone"


class DispatchStatus(StrEnum):
    PROPOSED = "proposed"
    RESERVED = "reserved"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# --------------------------------------------------------------------------
# Geometry (GeoJSON subset)
# --------------------------------------------------------------------------


class Point(Base):
    type: Literal["Point"] = "Point"
    coordinates: list[float]  # [lon, lat]

    @field_validator("coordinates")
    @classmethod
    def _len2(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("Point coordinates must be [lon, lat]")
        return v


class Polygon(Base):
    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]]


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


class PipelineStages(Base):
    """Per-stage status, mirroring S1-S8 in CLAUDE.md section 3."""

    overhead: StageStatus = StageStatus.PENDING  # S1
    ground: StageStatus = StageStatus.PENDING  # S2
    fusion: StageStatus = StageStatus.PENDING  # S3
    graph: StageStatus = StageStatus.PENDING  # S5
    allocation: StageStatus = StageStatus.PENDING  # S7/S8
    message: str | None = None


class EventCreate(Base):
    name: str
    disaster_type: str = Field(description="flood | earthquake | landslide | ...")
    region: str = ""
    bbox: list[float] | None = Field(default=None, description="[minlon, minlat, maxlon, maxlat]")
    air_transport_verified: bool = False


class Event(EventCreate):
    id: PyObjectId = id_field()
    crs: str | None = None
    cloud_fraction: float = 0.0
    stages: PipelineStages = Field(default_factory=PipelineStages)
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Damage
# --------------------------------------------------------------------------


class DamageDetection(Base):
    """One detected structure/region, geotagged. Emitted by S1 and S2."""

    bbox_px: list[int] | None = None  # [x0, y0, x1, y1] in source raster pixels
    geometry: Polygon | None = None
    centroid: Point
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["satellite", "ground", "fused"]
    area_m2: float | None = None
    model_version: str = "stub"


class ZoneCreate(Base):
    event_id: PyObjectId
    name: str
    geometry: Polygon
    centroid: Point
    population: int = 0


class Needs(Base):
    """Computed requirement for one zone (S4/S6)."""

    personnel: int = 0
    food_rations: int = 0
    water_litres: int = 0
    medical_kits: int = 0


class Zone(ZoneCreate):
    id: PyObjectId = id_field()
    severity: Severity = Severity.NONE
    damage_score: float = 0.0
    buildings_destroyed: int = 0
    detections: int = 0
    decided_by: Literal["satellite", "ground", "fused", "none"] = "none"
    needs: Needs = Field(default_factory=Needs)


# --------------------------------------------------------------------------
# Ground reports (F1 - crowd-sourced verification)
# --------------------------------------------------------------------------


class GroundReport(Base):
    id: PyObjectId = id_field()
    event_id: PyObjectId
    zone_id: PyObjectId | None = None
    image_path: str
    location: Point
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str = "stub"
    reporter: str | None = None
    uploaded_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Spatial graph (S5)
# --------------------------------------------------------------------------


class GraphNodeCreate(Base):
    event_id: PyObjectId
    name: str
    kind: NodeKind
    location: Point
    zone_id: PyObjectId | None = None
    capacity: int = 0


class GraphNode(GraphNodeCreate):
    id: PyObjectId = id_field()


class GraphEdgeCreate(Base):
    event_id: PyObjectId
    u: PyObjectId
    v: PyObjectId
    distance_km: float = Field(gt=0)
    status: EdgeStatus = EdgeStatus.OPEN
    source: Literal["cv", "manual", "seed"] = "manual"


class GraphEdge(GraphEdgeCreate):
    id: PyObjectId = id_field()


# --------------------------------------------------------------------------
# Inventory (F2)
# --------------------------------------------------------------------------


class InventoryItemCreate(Base):
    event_id: PyObjectId
    node_id: PyObjectId
    kind: ResourceKind
    sku: str = Field(description="e.g. 'doctor', 'rice_25kg', 'water_purifier'")
    quantity: int = Field(ge=0)
    unit: str = "unit"


class InventoryItem(InventoryItemCreate):
    id: PyObjectId = id_field()
    reserved: int = Field(default=0, ge=0)

    @property
    def available(self) -> int:
        return self.quantity - self.reserved


# --------------------------------------------------------------------------
# Allocation (S8) - the citation requirement from CLAUDE.md section 2
# --------------------------------------------------------------------------


class Citation(Base):
    """Evidence backing one dispatch decision. Never optional."""

    claim: str = Field(description="What this source establishes, in one sentence")
    source_title: str
    source_url: str
    published: str | None = None

    @field_validator("source_url")
    @classmethod
    def _http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("citation source_url must be an http(s) URL")
        return v


class DispatchItem(Base):
    kind: ResourceKind
    sku: str
    quantity: int = Field(gt=0)


class DispatchOrder(Base):
    zone_id: PyObjectId
    from_node_id: PyObjectId
    items: list[DispatchItem] = Field(min_length=1)
    transport_mode: TransportMode = TransportMode.ROAD
    route: list[PyObjectId] = Field(
        default_factory=list, description="Node ids, origin -> destination"
    )
    priority: int = Field(default=5, ge=1, le=10)
    rationale: str
    # Structural enforcement of "Traceable AI": an uncited order cannot exist.
    citations: list[Citation] = Field(min_length=1)


class AllocationPlan(Base):
    dispatches: list[DispatchOrder] = Field(default_factory=list)
    summary: str = ""
    degraded: bool = Field(
        default=False,
        description="True when the agent plan failed validation and the "
        "deterministic baseline was used instead.",
    )
    violations: list[str] = Field(default_factory=list)


class Dispatch(DispatchOrder):
    id: PyObjectId = id_field()
    event_id: PyObjectId
    status: DispatchStatus = DispatchStatus.PROPOSED
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Research (S7)
# --------------------------------------------------------------------------


class Precedent(Base):
    """One normalised finding from the historical-disaster web search."""

    claim: str
    metric: str | None = None
    value: str | None = None
    source_title: str
    source_url: str
    published: str | None = None

    def as_citation(self) -> Citation:
        return Citation(
            claim=self.claim,
            source_title=self.source_title,
            source_url=self.source_url,
            published=self.published,
        )
