from pydantic import BaseModel, Field


class DamageCounts(BaseModel):
    none: int = 0
    minor: int = 0
    major: int = 0
    destroyed: int = 0


class BuildingCounts(BaseModel):
    """Distinct buildings per damage class, from connected-component labeling."""

    none: int = 0
    minor: int = 0
    major: int = 0
    destroyed: int = 0


class Zone(BaseModel):
    rank: int
    bbox: list[int] = Field(
        description="x, y, width, height in pixels", min_length=4, max_length=4
    )
    damage_counts: DamageCounts
    building_counts: BuildingCounts
    priority_score: float
    confidence: float | None = Field(
        default=None,
        description="Mean predicted-class probability over building pixels (pytorch mode only)",
    )
    centroid_lat: float | None = None
    centroid_lng: float | None = None


class AnalysisSummary(BaseModel):
    total_building_pixels: int
    total_buildings: int
    destroyed_pct: float
    major_pct: float
    minor_pct: float


class AnalysisResult(BaseModel):
    zones: list[Zone]
    summary: AnalysisSummary
    mask_path: str | None = None
    mask_base64: str | None = None
    pair_id: str | None = None
    inference_mode: str
    geo_available: bool = False
    geo_mode: str = Field(
        default="image",
        description="wgs84 for real lat/lng; image for pixel-space zone map",
    )
    image_size: list[int] | None = Field(
        default=None,
        description="Mask/image width and height in pixels [w, h]",
        min_length=2,
        max_length=2,
    )
    geo_message: str | None = None


class ResourceUnit(BaseModel):
    """One line of the responder roster the allocation plan has to divide up."""

    name: str = Field(max_length=60)
    quantity: int = Field(ge=0, le=10_000)
    unit: str = Field(default="teams", max_length=30)


DEFAULT_RESOURCES: list[ResourceUnit] = [
    ResourceUnit(name="Search & rescue teams", quantity=8, unit="teams"),
    ResourceUnit(name="Medical units", quantity=6, unit="units"),
    ResourceUnit(name="Heavy equipment", quantity=4, unit="machines"),
    ResourceUnit(name="Relief supply trucks", quantity=10, unit="trucks"),
    ResourceUnit(name="Shelter capacity", quantity=1200, unit="people"),
]


class BriefRequest(BaseModel):
    """Require a typed analysis payload so clients cannot dump unbounded JSON."""

    analysis: AnalysisResult
    context: str | None = Field(default=None, max_length=2000)
    resources: list[ResourceUnit] | None = Field(
        default=None,
        max_length=20,
        description="Responder roster to allocate. Falls back to DEFAULT_RESOURCES.",
    )


class ZoneRisk(BaseModel):
    """Per-zone risk read produced by the LLM from the deterministic counts."""

    zone_rank: int
    risk_level: str = Field(description="critical | high | moderate | low")
    risk_score: float = Field(ge=0, le=100)
    primary_hazards: list[str] = Field(default_factory=list, max_length=6)
    population_at_risk: str = ""
    access_notes: str = ""
    rationale: str = ""


class ResourceAssignment(BaseModel):
    """One allocation decision: how much of a resource goes to which zone."""

    zone_rank: int
    resource: str
    quantity: int = Field(ge=0)
    unit: str = ""
    urgency: str = Field(default="", description="immediate | urgent | scheduled")
    justification: str = ""


class BriefResponse(BaseModel):
    brief: str
    # gemini | gemini-fallback | stub
    source: str
    risk_analysis: list[ZoneRisk] = Field(default_factory=list)
    resource_allocation: list[ResourceAssignment] = Field(default_factory=list)
    reserve_notes: str = ""


class DemoPair(BaseModel):
    id: str
    disaster_type: str
    pre_image: str
    post_image: str


class ReportRequest(BaseModel):
    analysis: AnalysisResult
    brief: str
