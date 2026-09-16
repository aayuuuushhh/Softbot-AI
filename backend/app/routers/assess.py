"""Assessment lifecycle: submit a pre/post pair, poll status, read the summary."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from ..schemas import AssessRequest, AssessmentSummary, InferenceMode
from ..services.mock_data import build_mock_assessment
from ..services.scenario import DEFAULT_EPICENTRE, build_real_assessment, layers_available
from ..store import store

router = APIRouter(prefix="/api", tags=["assessments"])


@router.post("/assess", response_model=AssessmentSummary, status_code=202)
def create_assessment(request: AssessRequest) -> AssessmentSummary:
    """Run an assessment over a pre/post image pair.

    MOCK mode returns a synthetic Kathmandu AOI immediately. MODEL and HEURISTIC modes are
    wired to the real pipeline in H24-H40; until then they are rejected rather than silently
    falling back, so nobody demos fake data by accident.
    """
    if request.mode is not InferenceMode.MOCK:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Inference mode '{request.mode.value}' is not wired up yet. "
                "Use mode='mock' until services/detect.py and services/classify.py land."
            ),
        )

    created = store.create(mode=request.mode, aoi_name=request.aoi_name)
    assessment = build_mock_assessment(
        assessment_id=created.id,
        aoi_name=request.aoi_name or "Kathmandu Valley (demo AOI)",
        created_at=created.created_at,
    )
    store.add(assessment)
    return assessment.summary()


@router.get("/assessments", response_model=list[AssessmentSummary])
def list_assessments() -> list[AssessmentSummary]:
    return [a.summary() for a in store.list()]


@router.get("/assessments/{assessment_id}", response_model=AssessmentSummary)
def get_assessment(assessment_id: str) -> AssessmentSummary:
    return _require(assessment_id).summary()


@router.post("/assessments/demo", response_model=AssessmentSummary, status_code=201)
def create_demo_assessment(
    synthetic: bool = Query(False, description="Force the generated grid even if OSM layers exist"),
    epicentre_lon: float | None = Query(None),
    epicentre_lat: float | None = Query(None),
    radius_km: float = Query(4.0, gt=0, le=50),
) -> AssessmentSummary:
    """One-click demo data.

    Uses the real OSM/WorldPop layers when they have been fetched (`scripts/fetch_osm.py`),
    and falls back to the generated grid otherwise so the dashboard always has something to
    show. Damage is simulated in both cases - `geometry_source` says which is which.

    Moving the epicentre re-ranks the wards, which is the point: the priority comes out of
    geography, not out of a hardcoded table.
    """
    created = store.create(mode=InferenceMode.MOCK)

    if layers_available() and not synthetic:
        epicentre = (
            epicentre_lon if epicentre_lon is not None else DEFAULT_EPICENTRE[0],
            epicentre_lat if epicentre_lat is not None else DEFAULT_EPICENTRE[1],
        )
        assessment = build_real_assessment(
            assessment_id=created.id,
            created_at=created.created_at,
            epicentre=epicentre,
            radius_km=radius_km,
        )
    else:
        assessment = build_mock_assessment(
            assessment_id=created.id,
            aoi_name="Kathmandu Valley (synthetic grid)",
            created_at=created.created_at,
        )

    store.add(assessment)
    return assessment.summary()


def _require(assessment_id: str):
    assessment = store.get(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment '{assessment_id}'")
    return assessment
