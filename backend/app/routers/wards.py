"""Ward cards - the brief's core output: where do we send teams first?"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import WardCard
from ..store import store

router = APIRouter(prefix="/api", tags=["wards"])


@router.get("/assessments/{assessment_id}/wards", response_model=list[WardCard])
def list_wards(assessment_id: str) -> list[WardCard]:
    """Ward cards, highest priority first. Recomputed on every request, so a human
    override is reflected immediately."""
    return _require(assessment_id).ward_cards()


@router.get("/assessments/{assessment_id}/wards/{ward_id}", response_model=WardCard)
def get_ward(assessment_id: str, ward_id: str) -> WardCard:
    for card in _require(assessment_id).ward_cards():
        if card.ward_id == ward_id:
            return card
    raise HTTPException(status_code=404, detail=f"No ward '{ward_id}' in '{assessment_id}'")


@router.get("/assessments/{assessment_id}/ward-boundaries")
def ward_boundaries(assessment_id: str) -> dict:
    """Ward polygons as GeoJSON, for the map's boundary layer."""
    assessment = _require(assessment_id)
    cards = {c.ward_id: c for c in assessment.ward_cards()}
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": ward.geometry,
                "properties": {
                    "ward_id": ward.ward_id,
                    "ward_name": ward.ward_name,
                    "priority_score": cards[ward.ward_id].priority_score,
                    "priority_band": cards[ward.ward_id].priority_band,
                    "priority_color": cards[ward.ward_id].priority_color,
                },
            }
            for ward in assessment.wards
            if ward.geometry and ward.ward_id in cards
        ],
    }


def _require(assessment_id: str):
    assessment = store.get(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment '{assessment_id}'")
    return assessment
