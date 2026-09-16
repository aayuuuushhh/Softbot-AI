"""Building footprints and human-in-the-loop overrides."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..schemas import BuildingOverride, BuildingProperties, FeatureCollection, WardCard
from ..store import store

router = APIRouter(prefix="/api", tags=["buildings"])


@router.get("/assessments/{assessment_id}/buildings", response_model=FeatureCollection)
def list_buildings(
    assessment_id: str,
    ward_id: str | None = Query(None, description="Restrict to one ward"),
) -> FeatureCollection:
    assessment = store.get(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment '{assessment_id}'")
    return assessment.feature_collection(ward_id=ward_id)


@router.patch("/buildings/{building_id}", response_model=WardCard)
def override_building(building_id: str, override: BuildingOverride) -> WardCard:
    """A human official corrects the AI.

    Returns the affected ward's re-scored card, so the dashboard can update the priority
    ranking in one round-trip. This is the human-in-the-loop guarantee from the brief:
    the AI does the first pass, people decide.
    """
    hit = store.get_building(building_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"No building '{building_id}'")
    assessment, building = hit

    building.damage_class = override.damage_class
    building.reviewed_by_human = True
    building.review_note = override.note
    building.confidence = 1.0  # a human looked at it

    for card in assessment.ward_cards():
        if card.ward_id == building.ward_id:
            return card
    raise HTTPException(status_code=500, detail=f"Ward '{building.ward_id}' vanished during re-score")


@router.get("/buildings/{building_id}", response_model=BuildingProperties)
def get_building(building_id: str) -> BuildingProperties:
    hit = store.get_building(building_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"No building '{building_id}'")
    _, b = hit
    return BuildingProperties(
        id=b.id,
        ward_id=b.ward_id,
        damage_class=b.damage_class,
        confidence=b.confidence,
        reviewed_by_human=b.reviewed_by_human,
        review_note=b.review_note,
    )
