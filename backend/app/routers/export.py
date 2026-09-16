"""Responder-ready exports: GeoJSON for GIS, CSV for a spreadsheet in a field office."""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ..store import store

router = APIRouter(prefix="/api", tags=["export"])


@router.get("/assessments/{assessment_id}/export")
def export_assessment(
    assessment_id: str,
    format: str = Query("geojson", pattern="^(geojson|csv)$"),
    layer: str = Query("buildings", pattern="^(buildings|wards)$"),
) -> Response:
    assessment = store.get(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment '{assessment_id}'")

    stem = f"{assessment_id}_{layer}"

    if layer == "buildings" and format == "geojson":
        body = assessment.feature_collection().model_dump_json()
        return _download(body, "application/geo+json", f"{stem}.geojson")

    if layer == "buildings":
        rows = [
            {
                "building_id": f.properties.id,
                "ward_id": f.properties.ward_id,
                "damage_class": f.properties.damage_class.value,
                "confidence": f.properties.confidence,
                "reviewed_by_human": f.properties.reviewed_by_human,
            }
            for f in assessment.feature_collection().features
        ]
        return _download(_to_csv(rows), "text/csv", f"{stem}.csv")

    cards = assessment.ward_cards()
    if format == "geojson":
        # Ward polygons carrying the full card, so QGIS users get the priority map directly.
        by_id = {c.ward_id: c for c in cards}
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": w.geometry,
                    "properties": json.loads(by_id[w.ward_id].model_dump_json()),
                }
                for w in assessment.wards
                if w.geometry and w.ward_id in by_id
            ],
        }
        return _download(json.dumps(payload), "application/geo+json", f"{stem}.geojson")

    rows = [
        {
            "ward_id": c.ward_id,
            "ward_name": c.ward_name,
            "total_buildings": c.total_buildings,
            "damaged_buildings": c.damaged_buildings,
            "severely_damaged": c.severely_damaged,
            "affected_population": c.affected_population,
            "critical_facilities": c.critical_facilities,
            "facility_types": "|".join(c.facility_types),
            "priority_score": c.priority_score,
            "priority_band": c.priority_band,
            "reviewed_count": c.reviewed_count,
        }
        for c in cards
    ]
    return _download(_to_csv(rows), "text/csv", f"{stem}.csv")


def _to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _download(body: str, media_type: str, filename: str) -> Response:
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
