"""Ground image ingestion (S2) and satellite pair analysis (S1)."""

from __future__ import annotations

import logging
import shutil

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from core import db, pipeline
from core.schemas import GroundReport
from core.vision.render import detections_to_geojson

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["perception"])

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


async def _require_event(event_id: str) -> dict:
    doc = await db.get_db()[db.EVENTS].find_one({"_id": db.oid(event_id)})
    if doc is None:
        raise HTTPException(404, f"no such event: {event_id}")
    return doc


@router.post("/events/{event_id}/analyze", status_code=200)
async def analyze_pair(
    event_id: str,
    pre: UploadFile = File(..., description="pre-event GeoTIFF"),
    post: UploadFile = File(..., description="post-event GeoTIFF"),
) -> dict:
    """S1 + S3: change-detect a pre/post satellite pair and fuse into zones.

    Runs synchronously: the stub backend finishes a 1024x1024 pair in under a
    second, and a caller that gets a 200 knows the damage map is queryable.
    """
    await _require_event(event_id)
    directory = pipeline.event_dir(event_id)

    for upload, name in ((pre, "pre.tif"), (post, "post.tif")):
        target = directory / name
        with target.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh, length=1024 * 1024)
        if target.stat().st_size > MAX_UPLOAD_BYTES:
            target.unlink()
            raise HTTPException(413, f"{name} exceeds {MAX_UPLOAD_BYTES // 1024**2} MiB")

    try:
        result = await pipeline.run_overhead_analysis(event_id)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("analysis failed")
        raise HTTPException(500, f"analysis failed: {exc}") from exc

    result["severity_breakdown"] = await pipeline.severity_breakdown(event_id)
    return result


@router.post("/upload/ground", response_model=GroundReport, status_code=201)
async def upload_ground_image(
    event_id: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    reporter: str | None = Form(None),
    image: UploadFile = File(...),
) -> GroundReport:
    """S2: a field worker uploads a photo of a damaged structure.

    The report is classified, attached to whichever zone contains it, and
    fusion re-runs immediately — so a ground photo can override a cloud-obscured
    satellite verdict within one request.
    """
    await _require_event(event_id)
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(422, f"coordinates out of range: {lat},{lon}")

    directory = pipeline.event_dir(event_id) / "ground"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = (image.filename or "upload.jpg").split(".")[-1][:8]
    target = directory / f"{lat:.5f}_{lon:.5f}.{suffix}"
    with target.open("wb") as fh:
        shutil.copyfileobj(image.file, fh, length=1024 * 1024)

    try:
        return await pipeline.ingest_ground_image(event_id, target, lon, lat, reporter)
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc


@router.get("/events/{event_id}/damage")
async def damage_geojson(event_id: str) -> dict:
    """S3 output as GeoJSON: detections plus per-zone fused severity."""
    await _require_event(event_id)
    detections = await pipeline.load_detections(event_id)

    zone_features = []
    async for zone in db.get_db()[db.ZONES].find({"event_id": event_id}):
        zone_features.append({
            "type": "Feature",
            "geometry": zone["geometry"],
            "properties": {
                "id": str(zone["_id"]),
                "name": zone["name"],
                "severity": zone.get("severity", "none"),
                "damage_score": zone.get("damage_score", 0.0),
                "buildings_destroyed": zone.get("buildings_destroyed", 0),
                "detections": zone.get("detections", 0),
                "decided_by": zone.get("decided_by", "none"),
                "population": zone.get("population", 0),
            },
        })

    return {
        "detections": detections_to_geojson(detections),
        "zones": {"type": "FeatureCollection", "features": zone_features},
        "severity_breakdown": await pipeline.severity_breakdown(event_id),
    }


@router.get("/events/{event_id}/overlay.png", response_class=Response)
async def damage_overlay(event_id: str) -> Response:
    """Cached pre | post | damage triptych from the last analysis run."""
    await _require_event(event_id)
    path = pipeline.event_dir(event_id) / "overlay.png"
    if not path.exists():
        raise HTTPException(404, "no overlay yet - POST to /analyze first")
    return Response(content=path.read_bytes(), media_type="image/png")


@router.get("/events/{event_id}/ground-reports", response_model=list[GroundReport])
async def list_ground_reports(event_id: str) -> list[GroundReport]:
    cursor = db.get_db()[db.GROUND_REPORTS].find({"event_id": event_id})
    return [GroundReport.model_validate(db.doc_out(d)) async for d in cursor]
