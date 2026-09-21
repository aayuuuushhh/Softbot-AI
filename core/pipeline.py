"""Orchestration for the perception stages (S1 -> S2 -> S3).

Keeps the FastAPI routes thin: they handle upload and serialisation, this
handles ordering, persistence and stage bookkeeping.
"""

from __future__ import annotations

import logging
from pathlib import Path

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import shape

from core import db
from core.config import get_settings
from core.schemas import (
    DamageDetection,
    GroundReport,
    Severity,
    StageStatus,
    utcnow,
)
from core.vision.fusion import fuse
from core.vision.inference import get_ground_backend, get_satellite_backend
from core.vision.preprocess import (
    estimate_cloud_fraction,
    pixel_area_m2,
    read_geotiff,
)
from core.vision.render import render_triptych, to_png_bytes

log = logging.getLogger(__name__)


def event_dir(event_id: str) -> Path:
    path = get_settings().upload_dir / event_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _set_stage(event_id: str, stage: str, status: StageStatus, message: str | None = None):
    update = {f"stages.{stage}": status.value}
    if message is not None:
        update["stages.message"] = message
    await db.get_db()[db.EVENTS].update_one({"_id": db.oid(event_id)}, {"$set": update})


async def run_overhead_analysis(event_id: str) -> dict:
    """S1 + S3: change-detect the stored pre/post pair, then fuse into zones.

    Ground reports uploaded before this runs are picked up by the fusion step,
    so the order in which the two modalities arrive does not matter.
    """
    directory = event_dir(event_id)
    pre_path, post_path = directory / "pre.tif", directory / "post.tif"
    if not pre_path.exists() or not post_path.exists():
        raise FileNotFoundError("pre.tif and post.tif must be uploaded first")

    await _set_stage(event_id, "overhead", StageStatus.RUNNING)
    try:
        pre = read_geotiff(pre_path)
        post = read_geotiff(post_path)
        cloud = estimate_cloud_fraction(post.image)
        gsd = pixel_area_m2(pre.transform, pre.crs)

        backend = get_satellite_backend()
        detections = backend.detect(pre, post)
        log.info("S1 (%s): %d detections, cloud=%.3f", backend.name, len(detections), cloud)

        database = db.get_db()
        await database[db.DETECTIONS].delete_many({"event_id": event_id})
        if detections:
            await database[db.DETECTIONS].insert_many(
                [{"event_id": event_id, **d.model_dump()} for d in detections]
            )

        # Cache the overlay so /overlay.png does not re-run inference.
        (directory / "overlay.png").write_bytes(
            to_png_bytes(render_triptych(pre.image, post.image, detections))
        )

        await database[db.EVENTS].update_one(
            {"_id": db.oid(event_id)},
            {"$set": {"cloud_fraction": round(float(cloud), 4), "crs": pre.crs}},
        )
        await _set_stage(event_id, "overhead", StageStatus.DONE)
    except Exception as exc:
        await _set_stage(event_id, "overhead", StageStatus.FAILED, str(exc))
        log.exception("S1 failed for event %s", event_id)
        raise

    fused = await run_fusion(event_id)
    return {
        "detections": len(detections),
        "cloud_fraction": round(float(cloud), 4),
        "gsd_m2_per_px": round(gsd, 1),
        "backend": backend.name,
        "zones_updated": len(fused),
    }


async def run_fusion(event_id: str) -> list[dict]:
    """S3: combine stored detections and ground reports into zone damage state."""
    await _set_stage(event_id, "fusion", StageStatus.RUNNING)
    try:
        database = db.get_db()
        event = await database[db.EVENTS].find_one({"_id": db.oid(event_id)})
        if event is None:
            raise ValueError(f"no such event: {event_id}")

        zones = [z async for z in database[db.ZONES].find({"event_id": event_id})]
        detections = [
            DamageDetection.model_validate(db.doc_out(d))
            async for d in database[db.DETECTIONS].find({"event_id": event_id})
        ]
        reports = [
            GroundReport.model_validate(db.doc_out(r))
            async for r in database[db.GROUND_REPORTS].find({"event_id": event_id})
        ]

        gsd = None
        pre_path = event_dir(event_id) / "pre.tif"
        if pre_path.exists():
            raster = read_geotiff(pre_path)
            gsd = pixel_area_m2(raster.transform, raster.crs)

        results = fuse(
            zones, detections, reports,
            cloud_fraction=float(event.get("cloud_fraction", 0.0)),
            gsd_m2_per_px=gsd,
        )

        out = []
        for zd in results:
            await database[db.ZONES].update_one(
                {"_id": db.oid(zd.zone_id)},
                {"$set": {
                    "severity": zd.severity.value,
                    "damage_score": zd.damage_score,
                    "buildings_destroyed": zd.buildings_destroyed,
                    "detections": zd.detections,
                    "decided_by": zd.decided_by,
                }},
            )
            out.append({
                "zone_id": zd.zone_id, "severity": zd.severity.value,
                "damage_score": zd.damage_score, "confidence": zd.confidence,
                "detections": zd.detections, "ground_reports": zd.ground_reports,
                "decided_by": zd.decided_by, "damaged_area_m2": zd.damaged_area_m2,
                "notes": zd.notes,
            })

        await _set_stage(event_id, "fusion", StageStatus.DONE)
        return out
    except Exception as exc:
        await _set_stage(event_id, "fusion", StageStatus.FAILED, str(exc))
        log.exception("S3 failed for event %s", event_id)
        raise


async def ingest_ground_image(
    event_id: str, image_path: Path, lon: float, lat: float, reporter: str | None = None
) -> GroundReport:
    """S2: classify a field photo, attach it to the zone containing it, re-fuse."""
    await _set_stage(event_id, "ground", StageStatus.RUNNING)
    try:
        backend = get_ground_backend()
        severity, confidence = backend.classify(str(image_path))

        database = db.get_db()
        point = ShapelyPoint(lon, lat)
        zone_id = None
        async for zone in database[db.ZONES].find({"event_id": event_id}):
            if shape(zone["geometry"]).contains(point):
                zone_id = str(zone["_id"])
                break

        doc = {
            "event_id": event_id,
            "zone_id": zone_id,
            "image_path": str(image_path),
            "location": {"type": "Point", "coordinates": [lon, lat]},
            "severity": severity.value,
            "confidence": round(float(confidence), 4),
            "model_version": backend.name,
            "reporter": reporter,
            "uploaded_at": utcnow(),
        }
        doc["_id"] = (await database[db.GROUND_REPORTS].insert_one(doc)).inserted_id
        await _set_stage(event_id, "ground", StageStatus.DONE)
        log.info(
            "S2 (%s): %s conf=%.2f at %.4f,%.4f -> zone %s",
            backend.name, severity.value, confidence, lon, lat, zone_id,
        )
    except Exception as exc:
        await _set_stage(event_id, "ground", StageStatus.FAILED, str(exc))
        raise

    # A new ground report can override the satellite verdict for its zone.
    await run_fusion(event_id)
    return GroundReport.model_validate(db.doc_out(doc))


async def load_detections(event_id: str) -> list[DamageDetection]:
    return [
        DamageDetection.model_validate(db.doc_out(d))
        async for d in db.get_db()[db.DETECTIONS].find({"event_id": event_id})
    ]


async def severity_breakdown(event_id: str) -> dict[str, int]:
    counts = {s.value: 0 for s in Severity}
    async for zone in db.get_db()[db.ZONES].find({"event_id": event_id}):
        counts[zone.get("severity", Severity.NONE.value)] += 1
    return counts
