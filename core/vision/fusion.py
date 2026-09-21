"""S3 - fuse overhead detections with ground reports into per-zone damage.

CLAUDE.md F1: *"When ground images and satellite detections overlap, weight the
ground image higher for severity classification, bypassing cloud-cover
limitations."*

Implemented as confidence-weighted combination with two properties:

* Satellite confidence is scaled by ``(1 - cloud_fraction)``. Under heavy cloud
  the overhead signal fades out on its own, rather than reporting confident
  nonsense about a scene it cannot see.
* Ground reports carry a fixed high weight (``settings.ground_weight``) and, when
  they disagree with the satellite, they win.

Every zone records ``decided_by``, so a dispatch can always be traced back to
the modality that produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import shape

from core.config import get_settings
from core.schemas import (
    DamageDetection,
    GroundReport,
    Severity,
)

log = logging.getLogger(__name__)

SEVERITY_SCORE: dict[str, float] = {
    Severity.NONE.value: 0.0,
    Severity.MINOR.value: 0.33,
    Severity.MAJOR.value: 0.66,
    Severity.DESTROYED.value: 1.0,
}



@dataclass(slots=True)
class ZoneDamage:
    """Fused damage state for one zone."""

    zone_id: str
    severity: Severity
    damage_score: float
    confidence: float
    buildings_destroyed: int
    detections: int
    ground_reports: int
    decided_by: str  # satellite | ground | fused | none
    damaged_area_m2: float = 0.0
    # Share of the ward damaged, from overhead damaged area only (0-1). None when
    # there is no overhead evidence. Severity can be overridden by a ground photo;
    # extent cannot - one photo shows one building, not how much of the ward fell.
    extent: float | None = None
    notes: list[str] = field(default_factory=list)


def _score_to_severity(score: float) -> Severity:
    if score < 0.15:
        return Severity.NONE
    if score < 0.35:
        return Severity.MINOR
    if score < 0.60:
        return Severity.MAJOR
    return Severity.DESTROYED


def _zone_area_m2(geometry: dict) -> float:
    """Approximate polygon area in m^2 for a WGS84 ring."""
    poly = shape(geometry)
    centroid_lat = poly.centroid.y
    from math import cos, radians

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(cos(radians(centroid_lat)), 0.01)
    return abs(poly.area) * m_per_deg_lat * m_per_deg_lon


def fuse_zone(
    zone_id: str,
    zone_geometry: dict,
    detections: list[DamageDetection],
    ground_reports: list[GroundReport],
    cloud_fraction: float = 0.0,
    gsd_m2_per_px: float | None = None,
) -> ZoneDamage:
    """Fuse both modalities for a single zone.

    `gsd_m2_per_px` is the imagery's ground sample area. When it is coarser
    than `settings.min_structure_gsd_m2` the building count is withheld rather
    than fabricated from an area figure the imagery cannot resolve.
    """
    settings = get_settings()
    polygon = shape(zone_geometry)
    zone_area = max(_zone_area_m2(zone_geometry), 1.0)

    # --- satellite side ---
    inside = [
        d for d in detections
        if polygon.contains(ShapelyPoint(*d.centroid.coordinates))
    ]
    damaged_area = 0.0
    sat_score = 0.0
    sat_confidence = 0.0
    if inside:
        for d in inside:
            weight = SEVERITY_SCORE.get(str(d.severity), 0.0)
            damaged_area += (d.area_m2 or 0.0) * weight
        fraction = damaged_area / zone_area
        sat_score = min(fraction / settings.damage_saturation_fraction, 1.0)
        sat_confidence = sum(d.confidence for d in inside) / len(inside)

    # Cloud cover is exactly why ground imagery exists in this system.
    sat_confidence *= max(0.0, 1.0 - cloud_fraction)

    # --- ground side ---
    ground_score = 0.0
    ground_confidence = 0.0
    if ground_reports:
        total_weight = sum(r.confidence for r in ground_reports) or 1e-6
        ground_score = sum(
            SEVERITY_SCORE.get(str(r.severity), 0.0) * r.confidence for r in ground_reports
        ) / total_weight
        ground_confidence = total_weight / len(ground_reports)

    # --- combine ---
    notes: list[str] = []
    gw = settings.ground_weight
    sat_effective = (1.0 - gw) * sat_confidence
    ground_effective = gw * ground_confidence

    if not inside and not ground_reports:
        return ZoneDamage(
            zone_id=zone_id, severity=Severity.NONE, damage_score=0.0, confidence=0.0,
            buildings_destroyed=0, detections=0, ground_reports=0, decided_by="none",
        )

    if ground_reports and not inside:
        score, confidence, decided_by = ground_score, ground_confidence, "ground"
        notes.append("no overhead detections; ground reports alone")
    elif inside and not ground_reports:
        score, confidence, decided_by = sat_score, sat_confidence, "satellite"
        if cloud_fraction > 0.3:
            notes.append(
                f"cloud_fraction={cloud_fraction:.2f} - confidence reduced, "
                "ground verification recommended"
            )
    else:
        total = sat_effective + ground_effective
        score = (
            (sat_score * sat_effective + ground_score * ground_effective) / total
            if total > 1e-6
            else max(sat_score, ground_score)
        )
        confidence = min(1.0, max(sat_confidence, ground_confidence) + 0.15)
        decided_by = "fused"
        if abs(ground_score - sat_score) > 0.3:
            # The explicit F1 rule: on disagreement, the photograph wins.
            score = ground_score
            decided_by = "ground"
            notes.append(
                f"ground ({ground_score:.2f}) overrode satellite ({sat_score:.2f}) "
                f"under cloud_fraction={cloud_fraction:.2f}"
            )

    severity = _score_to_severity(score)

    resolvable = gsd_m2_per_px is None or gsd_m2_per_px <= settings.min_structure_gsd_m2
    if not damaged_area:
        buildings = 0
    elif resolvable:
        buildings = int(round(damaged_area / settings.building_footprint_m2))
    else:
        # Reporting "10,865 buildings destroyed" from 35 m/px imagery would be
        # a fabrication. Damaged area is what this resolution supports.
        buildings = 0
        notes.append(
            f"imagery too coarse to resolve structures "
            f"({gsd_m2_per_px:.0f} m2/px); reporting damaged area only"
        )
    if not inside and ground_reports:
        buildings = 0
        notes.append("building count unavailable without overhead coverage")

    return ZoneDamage(
        zone_id=zone_id,
        severity=severity,
        damage_score=round(float(score), 4),
        confidence=round(float(min(confidence, 1.0)), 4),
        buildings_destroyed=buildings,
        detections=len(inside),
        ground_reports=len(ground_reports),
        decided_by=decided_by,
        damaged_area_m2=round(damaged_area, 1),
        extent=round(float(sat_score), 4) if inside else None,
        notes=notes,
    )


def fuse(
    zones: list[dict],
    detections: list[DamageDetection],
    ground_reports: list[GroundReport],
    cloud_fraction: float = 0.0,
    gsd_m2_per_px: float | None = None,
) -> list[ZoneDamage]:
    """Fuse every zone. `zones` are raw docs with `_id` and `geometry`."""
    by_zone: dict[str, list[GroundReport]] = {}
    unassigned: list[GroundReport] = []
    for report in ground_reports:
        if report.zone_id:
            by_zone.setdefault(str(report.zone_id), []).append(report)
        else:
            unassigned.append(report)

    # A report without a zone still has coordinates; place it geometrically.
    for report in unassigned:
        point = ShapelyPoint(*report.location.coordinates)
        for zone in zones:
            if shape(zone["geometry"]).contains(point):
                by_zone.setdefault(str(zone["_id"]), []).append(report)
                break

    results = [
        fuse_zone(
            zone_id=str(zone["_id"]),
            zone_geometry=zone["geometry"],
            detections=detections,
            ground_reports=by_zone.get(str(zone["_id"]), []),
            cloud_fraction=cloud_fraction,
            gsd_m2_per_px=gsd_m2_per_px,
        )
        for zone in zones
    ]
    log.info(
        "S3 fusion: %d zones, %d detections, %d ground reports, cloud=%.2f",
        len(zones), len(detections), len(ground_reports), cloud_fraction,
    )
    return results
