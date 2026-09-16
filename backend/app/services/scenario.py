"""Build an assessment from the real Nepal layers.

Real ward boundaries, real building footprints, real critical facilities, real (census-anchored)
population. The one synthetic part is the damage itself, because that needs pre/post imagery we
do not have yet - so it is modelled as a shaking field rather than invented per ward.

An earthquake's damage decays with distance from the epicentre, so the damage here does too.
That matters for more than realism: it means the ward ranking *emerges from geography* instead
of being hardcoded, which is the thing the priority score is supposed to be doing. Move the
epicentre and a different ward becomes the top priority.

Replaced by services/classify.py the moment imagery lands. Everything downstream - the joins,
the aggregation, the scoring, the review loop - is already the real implementation.

See PROJECT_PLAN.md sections 2 and 4.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from ..config import paths_config, priority_config, resolve
from ..schemas import AssessmentStatus, DamageClass, GeometrySource, InferenceMode
from ..store import Assessment, Building, Ward
from .gis import (  # noqa: F401
    WardGeometry,
    METRES_PER_DEGREE_LAT,
    assign_buildings_to_wards,
    facilities_near,
    load_facilities,
    load_geojson,
    load_wards,
    metres_per_degree_lon,
    ward_population_table,
)

# Central Kathmandu. Chosen so the demo AOI is inside the damage field; the Gorkha 2015
# epicentre was ~80 km north-west, which would leave the whole valley in the same band.
DEFAULT_EPICENTRE = (85.315, 27.705)
DEFAULT_RADIUS_KM = 4.0

# Damage mix at the epicentre and at the edge of the field. Buildings are sampled from an
# interpolation between the two, by shaking intensity.
# order: no-damage, minor, major, destroyed
# Calibrated against Nepal 2015: in the worst-hit dense wards roughly 40% of buildings took
# some damage and under 20% were badly hit. A model that flattens half a city reads as
# fantasy to anyone who was there.
MIX_AT_EPICENTRE = (0.58, 0.24, 0.12, 0.06)
MIX_AT_EDGE = (0.96, 0.035, 0.004, 0.001)

_CLASSES = [DamageClass.NO_DAMAGE, DamageClass.MINOR, DamageClass.MAJOR, DamageClass.DESTROYED]


def layers_available() -> bool:
    paths = paths_config()["nepal"]
    return all(resolve(paths[key]).exists() for key in ("wards", "facilities"))


def _intensity(lon: float, lat: float, epicentre: tuple[float, float], radius_km: float) -> float:
    """Shaking intensity, 1.0 at the epicentre decaying to ~0 at `radius_km`.

    Gaussian rather than linear: real ground motion attenuates smoothly, and a linear ramp
    puts a visible hard edge on the map that invites the wrong question at demo time.
    """
    lon_scale = metres_per_degree_lon(lat) / 1000.0
    dx = (lon - epicentre[0]) * lon_scale
    dy = (lat - epicentre[1]) * METRES_PER_DEGREE_LAT / 1000.0
    distance_km = math.hypot(dx, dy)
    return math.exp(-((distance_km / radius_km) ** 2))


def _mix_for(intensity: float) -> tuple[float, ...]:
    return tuple(
        edge + (centre - edge) * intensity
        for centre, edge in zip(MIX_AT_EPICENTRE, MIX_AT_EDGE)
    )


def _confidence(rng: random.Random, damage: DamageClass) -> float:
    """Least certain about minor damage - which is exactly where human review earns its keep."""
    base = {
        DamageClass.NO_DAMAGE: 0.88,
        DamageClass.MINOR: 0.62,
        DamageClass.MAJOR: 0.71,
        DamageClass.DESTROYED: 0.86,
    }[damage]
    return round(min(0.99, max(0.35, rng.gauss(base, 0.08))), 2)


def build_real_assessment(
    assessment_id: str,
    created_at: str,
    epicentre: tuple[float, float] = DEFAULT_EPICENTRE,
    radius_km: float = DEFAULT_RADIUS_KM,
    max_buildings: int = 15000,
    seed: int = 2015,
) -> Assessment:
    """Assemble wards, buildings, facilities and population into an Assessment."""
    paths = paths_config()["nepal"]
    rng = random.Random(seed)

    ward_geometries = load_wards(resolve(paths["wards"]))
    facilities = load_facilities(resolve(paths["facilities"]))

    buildings_path = resolve(paths.get("buildings", "data/nepal/buildings.geojson"))
    raw_buildings = load_geojson(buildings_path) if buildings_path.exists() else []

    # Cap for the browser: 40k polygons will not pan smoothly. Sample rather than truncate,
    # so the subset stays spatially representative instead of clipping to one corner.
    if max_buildings and len(raw_buildings) > max_buildings:
        raw_buildings = rng.sample(raw_buildings, max_buildings)

    by_ward = assign_buildings_to_wards(raw_buildings, ward_geometries)

    population_config = priority_config().get("population", {})
    calibration = population_config.get("calibration", {})
    populations = ward_population_table(
        resolve(paths["population"]),
        ward_geometries,
        calibrate_to=calibration.get("reference_total") if calibration.get("enabled") else None,
    )

    facility_radius = priority_config().get("facility_radius_m", 500)

    wards: list[Ward] = []
    buildings: list[Building] = []
    counter = 0

    for ward_geometry in ward_geometries:
        members = by_ward.get(ward_geometry.ward_id, [])
        if not members:
            continue

        damaged_points: list[tuple[float, float]] = []
        ward_buildings: list[Building] = []

        for feature in members:
            ring = feature["geometry"]["coordinates"][0]
            lon = sum(p[0] for p in ring) / len(ring)
            lat = sum(p[1] for p in ring) / len(ring)

            damage = rng.choices(_CLASSES, weights=_mix_for(_intensity(lon, lat, epicentre, radius_km)))[0]
            counter += 1
            ward_buildings.append(
                Building(
                    id=feature["properties"].get("id") or f"b_{counter:06d}",
                    ward_id=ward_geometry.ward_id,
                    geometry=feature["geometry"],
                    damage_class=damage,
                    confidence=_confidence(rng, damage),
                )
            )
            # Only severe damage pulls a facility into the score. A school 400 m from a
            # cracked wall is not a reason to divert the first team.
            if damage in (DamageClass.MAJOR, DamageClass.DESTROYED):
                damaged_points.append((lon, lat))

        buildings.extend(ward_buildings)

        ward_population = populations.get(ward_geometry.ward_id, 0)
        wards.append(
            Ward(
                ward_id=ward_geometry.ward_id,
                ward_name=ward_geometry.ward_name,
                geometry=ward_geometry.geometry,
                facility_types=facilities_near(facilities, damaged_points, facility_radius),
                # Affected population is apportioned from the ward's census-anchored total by
                # the share of its buildings that are damaged.
                people_per_building=(
                    ward_population / len(ward_buildings) if ward_buildings else 0.0
                ),
                road_disruption=0.0,
                distance_from_staging_km=round(
                    _distance_km(ward_geometry, epicentre), 2
                ),
            )
        )

    return Assessment(
        id=assessment_id,
        status=AssessmentStatus.COMPLETE,
        mode=InferenceMode.MOCK,
        geometry_source=GeometrySource.OSM,
        aoi_name="Kathmandu Metropolitan City",
        created_at=created_at,
        buildings=buildings,
        wards=wards,
    )


def _distance_km(ward: WardGeometry, point: tuple[float, float]) -> float:  # type: ignore[name-defined]
    centroid = ward.shapely_geometry.centroid
    lon_scale = metres_per_degree_lon(centroid.y) / 1000.0
    dx = (centroid.x - point[0]) * lon_scale
    dy = (centroid.y - point[1]) * METRES_PER_DEGREE_LAT / 1000.0
    return math.hypot(dx, dy)
