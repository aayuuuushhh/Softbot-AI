"""Synthetic Kathmandu Valley assessment.

Lets the frontend, the ward aggregation and the priority scoring be built and demoed before any
imagery or model exists (PROJECT_PLAN.md section 6, gate H0-H8). Deterministic: the same seed
gives the same map every run, so the demo never surprises you.

Replaced by the real pipeline in `services/detect.py` + `services/classify.py`; everything
downstream of this module is production code already.
"""

from __future__ import annotations

import random

from ..schemas import AssessmentStatus, DamageClass, InferenceMode
from ..store import Assessment, Building, Ward

# A ward-sized grid over the Kathmandu Valley.
ORIGIN_LON, ORIGIN_LAT = 85.290, 27.680
WARD_W, WARD_H = 0.018, 0.014
GRID_COLS = 4

# ward number -> (buildings, damage profile, facilities, road disruption, km from staging)
# Ward 8 is the brief's worked example, so it is the clear Very High.
WARD_PROFILES = {
    1: (140, (0.80, 0.14, 0.04, 0.02), ["school"], 0.05, 1.2),
    2: (165, (0.72, 0.19, 0.06, 0.03), [], 0.10, 2.0),
    3: (120, (0.86, 0.10, 0.03, 0.01), ["school", "police"], 0.05, 2.6),
    4: (190, (0.60, 0.25, 0.10, 0.05), ["bridge"], 0.35, 3.4),
    5: (110, (0.90, 0.08, 0.015, 0.005), [], 0.00, 1.8),
    6: (175, (0.66, 0.22, 0.08, 0.04), ["school", "bridge"], 0.20, 4.1),
    7: (130, (0.82, 0.13, 0.035, 0.015), ["hospital"], 0.10, 2.2),
    8: (205, (0.38, 0.45, 0.11, 0.06), ["hospital", "school", "bridge"], 0.45, 5.0),
}

# People affected per damaged building. Kathmandu's core is multi-storey and
# multi-household, so the urban wards carry far more people per footprint.
# A proxy until the WorldPop zonal sum lands - PROJECT_PLAN.md section 4.
PEOPLE_PER_BUILDING = {8: 19.0, 4: 14.0, 6: 12.0, 2: 10.0}
DEFAULT_PEOPLE_PER_BUILDING = 8.0

_CLASSES = [DamageClass.NO_DAMAGE, DamageClass.MINOR, DamageClass.MAJOR, DamageClass.DESTROYED]


def _ward_bounds(number: int) -> tuple[float, float, float, float]:
    col = (number - 1) % GRID_COLS
    row = (number - 1) // GRID_COLS
    min_lon = ORIGIN_LON + col * WARD_W
    min_lat = ORIGIN_LAT + row * WARD_H
    return min_lon, min_lat, min_lon + WARD_W, min_lat + WARD_H


def _rect(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat], [max_lon, min_lat],
            [max_lon, max_lat], [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }


def _confidence(rng: random.Random, damage: DamageClass) -> float:
    """The model is most sure about the extremes, least sure about minor damage -
    which is exactly where human review matters most."""
    base = {
        DamageClass.NO_DAMAGE: 0.88,
        DamageClass.MINOR: 0.62,
        DamageClass.MAJOR: 0.71,
        DamageClass.DESTROYED: 0.86,
    }[damage]
    return round(min(0.99, max(0.35, rng.gauss(base, 0.08))), 2)


def build_mock_assessment(
    assessment_id: str,
    aoi_name: str = "Kathmandu Valley (demo AOI)",
    seed: int = 2015,
    created_at: str = "",
) -> Assessment:
    rng = random.Random(seed)
    wards: list[Ward] = []
    buildings: list[Building] = []
    counter = 0

    for number, (count, profile, facilities, disruption, distance) in WARD_PROFILES.items():
        min_lon, min_lat, max_lon, max_lat = _ward_bounds(number)
        ward_id = f"ktm-{number:02d}"
        wards.append(
            Ward(
                ward_id=ward_id,
                ward_name=f"Ward {number}",
                geometry=_rect(min_lon, min_lat, max_lon, max_lat),
                facility_types=facilities,
                people_per_building=PEOPLE_PER_BUILDING.get(number, DEFAULT_PEOPLE_PER_BUILDING),
                road_disruption=disruption,
                distance_from_staging_km=distance,
            )
        )

        for _ in range(count):
            # Keep footprints off the ward edge so nothing straddles a boundary.
            lon = rng.uniform(min_lon + 0.0008, max_lon - 0.0010)
            lat = rng.uniform(min_lat + 0.0008, max_lat - 0.0008)
            w = rng.uniform(0.00012, 0.00028)
            h = rng.uniform(0.00010, 0.00022)
            damage = rng.choices(_CLASSES, weights=profile)[0]
            counter += 1
            buildings.append(
                Building(
                    id=f"b_{counter:05d}",
                    ward_id=ward_id,
                    geometry=_rect(lon, lat, lon + w, lat + h),
                    damage_class=damage,
                    confidence=_confidence(rng, damage),
                )
            )

    return Assessment(
        id=assessment_id,
        status=AssessmentStatus.COMPLETE,
        mode=InferenceMode.MOCK,
        aoi_name=aoi_name,
        created_at=created_at,
        buildings=buildings,
        wards=wards,
    )
