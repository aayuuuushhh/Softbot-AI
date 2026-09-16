"""Real spatial operations on the Nepal layers.

Three joins the priority score depends on:
  buildings -> wards        point-in-polygon, so damage aggregates to a dispatch unit
  facilities -> damage      what critical infrastructure sits near the destruction
  population -> ward        WorldPop zonal sum, the real version of the demo's proxy

Uses shapely + rasterio directly rather than geopandas: we need three operations, not a
dataframe library, and the import is a second faster on a cold start.

Distances are metric. Nepal sits in UTM 45N (EPSG:32645); rather than reproject every
geometry we scale degrees locally, which is accurate to well under a metre at ward scale
and avoids a pyproj round-trip per building.

See PROJECT_PLAN.md section 2 and 4.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import shape
from shapely.prepared import prep
from shapely.strtree import STRtree

METRES_PER_DEGREE_LAT = 110_540.0


@dataclass
class WardGeometry:
    ward_id: str
    ward_name: str
    full_name: str
    municipality: str
    geometry: dict          # GeoJSON
    shapely_geometry: object


@dataclass
class Facility:
    facility_type: str
    name: str
    lon: float
    lat: float


def metres_per_degree_lon(latitude: float) -> float:
    return 111_320.0 * math.cos(math.radians(latitude))


def degrees_for_metres(metres: float, latitude: float) -> tuple[float, float]:
    """(lon_degrees, lat_degrees) covering `metres` at this latitude."""
    return metres / metres_per_degree_lon(latitude), metres / METRES_PER_DEGREE_LAT


# --- Loading --------------------------------------------------------------


def load_geojson(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["features"]


def load_wards(path: Path) -> list[WardGeometry]:
    wards = []
    for feature in load_geojson(path):
        properties = feature["properties"]
        wards.append(
            WardGeometry(
                ward_id=properties["ward_id"],
                ward_name=properties.get("ward_name", properties.get("full_name", "Ward")),
                full_name=properties.get("full_name", ""),
                municipality=properties.get("municipality", ""),
                geometry=feature["geometry"],
                shapely_geometry=shape(feature["geometry"]),
            )
        )
    return wards


def load_facilities(path: Path) -> list[Facility]:
    facilities = []
    for feature in load_geojson(path):
        lon, lat = feature["geometry"]["coordinates"][:2]
        properties = feature["properties"]
        facilities.append(
            Facility(
                facility_type=properties["facility_type"],
                name=properties.get("name") or properties["facility_type"].title(),
                lon=lon,
                lat=lat,
            )
        )
    return facilities


# --- Joins ----------------------------------------------------------------


def assign_buildings_to_wards(
    buildings: list[dict], wards: list[WardGeometry]
) -> dict[str, list[dict]]:
    """Point-in-polygon on each building's centroid, via an R-tree.

    Buildings outside every ward (the bbox always overhangs the boundary) are dropped -
    a building we cannot dispatch a team to is not useful to rank.
    """
    if not wards:
        return {}

    geometries = [w.shapely_geometry for w in wards]
    tree = STRtree(geometries)
    prepared = [prep(g) for g in geometries]

    assigned: dict[str, list[dict]] = {w.ward_id: [] for w in wards}
    for feature in buildings:
        point = shape(feature["geometry"]).centroid
        for index in tree.query(point):
            if prepared[index].contains(point):
                assigned[wards[index].ward_id].append(feature)
                break
    return assigned


def facilities_near(
    facilities: list[Facility],
    points: list[tuple[float, float]],
    radius_m: float,
) -> list[str]:
    """Facility types within `radius_m` of any of `points` (damaged building centroids).

    Returns each matching facility's type once per facility, so a ward with three hospitals
    scores higher than a ward with one.
    """
    if not facilities or not points:
        return []

    latitude = sum(p[1] for p in points) / len(points)
    lon_scale = metres_per_degree_lon(latitude)

    tree = STRtree([shape({"type": "Point", "coordinates": [p[0], p[1]]}) for p in points])
    lon_radius, lat_radius = degrees_for_metres(radius_m, latitude)

    found: list[str] = []
    for facility in facilities:
        box = shape({
            "type": "Polygon",
            "coordinates": [[
                [facility.lon - lon_radius, facility.lat - lat_radius],
                [facility.lon + lon_radius, facility.lat - lat_radius],
                [facility.lon + lon_radius, facility.lat + lat_radius],
                [facility.lon - lon_radius, facility.lat + lat_radius],
                [facility.lon - lon_radius, facility.lat - lat_radius],
            ]],
        })
        for index in tree.query(box):
            point = points[index]
            dx = (point[0] - facility.lon) * lon_scale
            dy = (point[1] - facility.lat) * METRES_PER_DEGREE_LAT
            if dx * dx + dy * dy <= radius_m * radius_m:
                found.append(facility.facility_type)
                break
    return found


def population_in_ward(raster_path: Path, ward: WardGeometry) -> int | None:
    """WorldPop zonal sum over a ward polygon.

    Returns None when the raster is missing, so callers can fall back to the proxy rather
    than silently reporting zero people affected.
    """
    if not Path(raster_path).exists():
        return None

    try:
        import numpy as np
        import rasterio
        from rasterio.mask import mask

        with rasterio.open(raster_path) as source:
            clipped, _ = mask(source, [ward.geometry], crop=True, filled=True, nodata=0)
            values = np.asarray(clipped, dtype="float64")
            # WorldPop marks no-data with a large negative sentinel.
            values[values < 0] = 0
            return int(round(float(values.sum())))
    except Exception:
        return None


def ward_population_table(
    raster_path: Path,
    wards: list[WardGeometry],
    calibrate_to: int | None = None,
) -> dict[str, int]:
    """Population per ward, optionally anchored to a census total for the whole AOI.

    `calibrate_to` scales every ward by one factor so the AOI sums to a known census figure.
    See the `population.calibration` note in config/priority.yaml for why this is needed:
    WorldPop's constrained raster is nationally sound but locally over-concentrated, and an
    "estimated affected population" that reads 3x high is worse than useless to a responder.
    A single scale factor preserves the relative distribution, which is what the score uses.
    """
    raw: dict[str, int] = {}
    for ward in wards:
        population = population_in_ward(raster_path, ward)
        if population is not None:
            raw[ward.ward_id] = population

    if not calibrate_to or not raw:
        return raw

    total = sum(raw.values())
    if total <= 0:
        return raw

    scale = calibrate_to / total
    return {ward_id: int(round(value * scale)) for ward_id, value in raw.items()}
