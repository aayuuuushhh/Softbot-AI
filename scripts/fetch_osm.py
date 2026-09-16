#!/usr/bin/env python3
"""Fetch real Nepal GIS layers from OpenStreetMap via the Overpass API.

    python scripts/fetch_osm.py --all                      # wards + facilities + buildings
    python scripts/fetch_osm.py --wards --municipality Kathmandu

Replaces the 200 MB Geofabrik extract and the GDAL dependency: Overpass returns exactly the
AOI we ask for, already as JSON.

Nepal's wards are `admin_level=9` relations named "<Municipality>-<NN>" (e.g. "Kathmandu-08"),
which is also our aggregation unit - so this is the real version of what the demo currently
fakes.

Data (c) OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass rejects requests without one.
USER_AGENT = "softbot-disaster-assessment/0.1 (hackathon project)"

# Kathmandu Valley. south, west, north, east - Overpass's order.
DEFAULT_BBOX = (27.67, 85.28, 27.76, 85.38)

FACILITY_AMENITIES = ["hospital", "clinic", "school", "police", "fire_station"]

# Maps an OSM tag to the keys used in config/priority.yaml.
AMENITY_TO_TYPE = {
    "hospital": "hospital",
    "clinic": "hospital",  # scored the same: it is where casualties go
    "school": "school",
    "police": "police",
    "fire_station": "fire_station",
}


def query(overpass_ql: str, retries: int = 3) -> dict:
    """POST an Overpass QL query. Retries - the public instance rate-limits under load."""
    for attempt in range(1, retries + 1):
        try:
            request = Request(
                OVERPASS_URL,
                data=urlencode({"data": overpass_ql}).encode(),
                headers={"User-Agent": USER_AGENT},
            )
            with urlopen(request, timeout=180) as response:
                return json.loads(response.read())
        except Exception as error:  # noqa: BLE001 - any failure is worth one more try
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"  attempt {attempt} failed ({error}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# --- Geometry -------------------------------------------------------------


def stitch_rings(ways: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join relation member ways into closed rings.

    Overpass returns a boundary relation as loose way fragments in arbitrary order and
    direction. Each ring is built by repeatedly attaching whichever remaining fragment shares
    an endpoint, flipping it when necessary.
    """
    remaining = [w for w in ways if len(w) >= 2]
    rings: list[list[tuple[float, float]]] = []

    while remaining:
        ring = remaining.pop(0)
        extended = True
        while extended and ring[0] != ring[-1]:
            extended = False
            for index, candidate in enumerate(remaining):
                if candidate[0] == ring[-1]:
                    ring = ring + candidate[1:]
                elif candidate[-1] == ring[-1]:
                    ring = ring + candidate[-2::-1]
                elif candidate[-1] == ring[0]:
                    ring = candidate[:-1] + ring
                elif candidate[0] == ring[0]:
                    ring = candidate[:0:-1] + ring
                else:
                    continue
                remaining.pop(index)
                extended = True
                break

        if len(ring) >= 4:
            if ring[0] != ring[-1]:
                ring = ring + [ring[0]]  # close a ring OSM left open
            rings.append(ring)

    return rings


def relation_to_polygon(element: dict) -> dict | None:
    """Boundary relation -> GeoJSON Polygon/MultiPolygon. Ignores inner rings (enclaves are
    not worth the complexity at ward scale)."""
    outer = [
        [(round(p["lon"], 7), round(p["lat"], 7)) for p in member["geometry"]]
        for member in element.get("members", [])
        if member.get("type") == "way"
        and member.get("role") in ("outer", "")
        and member.get("geometry")
    ]
    rings = stitch_rings(outer)
    if not rings:
        return None
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [[list(p) for p in rings[0]]]}
    return {"type": "MultiPolygon", "coordinates": [[[list(p) for p in r]] for r in rings]}


def way_to_polygon(element: dict) -> dict | None:
    geometry = element.get("geometry") or []
    if len(geometry) < 4:
        return None
    ring = [[round(p["lon"], 7), round(p["lat"], 7)] for p in geometry]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def centroid(element: dict) -> tuple[float, float] | None:
    if "lat" in element and "lon" in element:
        return round(element["lon"], 7), round(element["lat"], 7)
    center = element.get("center")
    if center:
        return round(center["lon"], 7), round(center["lat"], 7)
    return None


def write_geojson(path: Path, features: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"  wrote {len(features)} features -> {path}")


# --- Layers ---------------------------------------------------------------


def fetch_wards(bbox: tuple[float, ...], municipality: str | None, out: Path) -> None:
    """Nepal ward boundaries: admin_level=9, named '<Municipality>-<NN>'."""
    print("Wards (admin_level=9)...")
    data = query(f"""[out:json][timeout:180];
relation["boundary"="administrative"]["admin_level"="9"]({",".join(map(str, bbox))});
out geom;""")

    features = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name") or ""
        if municipality and not name.lower().startswith(municipality.lower()):
            continue
        geometry = relation_to_polygon(element)
        if geometry is None:
            continue

        municipality_name, _, ward_number = name.rpartition("-")
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "ward_id": f"osm-{element['id']}",
                "ward_name": f"Ward {ward_number.lstrip('0') or ward_number}"
                            if ward_number.isdigit() else name,
                "full_name": name,
                "municipality": municipality_name or tags.get("is_in:municipality") or "",
                "osm_id": element["id"],
            },
        })

    features.sort(key=lambda f: f["properties"]["full_name"])
    write_geojson(out, features)


def fetch_facilities(bbox: tuple[float, ...], out: Path) -> None:
    """Critical infrastructure: the `infra` term of the priority score."""
    print("Critical facilities...")
    amenity_regex = "|".join(FACILITY_AMENITIES)
    box = ",".join(map(str, bbox))
    data = query(f"""[out:json][timeout:180];
(
  node["amenity"~"^({amenity_regex})$"]({box});
  way["amenity"~"^({amenity_regex})$"]({box});
  relation["amenity"~"^({amenity_regex})$"]({box});
  way["bridge"="yes"]["highway"]({box});
);
out center;""")

    features = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        point = centroid(element)
        if point is None:
            continue

        amenity = tags.get("amenity")
        facility_type = AMENITY_TO_TYPE.get(amenity) if amenity else (
            "bridge" if tags.get("bridge") == "yes" else None
        )
        if facility_type is None:
            continue

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(point)},
            "properties": {
                "facility_type": facility_type,
                "name": tags.get("name:en") or tags.get("name") or facility_type.title(),
                "osm_id": element["id"],
            },
        })

    write_geojson(out, features)


def fetch_buildings(bbox: tuple[float, ...], out: Path, limit: int | None) -> None:
    """Building footprints - stage 1 of the pipeline, no segmentation model required."""
    print("Building footprints...")
    data = query(f"""[out:json][timeout:180];
way["building"]({",".join(map(str, bbox))});
out geom;""")

    features = []
    for element in data.get("elements", []):
        geometry = way_to_polygon(element)
        if geometry is None:
            continue
        tags = element.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "id": f"osm-{element['id']}",
                "building": tags.get("building", "yes"),
                "name": tags.get("name:en") or tags.get("name"),
            },
        })
        if limit and len(features) >= limit:
            print(f"  stopped at --limit {limit}")
            break

    write_geojson(out, features)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Fetch every layer")
    parser.add_argument("--wards", action="store_true")
    parser.add_argument("--facilities", action="store_true")
    parser.add_argument("--buildings", action="store_true")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                        default=list(DEFAULT_BBOX), help="south west north east")
    parser.add_argument("--municipality", default="Kathmandu",
                        help="Keep only wards of this municipality; empty string keeps all")
    parser.add_argument("--limit", type=int, default=20000,
                        help="Cap building footprints (Kathmandu has ~500k)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "nepal")
    args = parser.parse_args()

    if not any([args.all, args.wards, args.facilities, args.buildings]):
        parser.error("pick at least one of --all/--wards/--facilities/--buildings")

    bbox = tuple(args.bbox)
    print(f"bbox {bbox}  ->  {args.out_dir}\n")

    if args.all or args.wards:
        fetch_wards(bbox, args.municipality or None, args.out_dir / "wards.geojson")
    if args.all or args.facilities:
        fetch_facilities(bbox, args.out_dir / "facilities.geojson")
    if args.all or args.buildings:
        fetch_buildings(bbox, args.out_dir / "buildings.geojson", args.limit)

    print("\nData (c) OpenStreetMap contributors, ODbL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
