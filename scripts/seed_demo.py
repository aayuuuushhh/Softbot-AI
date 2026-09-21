#!/usr/bin/env python3
"""Seed a demo scenario: a monsoon flood in Rasuwa district, Nepal.

Creates one event, 8 ward zones, 3 supply nodes, a road graph with two blocked
edges, and a starting inventory. Everything downstream (graph routing, the
allocation agent, the dashboard) needs this to exist.

    python scripts/seed_demo.py [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db  # noqa: E402
from core.schemas import (  # noqa: E402
    EdgeStatus,
    NodeKind,
    PipelineStages,
    ResourceKind,
    utcnow,
)

EVENT = {
    "name": "Rasuwa Monsoon Flood 2026",
    "disaster_type": "flood",
    "region": "Rasuwa District, Bagmati Province, Nepal",
    "bbox": [85.20, 27.95, 85.55, 28.25],
    "air_transport_verified": False,
}

# (name, lon, lat, population)
ZONES = [
    ("Dhunche",        85.297, 28.112, 3200),
    ("Syaphru Besi",   85.339, 28.161, 2100),
    ("Ramche",         85.256, 28.036,  1800),
    ("Laharepauwa",    85.283, 27.988,  2600),
    ("Kalikasthan",    85.245, 27.973,  3050),
    ("Haku",           85.312, 28.070,  1450),
    ("Gatlang",        85.238, 28.187,  1250),
    ("Thuman",         85.375, 28.205,   900),
]

# (name, kind, lon, lat, capacity)
SUPPLY_NODES = [
    ("Dhunche District HQ Depot", NodeKind.HOLDING_CENTER, 85.300, 28.110, 5000),
    ("Kalikasthan Staging",       NodeKind.STAGING_NODE,   85.247, 27.971, 2000),
    ("Betrawati Forward Base",    NodeKind.STAGING_NODE,   85.192, 27.930, 3000),
]

# (from, to, km, status) - names refer to zones or supply nodes
ROADS = [
    ("Betrawati Forward Base", "Kalikasthan Staging", 8.4, EdgeStatus.OPEN),
    ("Kalikasthan Staging", "Kalikasthan", 0.6, EdgeStatus.OPEN),
    ("Kalikasthan Staging", "Laharepauwa", 3.2, EdgeStatus.OPEN),
    ("Kalikasthan Staging", "Ramche", 7.1, EdgeStatus.DEGRADED),
    ("Kalikasthan Staging", "Dhunche District HQ Depot", 22.5, EdgeStatus.OPEN),
    ("Dhunche District HQ Depot", "Dhunche", 0.4, EdgeStatus.OPEN),
    ("Dhunche District HQ Depot", "Haku", 9.8, EdgeStatus.BLOCKED),      # landslide
    ("Dhunche District HQ Depot", "Syaphru Besi", 16.2, EdgeStatus.OPEN),
    ("Dhunche District HQ Depot", "Gatlang", 18.7, EdgeStatus.DEGRADED),
    ("Syaphru Besi", "Thuman", 11.4, EdgeStatus.BLOCKED),                # bridge washed out
    ("Ramche", "Laharepauwa", 5.5, EdgeStatus.OPEN),
]

# (node name, kind, sku, quantity, unit)
STOCK = [
    ("Dhunche District HQ Depot", ResourceKind.PERSONNEL, "doctor", 12, "person"),
    ("Dhunche District HQ Depot", ResourceKind.PERSONNEL, "rescue_personnel", 40, "person"),
    ("Dhunche District HQ Depot", ResourceKind.PERSONNEL, "engineer", 8, "person"),
    ("Dhunche District HQ Depot", ResourceKind.FOOD, "rice_25kg", 600, "sack"),
    ("Dhunche District HQ Depot", ResourceKind.WATER, "water_purifier", 45, "unit"),
    ("Dhunche District HQ Depot", ResourceKind.MEDICAL, "trauma_kit", 150, "kit"),
    ("Kalikasthan Staging", ResourceKind.PERSONNEL, "rescue_personnel", 18, "person"),
    ("Kalikasthan Staging", ResourceKind.FOOD, "rice_25kg", 240, "sack"),
    ("Kalikasthan Staging", ResourceKind.WATER, "water_purifier", 20, "unit"),
    ("Kalikasthan Staging", ResourceKind.MEDICAL, "trauma_kit", 60, "kit"),
    ("Betrawati Forward Base", ResourceKind.PERSONNEL, "doctor", 6, "person"),
    ("Betrawati Forward Base", ResourceKind.PERSONNEL, "rescue_personnel", 55, "person"),
    ("Betrawati Forward Base", ResourceKind.FOOD, "rice_25kg", 900, "sack"),
    ("Betrawati Forward Base", ResourceKind.WATER, "water_purifier", 70, "unit"),
    ("Betrawati Forward Base", ResourceKind.MEDICAL, "trauma_kit", 210, "kit"),
]

BOX = 0.012  # ~1.3 km half-width for the synthetic ward polygons


def square(lon: float, lat: float, half: float = BOX) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - half, lat - half],
                [lon + half, lat - half],
                [lon + half, lat + half],
                [lon - half, lat + half],
                [lon - half, lat - half],
            ]
        ],
    }


async def seed(reset: bool, event_name: str | None = None, verbose: bool = True) -> str:
    """Insert the demo scenario. `event_name` overrides the name (tests use this)."""
    database = db.get_db()
    await db.init_indexes()
    event = {**EVENT, "name": event_name or EVENT["name"]}
    say = print if verbose else (lambda *a, **k: None)

    if reset:
        existing = await database[db.EVENTS].find_one({"name": event["name"]})
        if existing:
            eid = str(existing["_id"])
            for coll in (db.ZONES, db.NODES, db.EDGES, db.INVENTORY, db.DISPATCHES,
                         db.GROUND_REPORTS, db.AGENT_RUNS, db.DETECTIONS, db.LEDGER_LOG):
                await database[coll].delete_many({"event_id": eid})
            await database[db.EVENTS].delete_one({"_id": existing["_id"]})
            say(f"removed previous demo event {eid}")

    event_doc = dict(event)
    event_doc["stages"] = PipelineStages().model_dump()
    event_doc["cloud_fraction"] = 0.0
    event_doc["crs"] = "EPSG:4326"
    event_doc["created_at"] = utcnow()
    event_id = str((await database[db.EVENTS].insert_one(event_doc)).inserted_id)
    say(f"event {event_id}  {event['name']}")

    # Zones become graph nodes too, so routing has somewhere to deliver.
    name_to_node: dict[str, str] = {}

    for name, lon, lat, pop in ZONES:
        zone_doc = {
            "event_id": event_id,
            "name": name,
            "geometry": square(lon, lat),
            "centroid": {"type": "Point", "coordinates": [lon, lat]},
            "population": pop,
            "severity": "none",
            "damage_score": 0.0,
            "buildings_destroyed": 0,
            "detections": 0,
            "decided_by": "none",
            "needs": {"personnel": 0, "food_rations": 0,
                      "water_litres": 0, "medical_kits": 0},
        }
        zone_id = str((await database[db.ZONES].insert_one(zone_doc)).inserted_id)
        node_doc = {
            "event_id": event_id,
            "name": name,
            "kind": NodeKind.ZONE.value,
            "location": {"type": "Point", "coordinates": [lon, lat]},
            "zone_id": zone_id,
            "capacity": 0,
        }
        name_to_node[name] = str((await database[db.NODES].insert_one(node_doc)).inserted_id)
    say(f"  {len(ZONES)} zones (each also a graph node)")

    for name, kind, lon, lat, cap in SUPPLY_NODES:
        node_doc = {
            "event_id": event_id,
            "name": name,
            "kind": kind.value,
            "location": {"type": "Point", "coordinates": [lon, lat]},
            "zone_id": None,
            "capacity": cap,
        }
        name_to_node[name] = str((await database[db.NODES].insert_one(node_doc)).inserted_id)
    say(f"  {len(SUPPLY_NODES)} supply nodes")

    edge_docs = []
    for u, v, km, status in ROADS:
        edge_docs.append({
            "event_id": event_id,
            "u": name_to_node[u],
            "v": name_to_node[v],
            "distance_km": km,
            "status": status.value,
            "source": "seed",
        })
    await database[db.EDGES].insert_many(edge_docs)
    blocked = sum(1 for r in ROADS if r[3] is EdgeStatus.BLOCKED)
    say(f"  {len(ROADS)} roads ({blocked} blocked, "
          f"{sum(1 for r in ROADS if r[3] is EdgeStatus.DEGRADED)} degraded)")

    stock_docs = [
        {
            "event_id": event_id,
            "node_id": name_to_node[node],
            "kind": kind.value,
            "sku": sku,
            "quantity": qty,
            "reserved": 0,
            "unit": unit,
        }
        for node, kind, sku, qty, unit in STOCK
    ]
    await database[db.INVENTORY].insert_many(stock_docs)
    say(f"  {len(STOCK)} inventory lines")

    say(f"\nseeded. event_id={event_id}")
    say(f"  curl localhost:8765/api/events/{event_id}")
    say(f"  curl 'localhost:8765/api/inventory/summary?event_id={event_id}'")
    return event_id


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true",
                        help="delete an existing demo event with the same name first")
    args = parser.parse_args()
    try:
        await seed(args.reset)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
