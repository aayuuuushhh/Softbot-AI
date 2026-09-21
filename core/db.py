"""MongoDB access layer.

A single shared `AsyncIOMotorClient` is opened by the FastAPI lifespan and
reused everywhere. Collection names live here so nothing else hardcodes them.
"""

from __future__ import annotations

import logging
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, GEOSPHERE, IndexModel

from core.config import get_settings

log = logging.getLogger(__name__)

# Collection names
EVENTS = "events"
ZONES = "zones"
GROUND_REPORTS = "ground_reports"
DETECTIONS = "detections"
NODES = "nodes"
EDGES = "edges"
INVENTORY = "inventory"
DISPATCHES = "dispatches"
AGENT_RUNS = "agent_runs"
RESEARCH_CACHE = "research_cache"
LEDGER_LOG = "ledger_log"

_client: AsyncIOMotorClient | None = None


def connect() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.mongodb_uri, tz_aware=True)
        log.info("connected to mongodb at %s", settings.mongodb_uri)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return connect()[get_settings().mongodb_db]


def reset() -> None:
    """Drop the cached client.

    Motor binds a client to the event loop that created it, so anything that
    runs across several loops (pytest-asyncio, a reloading worker) must reset
    between them or the next call raises "Event loop is closed".
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def close() -> None:
    reset()


def oid(value: str | ObjectId) -> ObjectId:
    """Coerce to ObjectId, raising a readable error instead of bson's."""
    if isinstance(value, ObjectId):
        return value
    if not ObjectId.is_valid(value):
        raise ValueError(f"not a valid object id: {value!r}")
    return ObjectId(value)


def doc_out(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Stringify `_id` and any *_id field so pydantic models accept the doc."""
    if doc is None:
        return None
    out = dict(doc)
    for key, value in out.items():
        if isinstance(value, ObjectId):
            out[key] = str(value)
        elif isinstance(value, list) and value and isinstance(value[0], ObjectId):
            out[key] = [str(v) for v in value]
    return out


async def init_indexes() -> None:
    """Create every index the system relies on. Safe to call repeatedly."""
    db = get_db()
    ttl_seconds = get_settings().research_cache_ttl_hours * 3600

    await db[ZONES].create_indexes(
        [
            IndexModel([("geometry", GEOSPHERE)], name="zone_geometry_2dsphere"),
            IndexModel([("event_id", ASCENDING)], name="zone_event"),
        ]
    )
    await db[NODES].create_indexes(
        [
            IndexModel([("location", GEOSPHERE)], name="node_location_2dsphere"),
            IndexModel([("event_id", ASCENDING)], name="node_event"),
        ]
    )
    await db[GROUND_REPORTS].create_indexes(
        [
            IndexModel([("location", GEOSPHERE)], name="ground_location_2dsphere"),
            IndexModel([("event_id", ASCENDING)], name="ground_event"),
        ]
    )
    await db[EDGES].create_indexes(
        [
            IndexModel([("event_id", ASCENDING), ("status", ASCENDING)], name="edge_event_status"),
            IndexModel([("u", ASCENDING)], name="edge_u"),
            IndexModel([("v", ASCENDING)], name="edge_v"),
        ]
    )
    await db[INVENTORY].create_indexes(
        [
            IndexModel(
                [("event_id", ASCENDING), ("node_id", ASCENDING), ("sku", ASCENDING)],
                name="inventory_event_node_sku",
                unique=True,
            ),
        ]
    )
    await db[DISPATCHES].create_indexes(
        [
            IndexModel(
                [("event_id", ASCENDING), ("status", ASCENDING)], name="dispatch_event_status"
            ),
        ]
    )
    await db[DETECTIONS].create_indexes(
        [IndexModel([("event_id", ASCENDING)], name="detection_event")]
    )
    await db[AGENT_RUNS].create_indexes(
        [IndexModel([("event_id", ASCENDING)], name="agent_run_event")]
    )
    await db[RESEARCH_CACHE].create_indexes(
        [
            IndexModel([("query_hash", ASCENDING)], name="research_query_hash", unique=True),
            IndexModel(
                [("created_at", ASCENDING)],
                name="research_ttl",
                expireAfterSeconds=ttl_seconds,
            ),
        ]
    )
    log.info("mongodb indexes ensured")


async def ping() -> bool:
    try:
        await connect().admin.command("ping")
        return True
    except Exception as exc:  # noqa: BLE001 - health check must not raise
        log.warning("mongodb ping failed: %s", exc)
        return False
