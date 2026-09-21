from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from core.config import get_settings


def _mongo_up() -> bool:
    parsed = urlparse(get_settings().mongodb_uri)
    try:
        with socket.create_connection(
            (parsed.hostname or "localhost", parsed.port or 27017), timeout=1
        ):
            return True
    except OSError:
        return False


MONGO_UP = _mongo_up()

requires_mongo = pytest.mark.skipif(
    not MONGO_UP, reason="mongodb not reachable; start it with scripts/run_mongo.sh"
)


@pytest.fixture(autouse=True)
def _fresh_mongo_client():
    """Each test gets its own event loop; the motor client must follow it."""
    from core import db

    db.reset()
    yield
    db.reset()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture(scope="session", autouse=True)
def _purge_test_events():
    """Remove anything a test created, even if that test failed part-way.

    Uses pymongo rather than motor so it stays independent of any event loop.
    """
    yield
    if not MONGO_UP:
        return
    from pymongo import MongoClient

    settings = get_settings()
    with MongoClient(settings.mongodb_uri) as client:
        db = client[settings.mongodb_db]
        ids = [str(e["_id"]) for e in db.events.find({"name": {"$regex": "^pytest"}})]
        if not ids:
            return
        for coll in ("zones", "nodes", "edges", "inventory", "dispatches",
                     "ground_reports", "agent_runs", "detections", "ledger_log"):
            db[coll].delete_many({"event_id": {"$in": ids}})
        db.events.delete_many({"name": {"$regex": "^pytest"}})
