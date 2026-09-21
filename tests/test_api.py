"""API-level smoke tests. Anything touching the database is skipped when
MongoDB is not running."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from tests.conftest import requires_mongo


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_health_reports_device_and_backends(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] in ("cuda", "cpu")
    assert set(body) >= {"mongodb", "cuda", "sat_backend", "ground_backend"}


async def test_openapi_exposes_every_router(client):
    paths = (await client.get("/openapi.json")).json()["paths"]
    for expected in (
        "/api/events",
        "/api/upload/ground",
        "/api/inventory",
        "/api/events/{event_id}/graph",
        "/api/events/{event_id}/allocate",
        "/api/events/{event_id}/report.pdf",
    ):
        assert expected in paths, f"{expected} missing from the OpenAPI schema"


@requires_mongo
async def test_report_for_a_missing_event_is_404_not_500(client):
    r = await client.get("/api/events/000000000000000000000000/report.pdf")
    assert r.status_code == 404


@requires_mongo
async def test_event_lifecycle(client):
    created = await client.post(
        "/api/events",
        json={"name": "pytest event", "disaster_type": "flood", "region": "test"},
    )
    assert created.status_code == 201
    event_id = created.json()["id"]
    assert created.json()["stages"]["overhead"] == "pending"

    fetched = await client.get(f"/api/events/{event_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "pytest event"

    assert (await client.delete(f"/api/events/{event_id}")).status_code == 204
    assert (await client.get(f"/api/events/{event_id}")).status_code == 404


@requires_mongo
async def test_inventory_cannot_be_adjusted_below_what_is_reserved(client):
    """F2's core invariant, checked at the API boundary."""
    event = (await client.post(
        "/api/events", json={"name": "pytest stock", "disaster_type": "flood"}
    )).json()
    event_id = event["id"]
    try:
        item = (await client.post("/api/inventory", json={
            "event_id": event_id, "node_id": "n1", "kind": "food",
            "sku": "rice_25kg", "quantity": 100, "unit": "sack",
        })).json()

        # Reserve 80 directly, then try to remove 50 of the remaining 100.
        from core import db
        await db.get_db()[db.INVENTORY].update_one(
            {"_id": db.oid(item["id"])}, {"$set": {"reserved": 80}}
        )

        refused = await client.patch(f"/api/inventory/{item['id']}?delta=-50")
        assert refused.status_code == 409

        allowed = await client.patch(f"/api/inventory/{item['id']}?delta=-20")
        assert allowed.status_code == 200
        assert allowed.json()["quantity"] == 80
    finally:
        await client.delete(f"/api/events/{event_id}")


@requires_mongo
async def test_bad_object_id_is_a_client_error_not_a_crash(client):
    r = await client.get("/api/events/not-an-object-id")
    assert r.status_code < 500
