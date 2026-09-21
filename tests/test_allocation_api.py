"""End-to-end allocation, ledger and reporting against a real MongoDB.

Seeds the Rasuwa demo scenario under a `pytest` event name, sets damage as
fusion would, and drives the API: allocate -> approve -> transit -> deliver,
plus the failure paths. Web research is stubbed to "unavailable" so the test is
deterministic and never spends API credit.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from agents import researcher
from api.main import app
from core import db, inventory
from tests.conftest import requires_mongo

pytestmark = requires_mongo

DAMAGE = {  # zone -> (severity, damage_score)
    "Dhunche": ("destroyed", 0.66),
    "Laharepauwa": ("destroyed", 0.64),
    "Haku": ("destroyed", 0.75),  # its only road is blocked in the seed
    "Syaphru Besi": ("major", 0.47),
    "Thuman": ("major", 0.54),
    "Gatlang": ("minor", 0.24),
}


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def event_id(monkeypatch):
    from scripts.seed_demo import seed

    async def no_research(context, **_):
        return researcher.ResearchResult(status="unavailable", error="stubbed in tests")

    monkeypatch.setattr(researcher, "research", no_research)
    eid = await seed(reset=True, event_name="pytest allocation", verbose=False)
    zones = db.get_db()[db.ZONES]
    for name, (sev, score) in DAMAGE.items():
        await zones.update_one(
            {"event_id": eid, "name": name},
            {"$set": {"severity": sev, "damage_score": score, "damage_extent": score}},
        )
    return eid


async def _stock(eid):
    return {
        (str(line["node_id"]), line["sku"]): line
        async for line in db.get_db()[db.INVENTORY].find({"event_id": eid})
    }


async def test_allocation_produces_cited_orders_within_stock(client, event_id):
    r = await client.post(f"/api/events/{event_id}/allocate")
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan["mode"] == "baseline"
    assert any("web research unavailable" in n for n in plan["notes"])
    assert plan["dispatches"]

    orders = (await client.get(f"/api/events/{event_id}/dispatches")).json()
    assert orders and all(o["citations"] for o in orders)
    assert all(o["status"] == "proposed" for o in orders)

    stock = await _stock(event_id)
    used = {}
    for o in orders:
        for it in o["items"]:
            key = (o["from_node_id"], it["sku"])
            used[key] = used.get(key, 0) + it["quantity"]
    for key, q in used.items():
        assert q <= stock[key]["quantity"], f"over-allocated {key}"

    run = (await client.get(f"/api/agent-runs/{plan['run_id']}")).json()
    assert run["research"]["status"] == "unavailable"
    assert run["parameters"]["water_l_per_person_day"]["value"] == 15.0


async def test_zone_behind_a_blocked_road_is_unmet_until_air_is_verified(client, event_id):
    plan = (await client.post(f"/api/events/{event_id}/allocate")).json()
    haku = [u for u in plan["unmet"] if u["name"] == "Haku"]
    assert haku and all("air" in u["reason"] for u in haku)
    reach = {
        r["name"]: r for r in (await client.get(f"/api/events/{event_id}/reachability")).json()
    }
    assert reach["Haku"]["status"] == "unreachable"

    await client.patch(f"/api/events/{event_id}/air-transport?verified=true")
    plan = (await client.post(f"/api/events/{event_id}/allocate")).json()
    zone_ids = {z["name"]: z["zone_id"] for z in plan["zone_rank"]}
    to_haku = [d for d in plan["dispatches"] if d["zone_id"] == zone_ids["Haku"]]
    assert to_haku and all(d["transport_mode"] == "air" for d in to_haku)
    assert not [u for u in plan["unmet"] if u["name"] == "Haku" and "air" in u["reason"]]


async def test_dispatch_lifecycle_moves_stock_exactly_once(client, event_id):
    await client.post(f"/api/events/{event_id}/allocate")
    order = (await client.get(f"/api/events/{event_id}/dispatches")).json()[0]
    item = order["items"][0]
    key = (order["from_node_id"], item["sku"])
    before = (await _stock(event_id))[key]

    r = await client.patch(f"/api/dispatches/{order['id']}/status?status=reserved")
    assert r.status_code == 200
    assert (await _stock(event_id))[key]["reserved"] == before["reserved"] + item["quantity"]

    again = await client.patch(f"/api/dispatches/{order['id']}/status?status=reserved")
    assert again.status_code == 409, "double reserve must be refused"

    await client.patch(f"/api/dispatches/{order['id']}/status?status=in_transit")
    r = await client.patch(f"/api/dispatches/{order['id']}/status?status=delivered")
    assert r.status_code == 200
    after = (await _stock(event_id))[key]
    assert after["quantity"] == before["quantity"] - item["quantity"]
    assert after["reserved"] == before["reserved"]

    cancel = await client.patch(f"/api/dispatches/{order['id']}/status?status=cancelled")
    assert cancel.status_code == 409, "a delivered order cannot be cancelled"

    ledger = (await client.get(f"/api/inventory/ledger?event_id={event_id}")).json()
    assert {e["action"] for e in ledger} >= {"reserve", "consume", "transition"}


async def test_cancelling_a_reserved_order_returns_its_stock(client, event_id):
    await client.post(f"/api/events/{event_id}/allocate")
    order = (await client.get(f"/api/events/{event_id}/dispatches")).json()[0]
    key = (order["from_node_id"], order["items"][0]["sku"])
    before = (await _stock(event_id))[key]["reserved"]
    await client.patch(f"/api/dispatches/{order['id']}/status?status=reserved")
    await client.patch(f"/api/dispatches/{order['id']}/status?status=cancelled")
    assert (await _stock(event_id))[key]["reserved"] == before


async def test_concurrent_reservations_cannot_oversell_the_last_units(event_id):
    stock = await _stock(event_id)
    (node, sku), line = next(iter(stock.items()))
    free = line["quantity"] - line["reserved"]
    attempts = [
        inventory.reserve(event_id, node, [{"sku": sku, "quantity": free}]) for _ in range(5)
    ]
    results = await asyncio.gather(*attempts, return_exceptions=True)
    assert sum(1 for r in results if r is None) == 1
    assert all(isinstance(r, inventory.LedgerError) for r in results if r is not None)
    after = (await _stock(event_id))[(node, sku)]
    assert after["reserved"] <= after["quantity"]


async def test_failed_multi_item_reservation_takes_nothing(event_id):
    stock = await _stock(event_id)
    node = next(n for (n, _s) in stock)
    skus = [s for (n, s) in stock if n == node][:2]
    before = {s: stock[(node, s)]["reserved"] for s in skus}
    items = [{"sku": skus[0], "quantity": 1}, {"sku": skus[1], "quantity": 10**6}]
    with pytest.raises(inventory.LedgerError):
        await inventory.reserve(event_id, node, items)
    after = await _stock(event_id)
    assert {s: after[(node, s)]["reserved"] for s in skus} == before


async def test_approve_all_reserves_most_urgent_first(client, event_id):
    await client.post(f"/api/events/{event_id}/allocate")
    r = (await client.post(f"/api/events/{event_id}/dispatches/approve")).json()
    assert r["reserved"]
    for line in (await _stock(event_id)).values():
        assert line["reserved"] <= line["quantity"]


async def test_needs_carry_derivations(client, event_id):
    needs = (await client.get(f"/api/events/{event_id}/needs?recompute=true")).json()
    dhunche = next(n for n in needs if n["name"] == "Dhunche")
    assert dhunche["affected_people"] > 0
    assert "damaged share" in dhunche["derivations"]["affected_people"]["formula"]


async def test_report_pdf_and_radio_summary(client, event_id):
    await client.post(f"/api/events/{event_id}/allocate")
    pdf = await client.get(f"/api/events/{event_id}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    txt = (await client.get(f"/api/events/{event_id}/summary.txt")).text
    assert "HAKU" in txt and "AIR TRANSPORT" in txt
    assert len(txt.splitlines()) < 40


async def test_graph_summary_reflects_blocked_roads(client, event_id):
    s = (await client.get(f"/api/events/{event_id}/graph/summary")).json()
    assert s["blocked_roads"] == 2
    assert s["passable_directed_edges"] == 2 * (s["roads"] - s["blocked_roads"])
