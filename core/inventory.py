"""S6: the live relief ledger - reservation, transit and delivery of stock.

Invariant (CLAUDE.md section 7): nothing is ever allocated beyond what exists.
Enforced at the storage layer, not by callers being careful:

  * every reservation is a single conditional update -
    `quantity - reserved >= n` is part of the filter, so two concurrent
    reservations cannot both succeed against the same last units;
  * a multi-item reservation that fails part-way rolls back what it took;
  * dispatch status changes are compare-and-set on the current status, so a
    double-click cannot reserve twice or deliver twice.

Standalone MongoDB has no multi-document transactions; the compensation above
is what stands in for one.

State machine for a dispatch:

    proposed --reserve--> reserved --depart--> in_transit --arrive--> delivered
        |                    |                     |
        +------cancel--------+-------cancel--------+   (cancel releases stock)
"""

from __future__ import annotations

import logging

from core import db
from core.schemas import DispatchStatus, utcnow

log = logging.getLogger(__name__)

TRANSITIONS: dict[str, set[str]] = {
    DispatchStatus.PROPOSED.value: {DispatchStatus.RESERVED.value, DispatchStatus.CANCELLED.value},
    DispatchStatus.RESERVED.value: {
        DispatchStatus.IN_TRANSIT.value,
        DispatchStatus.CANCELLED.value,
    },
    DispatchStatus.IN_TRANSIT.value: {
        DispatchStatus.DELIVERED.value,
        DispatchStatus.CANCELLED.value,
    },
    DispatchStatus.DELIVERED.value: set(),
    DispatchStatus.CANCELLED.value: set(),
}

# Statuses in which a dispatch's items are held in `reserved`.
HOLDING = {DispatchStatus.RESERVED.value, DispatchStatus.IN_TRANSIT.value}


class LedgerError(Exception):
    """Raised for any refused ledger operation. `status_code` maps it onto HTTP."""

    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


async def _log(event_id: str, action: str, **fields) -> None:
    await db.get_db()[db.LEDGER_LOG].insert_one(
        {"event_id": event_id, "action": action, "at": utcnow(), **fields}
    )


def _line_filter(event_id: str, node_id: str, sku: str) -> dict:
    return {"event_id": event_id, "node_id": node_id, "sku": sku}


async def reserve(event_id: str, node_id: str, items: list[dict], ref: str | None = None) -> None:
    """Reserve every item at one node, all or nothing."""
    inv = db.get_db()[db.INVENTORY]
    taken: list[dict] = []
    for item in items:
        qty = int(item["quantity"])
        doc = await inv.find_one_and_update(
            {
                **_line_filter(event_id, node_id, item["sku"]),
                "$expr": {"$gte": [{"$subtract": ["$quantity", "$reserved"]}, qty]},
            },
            {"$inc": {"reserved": qty}},
        )
        if doc is None:
            for t in taken:  # compensate
                await inv.update_one(
                    _line_filter(event_id, node_id, t["sku"]),
                    {"$inc": {"reserved": -int(t["quantity"])}},
                )
            line = await inv.find_one(_line_filter(event_id, node_id, item["sku"]))
            have = (line["quantity"] - line["reserved"]) if line else 0
            raise LedgerError(
                f"insufficient stock: {item['sku']} at node {node_id} - requested {qty}, "
                f"available {have}; nothing was reserved"
            )
        taken.append(item)
    await _log(event_id, "reserve", node_id=node_id, items=items, ref=ref)


async def release(event_id: str, node_id: str, items: list[dict], ref: str | None = None) -> None:
    """Return reserved units to the available pool."""
    inv = db.get_db()[db.INVENTORY]
    for item in items:
        qty = int(item["quantity"])
        res = await inv.update_one(
            {**_line_filter(event_id, node_id, item["sku"]), "reserved": {"$gte": qty}},
            {"$inc": {"reserved": -qty}},
        )
        if res.modified_count != 1:
            log.error(
                "release of %s x%d at %s found less reserved than expected",
                item["sku"],
                qty,
                node_id,
            )
    await _log(event_id, "release", node_id=node_id, items=items, ref=ref)


async def consume(event_id: str, node_id: str, items: list[dict], ref: str | None = None) -> None:
    """Delivered: units leave the pool for good (quantity and reserved both drop)."""
    inv = db.get_db()[db.INVENTORY]
    for item in items:
        qty = int(item["quantity"])
        res = await inv.update_one(
            {
                **_line_filter(event_id, node_id, item["sku"]),
                "reserved": {"$gte": qty},
                "quantity": {"$gte": qty},
            },
            {"$inc": {"reserved": -qty, "quantity": -qty}},
        )
        if res.modified_count != 1:
            raise LedgerError(f"cannot consume {item['sku']} x{qty} at {node_id}: not reserved")
    await _log(event_id, "consume", node_id=node_id, items=items, ref=ref)


async def transition(dispatch_id: str, new_status: str) -> dict:
    """Move a dispatch through the state machine, applying its stock effect.

    Compare-and-set on the old status first, then the stock effect; if the stock
    effect is refused the status is put back, so the two never disagree.
    """
    database = db.get_db()
    dispatch = await database[db.DISPATCHES].find_one({"_id": db.oid(dispatch_id)})
    if dispatch is None:
        raise LedgerError(f"no such dispatch: {dispatch_id}", 404)
    old = dispatch["status"]
    if new_status not in TRANSITIONS.get(old, set()):
        raise LedgerError(f"illegal transition {old} -> {new_status}", 409)

    swapped = await database[db.DISPATCHES].find_one_and_update(
        {"_id": dispatch["_id"], "status": old},
        {"$set": {"status": new_status, "updated_at": utcnow()}},
        return_document=True,
    )
    if swapped is None:
        raise LedgerError("dispatch changed concurrently; reload and retry", 409)

    event_id, node_id, items = dispatch["event_id"], dispatch["from_node_id"], dispatch["items"]
    try:
        if new_status == DispatchStatus.RESERVED.value:
            await reserve(event_id, node_id, items, ref=dispatch_id)
        elif new_status == DispatchStatus.DELIVERED.value:
            await consume(event_id, node_id, items, ref=dispatch_id)
        elif new_status == DispatchStatus.CANCELLED.value and old in HOLDING:
            await release(event_id, node_id, items, ref=dispatch_id)
    except LedgerError:
        await database[db.DISPATCHES].update_one(
            {"_id": dispatch["_id"], "status": new_status},
            {"$set": {"status": old, "updated_at": utcnow()}},
        )
        raise
    await _log(event_id, "transition", ref=dispatch_id, old=old, new=new_status)
    return swapped


async def available(event_id: str) -> dict[tuple[str, str], int]:
    """(node_id, sku) -> units still free to allocate."""
    out: dict[tuple[str, str], int] = {}
    async for line in db.get_db()[db.INVENTORY].find({"event_id": event_id}):
        out[(str(line["node_id"]), line["sku"])] = int(line["quantity"]) - int(
            line.get("reserved", 0)
        )
    return out


async def sku_kinds(event_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    async for line in db.get_db()[db.INVENTORY].find({"event_id": event_id}):
        out[line["sku"]] = line["kind"]
    return out
