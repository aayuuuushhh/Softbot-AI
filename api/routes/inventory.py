"""Resource and inventory endpoints (F2).

Stock movements driven by dispatches go through core/inventory.py (atomic
reserve / consume / release). These endpoints cover stock intake, manual
correction, and the audit log.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from core import db
from core.schemas import (
    GraphNode,
    InventoryItem,
    InventoryItemCreate,
    ResourceKind,
)

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


@router.post("", response_model=InventoryItem, status_code=201)
async def add_stock(payload: InventoryItemCreate) -> InventoryItem:
    """Insert or top up a stock line. Unique on (event_id, node_id, sku)."""
    doc = await db.get_db()[db.INVENTORY].find_one_and_update(
        {
            "event_id": payload.event_id,
            "node_id": payload.node_id,
            "sku": payload.sku,
        },
        {
            "$inc": {"quantity": payload.quantity},
            "$setOnInsert": {
                "kind": payload.kind,
                "unit": payload.unit,
                "reserved": 0,
            },
        },
        upsert=True,
        return_document=True,
    )
    return InventoryItem.model_validate(db.doc_out(doc))


@router.get("", response_model=list[InventoryItem])
async def list_stock(
    event_id: str,
    node_id: str | None = None,
    kind: ResourceKind | None = None,
) -> list[InventoryItem]:
    query: dict = {"event_id": event_id}
    if node_id:
        query["node_id"] = node_id
    if kind:
        query["kind"] = kind.value
    cursor = db.get_db()[db.INVENTORY].find(query)
    return [InventoryItem.model_validate(db.doc_out(d)) async for d in cursor]


@router.get("/summary", tags=["inventory"])
async def stock_summary(event_id: str) -> dict:
    """Global pool per resource kind: total, reserved, and still-available."""
    pipeline = [
        {"$match": {"event_id": event_id}},
        {
            "$group": {
                "_id": "$kind",
                "quantity": {"$sum": "$quantity"},
                "reserved": {"$sum": "$reserved"},
                "skus": {"$addToSet": "$sku"},
            }
        },
    ]
    out: dict[str, dict] = {}
    async for row in db.get_db()[db.INVENTORY].aggregate(pipeline):
        out[row["_id"]] = {
            "quantity": row["quantity"],
            "reserved": row["reserved"],
            "available": row["quantity"] - row["reserved"],
            "skus": sorted(row["skus"]),
        }
    return out


@router.patch("/{item_id}", response_model=InventoryItem)
async def adjust_stock(item_id: str, delta: int = Query(..., description="May be negative")):
    """Manual correction. Refuses to push quantity below what is reserved."""
    doc = await db.get_db()[db.INVENTORY].find_one_and_update(
        {"_id": db.oid(item_id), "$expr": {"$gte": [{"$add": ["$quantity", delta]}, "$reserved"]}},
        {"$inc": {"quantity": delta}},
        return_document=True,
    )
    if doc is None:
        raise HTTPException(
            409,
            "adjustment refused: item not found, or it would push quantity "
            "below the amount already reserved",
        )
    return InventoryItem.model_validate(db.doc_out(doc))


@router.get("/nodes", response_model=list[GraphNode])
async def list_nodes(event_id: str) -> list[GraphNode]:
    """Holding centres and staging nodes that hold stock."""
    cursor = db.get_db()[db.NODES].find({"event_id": event_id})
    return [GraphNode.model_validate(db.doc_out(d)) async for d in cursor]


@router.get("/ledger", tags=["inventory"])
async def ledger(event_id: str, limit: int = Query(200, le=1000)) -> list[dict]:
    """Every stock movement, newest first: reserve, release, consume, transition."""
    cursor = db.get_db()[db.LEDGER_LOG].find({"event_id": event_id}).sort("at", -1).limit(limit)
    return [db.doc_out(d) async for d in cursor]
