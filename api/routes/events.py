"""Event lifecycle: create, list, inspect. Analysis lands in milestone 2."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from core import db
from core.schemas import Event, EventCreate, PipelineStages, Zone, utcnow

router = APIRouter(prefix="/api/events", tags=["events"])


@router.post("", response_model=Event, status_code=201)
async def create_event(payload: EventCreate) -> Event:
    doc = payload.model_dump()
    doc["stages"] = PipelineStages().model_dump()
    doc["cloud_fraction"] = 0.0
    doc["crs"] = None
    doc["created_at"] = utcnow()
    result = await db.get_db()[db.EVENTS].insert_one(doc)
    doc["_id"] = result.inserted_id
    return Event.model_validate(db.doc_out(doc))


@router.get("", response_model=list[Event])
async def list_events(limit: int = Query(50, le=200)) -> list[Event]:
    cursor = db.get_db()[db.EVENTS].find().sort("created_at", -1).limit(limit)
    return [Event.model_validate(db.doc_out(d)) async for d in cursor]


@router.get("/{event_id}", response_model=Event)
async def get_event(event_id: str) -> Event:
    doc = await db.get_db()[db.EVENTS].find_one({"_id": db.oid(event_id)})
    if doc is None:
        raise HTTPException(404, f"no such event: {event_id}")
    return Event.model_validate(db.doc_out(doc))


@router.get("/{event_id}/zones", response_model=list[Zone])
async def list_zones(event_id: str) -> list[Zone]:
    cursor = db.get_db()[db.ZONES].find({"event_id": event_id})
    return [Zone.model_validate(db.doc_out(d)) async for d in cursor]


@router.delete("/{event_id}", status_code=204)
async def delete_event(event_id: str) -> None:
    """Remove an event and everything hanging off it."""
    database = db.get_db()
    _id = db.oid(event_id)
    if not await database[db.EVENTS].find_one({"_id": _id}):
        raise HTTPException(404, f"no such event: {event_id}")
    for coll in (db.ZONES, db.NODES, db.EDGES, db.INVENTORY,
                 db.DISPATCHES, db.GROUND_REPORTS, db.AGENT_RUNS):
        await database[coll].delete_many({"event_id": event_id})
    await database[db.EVENTS].delete_one({"_id": _id})
