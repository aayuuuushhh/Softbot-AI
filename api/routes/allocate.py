"""Agentic resource allocation (S7/S8) and the dispatch lifecycle."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from agents.allocator import run_allocation
from core import db, inventory
from core.needs import update_zone_needs
from core.schemas import Dispatch, DispatchStatus
from core.standards import baseline

router = APIRouter(prefix="/api", tags=["allocation"])


@router.post("/events/{event_id}/allocate")
async def allocate(event_id: str) -> dict:
    """Research historical precedents, adapt parameters, optimise, persist.

    Produces `proposed` dispatches; nothing leaves the pool until an operator
    moves an order to `reserved`.
    """
    if not await db.get_db()[db.EVENTS].find_one({"_id": db.oid(event_id)}):
        raise HTTPException(404, f"no such event: {event_id}")
    return await run_allocation(event_id)


@router.get("/events/{event_id}/dispatches", response_model=list[Dispatch])
async def list_dispatches(event_id: str, status: DispatchStatus | None = None) -> list[Dispatch]:
    """The field-team endpoint: every manifest, with its route and citations."""
    query: dict = {"event_id": event_id}
    if status:
        query["status"] = status.value
    cursor = db.get_db()[db.DISPATCHES].find(query).sort([("priority", 1), ("created_at", 1)])
    return [Dispatch.model_validate(db.doc_out(d)) async for d in cursor]


@router.get("/dispatches/{dispatch_id}", response_model=Dispatch)
async def get_dispatch(dispatch_id: str) -> Dispatch:
    doc = await db.get_db()[db.DISPATCHES].find_one({"_id": db.oid(dispatch_id)})
    if doc is None:
        raise HTTPException(404, f"no such dispatch: {dispatch_id}")
    return Dispatch.model_validate(db.doc_out(doc))


@router.patch("/dispatches/{dispatch_id}/status", response_model=Dispatch)
async def set_dispatch_status(dispatch_id: str, status: DispatchStatus) -> Dispatch:
    """proposed -> reserved -> in_transit -> delivered, or -> cancelled.

    `reserved` takes the stock out of the pool (409 if it is no longer there);
    `delivered` consumes it; `cancelled` returns it.
    """
    try:
        doc = await inventory.transition(dispatch_id, status.value)
    except inventory.LedgerError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    return Dispatch.model_validate(db.doc_out(doc))


@router.post("/events/{event_id}/dispatches/approve")
async def approve_all(event_id: str, max_priority: int = Query(10, ge=1, le=10)) -> dict:
    """Reserve every proposed order up to `max_priority`, most urgent first.

    Orders that can no longer be filled are reported, not half-reserved.
    """
    cursor = (
        db.get_db()[db.DISPATCHES]
        .find(
            {
                "event_id": event_id,
                "status": DispatchStatus.PROPOSED.value,
                "priority": {"$lte": max_priority},
            }
        )
        .sort("priority", 1)
    )
    reserved, refused = [], []
    async for d in cursor:
        try:
            await inventory.transition(str(d["_id"]), DispatchStatus.RESERVED.value)
            reserved.append(str(d["_id"]))
        except inventory.LedgerError as exc:
            refused.append({"id": str(d["_id"]), "reason": str(exc)})
    return {"reserved": reserved, "refused": refused}


@router.get("/events/{event_id}/agent-runs")
async def list_agent_runs(event_id: str, limit: int = Query(10, le=50)) -> list[dict]:
    cursor = (
        db.get_db()[db.AGENT_RUNS]
        .find(
            {"event_id": event_id},
            {"context": 0, "research.report": 0},
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    return [db.doc_out(r) async for r in cursor]


@router.get("/agent-runs/{run_id}")
async def get_agent_run(run_id: str) -> dict:
    """The full trace behind a plan: queries, report, precedents, proposal,
    accepted and rejected adjustments, parameters, invariant checks."""
    doc = await db.get_db()[db.AGENT_RUNS].find_one({"_id": db.oid(run_id)})
    if doc is None:
        raise HTTPException(404, f"no such agent run: {run_id}")
    return db.doc_out(doc)


@router.get("/events/{event_id}/needs")
async def event_needs(event_id: str, recompute: bool = False) -> list[dict]:
    """Per-zone needs with derivations. `recompute` resets them to the baseline parameters."""
    if recompute:
        return await update_zone_needs(event_id)
    out = []
    async for z in db.get_db()[db.ZONES].find({"event_id": event_id}):
        out.append(
            {
                "zone_id": str(z["_id"]),
                "name": z["name"],
                "severity": z.get("severity"),
                **(z.get("needs") or {}),
            }
        )
    return out


@router.get("/standards")
async def standards() -> dict:
    """Baseline planning parameters with their sources - what the agent starts from."""
    return {k: v.model_dump() for k, v in baseline().items()}
