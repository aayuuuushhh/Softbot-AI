"""Agentic resource allocation (S7/S8).

Scaffolded in milestone 1; the LangChain agent lands in milestone 4.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core import db
from core.schemas import Dispatch

router = APIRouter(prefix="/api", tags=["allocation"])

NOT_YET = "the allocation agent lands in milestone 4"


@router.post("/events/{event_id}/allocate")
async def allocate(event_id: str) -> dict:
    """Run historical research + graph optimisation, produce dispatch orders."""
    raise HTTPException(501, NOT_YET)


@router.get("/events/{event_id}/dispatches", response_model=list[Dispatch])
async def list_dispatches(event_id: str) -> list[Dispatch]:
    """The field-team endpoint: every manifest, with its citations."""
    cursor = db.get_db()[db.DISPATCHES].find({"event_id": event_id}).sort("priority", 1)
    return [Dispatch.model_validate(db.doc_out(d)) async for d in cursor]


@router.patch("/dispatches/{dispatch_id}/status", response_model=Dispatch)
async def set_dispatch_status(dispatch_id: str, status: str) -> Dispatch:
    raise HTTPException(501, NOT_YET)
