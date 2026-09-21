"""Field-ready PDF export.

Scaffolded in milestone 1; reportlab rendering lands in milestone 4.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["report"])


@router.get("/events/{event_id}/report.pdf")
async def event_report(event_id: str):
    """Offline situation report: damage summary, ranked zones, manifests, citations."""
    raise HTTPException(501, "PDF reporting lands in milestone 4")
