"""Field-ready exports: printable PDF and radio-length text summary."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from core import db
from core.report import gather, render_pdf, text_summary

router = APIRouter(prefix="/api", tags=["report"])


async def _data(event_id: str) -> dict:
    if not await db.get_db()[db.EVENTS].find_one({"_id": db.oid(event_id)}):
        raise HTTPException(404, f"no such event: {event_id}")
    return await gather(event_id)


@router.get("/events/{event_id}/report.pdf")
async def event_report(event_id: str) -> Response:
    """Offline situation report: damage, ranked zones, manifests, citations, parameters."""
    pdf = render_pdf(await _data(event_id))
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="uddhar-{event_id}.pdf"'},
    )


@router.get("/events/{event_id}/summary.txt", response_class=PlainTextResponse)
async def event_summary(event_id: str) -> str:
    """Top zones in under a page - for SMS, radio, or a printout carried by hand."""
    return text_summary(await _data(event_id))
