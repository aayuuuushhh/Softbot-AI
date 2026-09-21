"""Field-ready outputs: a printable PDF situation report and a radio-length text summary.

Both are built from what is in MongoDB at the time of the request - damage,
needs with derivations, reachability, dispatch manifests with citations, the
latest agent run's parameters - so a printed copy carries the same evidence
trail as the dashboard.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

from core import db
from core.config import get_settings
from core.spatial.graph_network import load_graph, reachability

LIMITATIONS = [
    "Damage is estimated from imagery. It is a first-pass picture for the hours before field "
    "assessment, not a replacement for it.",
    "Optical satellite imagery cannot see through cloud; zones under cloud rely on ground uploads.",
    "Uddhar ranks and proposes. Humans approve every dispatch.",
    "Quantities marked 'assumption' are planning placeholders, not measurements.",
]

SEVERITY_ORDER = {"destroyed": 0, "major": 1, "minor": 2, "none": 3}


async def gather(event_id: str) -> dict:
    database = db.get_db()
    event = await database[db.EVENTS].find_one({"_id": db.oid(event_id)})
    if event is None:
        raise ValueError(f"no such event: {event_id}")
    zones = [z async for z in database[db.ZONES].find({"event_id": event_id})]
    nodes = {str(n["_id"]): n async for n in database[db.NODES].find({"event_id": event_id})}
    dispatches = [
        d
        async for d in database[db.DISPATCHES]
        .find({"event_id": event_id, "status": {"$ne": "cancelled"}})
        .sort("priority", 1)
    ]
    run = await database[db.AGENT_RUNS].find_one({"event_id": event_id}, sort=[("created_at", -1)])

    graph, _n, _e, _ev = await load_graph(event_id)
    reach = reachability(graph)
    access = {}
    for node_id, r in reach.items():
        zid = graph.nodes[node_id].get("zone_id")
        if zid:
            access[str(zid)] = r["status"]

    zones.sort(
        key=lambda z: (
            SEVERITY_ORDER.get(z.get("severity", "none"), 9),
            -(z.get("needs") or {}).get("affected_people", 0),
        )
    )
    return {
        "event": event,
        "zones": zones,
        "nodes": nodes,
        "dispatches": dispatches,
        "run": run,
        "access": access,
    }


def text_summary(data: dict, top: int = 5) -> str:
    """Under one printed page; dictatable over a radio handset."""
    ev = data["event"]
    lines = [
        f"UDDHAR SITREP - {ev['name']}",
        f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Imagery-derived estimate; verify in field.",
        "",
    ]
    for i, z in enumerate([z for z in data["zones"] if z.get("severity") != "none"][:top], start=1):
        n = z.get("needs") or {}
        lon, lat = z["centroid"]["coordinates"]
        acc = data["access"].get(str(z["_id"]), "unknown").upper()
        sku = n.get("sku_demand", {})
        lines.append(
            f"{i}. {z['name'].upper()} {lat:.4f}N {lon:.4f}E - {z.get('severity', '').upper()}, "
            f"~{n.get('affected_people', 0):,} affected, access {acc}."
        )
        lines.append(
            f"   NEED: {sku.get('doctor', 0)} doctors, {sku.get('rescue_personnel', 0)} rescuers, "
            f"{sku.get('rice_25kg', 0)} rice sacks, {n.get('water_litres', 0):,} L water/day, "
            f"{sku.get('trauma_kit', 0)} trauma kits."
        )
        pending = [d for d in data["dispatches"] if d["zone_id"] == str(z["_id"])]
        if pending:
            for d in pending[:2]:
                items = ", ".join(f"{it['quantity']} {it['sku']}" for it in d["items"])
                lines.append(
                    f"   DISPATCH ({d['status']}): {items} by {d['transport_mode']} from "
                    f"{data['nodes'].get(d['from_node_id'], {}).get('name', '?')}."
                )
        elif acc == "UNREACHABLE":
            lines.append("   NO ROAD ACCESS - REQUIRES AIR TRANSPORT VERIFICATION.")
    if len(lines) == 3:
        lines.append("No damaged zones recorded.")
    return "\n".join(lines) + "\n"


def render_pdf(data: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    ev, run = data["event"], data["run"]
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=7.5, leading=9)
    story = []

    def table(rows, widths, header_bg=colors.HexColor("#1f2937")):
        t = Table(
            [[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=widths, repeatRows=1
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f3f4f6")],
                    ),
                ]
            )
        )
        return t

    story.append(Paragraph(f"Uddhar situation report - {ev['name']}", styles["Title"]))
    story.append(
        Paragraph(
            f"{ev.get('disaster_type', '')} &middot; {ev.get('region', '')} &middot; generated "
            f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC &middot; cloud cover "
            f"{100 * float(ev.get('cloud_fraction', 0)):.0f}% &middot; air transport "
            f"{'VERIFIED' if ev.get('air_transport_verified') else 'not verified'}",
            small,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Read this first", styles["Heading3"]))
    for item in LIMITATIONS:
        story.append(Paragraph(f"&bull; {item}", small))
    if run:
        mode = run.get("mode", "baseline")
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                f"<b>Allocation mode:</b> {mode}{' (DEGRADED)' if run.get('degraded') else ''}. "
                f"{run.get('summary', '')}",
                small,
            )
        )
        for note in run.get("notes", []):
            story.append(Paragraph(f"<b>Note:</b> {note}", small))

    overlay = get_settings().upload_dir / str(ev["_id"]) / "overlay.png"
    if Path(overlay).exists():
        from PIL import Image as PILImage

        # Downscale to print resolution as JPEG: the full overlay is several MB,
        # and this PDF may travel over a satellite phone link.
        with PILImage.open(overlay) as im:
            im = im.convert("RGB")
            im.thumbnail((1800, 1800))
            jpg = io.BytesIO()
            im.save(jpg, "JPEG", quality=80)
        jpg.seek(0)
        story.append(Spacer(1, 3 * mm))
        img = Image(jpg)
        scale = (180 * mm) / img.imageWidth
        img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
        story.append(img)
        story.append(
            Paragraph(
                "Pre-event | post-event | detected damage (red = destroyed, "
                "orange = major, yellow = minor).",
                small,
            )
        )

    story.append(Paragraph("Zones, worst first", styles["Heading2"]))
    rows = [
        [
            "Zone",
            "Severity",
            "Decided by",
            "Affected",
            "Access",
            "Doctors",
            "Rescuers",
            "Rice sacks",
            "Water L/day",
            "Trauma kits",
        ]
    ]
    for z in data["zones"]:
        n = z.get("needs") or {}
        s = n.get("sku_demand", {})
        rows.append(
            [
                z["name"],
                z.get("severity", "none"),
                z.get("decided_by", "none"),
                f"{n.get('affected_people', 0):,}",
                data["access"].get(str(z["_id"]), "unknown"),
                s.get("doctor", 0),
                s.get("rescue_personnel", 0),
                s.get("rice_25kg", 0),
                f"{n.get('water_litres', 0):,}",
                s.get("trauma_kit", 0),
            ]
        )
    story.append(
        table(
            rows,
            [
                26 * mm,
                16 * mm,
                16 * mm,
                16 * mm,
                18 * mm,
                14 * mm,
                15 * mm,
                15 * mm,
                18 * mm,
                16 * mm,
            ],
        )
    )

    worst = [z for z in data["zones"] if (z.get("needs") or {}).get("derivations")][:3]
    if worst:
        story.append(Paragraph("How the numbers were derived", styles["Heading3"]))
        for z in worst:
            aff = z["needs"]["derivations"].get("affected_people", {})
            story.append(
                Paragraph(f"<b>{z['name']}</b>: affected = {aff.get('formula', '')}", small)
            )

    # References: every distinct citation across the manifests, numbered once.
    refs: dict[str, int] = {}

    def ref(c: dict) -> int:
        key = c["source_url"] + "|" + c["claim"]
        if key not in refs:
            refs[key] = len(refs) + 1
        return refs[key]

    story.append(PageBreak())
    story.append(Paragraph("Dispatch manifests", styles["Heading2"]))
    if not data["dispatches"]:
        story.append(Paragraph("No dispatches planned yet.", small))
    else:
        rows = [["P", "Zone", "From", "Mode", "Items", "Route", "Status", "Refs"]]
        zone_names = {str(z["_id"]): z["name"] for z in data["zones"]}
        for d in data["dispatches"]:
            path = " &rarr; ".join(
                data["nodes"].get(p, {}).get("name", "?") for p in d.get("route", [])
            )
            rows.append(
                [
                    d["priority"],
                    zone_names.get(d["zone_id"], "?"),
                    data["nodes"].get(d["from_node_id"], {}).get("name", "?"),
                    d["transport_mode"],
                    "<br/>".join(f"{i['quantity']} {i['sku']}" for i in d["items"]),
                    path,
                    d["status"],
                    ", ".join(f"[{ref(c)}]" for c in d["citations"]),
                ]
            )
        story.append(
            table(rows, [7 * mm, 22 * mm, 28 * mm, 11 * mm, 28 * mm, 48 * mm, 16 * mm, 14 * mm])
        )
        story.append(Spacer(1, 2 * mm))
        for d in data["dispatches"][:12]:
            story.append(Paragraph(f"<b>P{d['priority']}</b> {d['rationale']}", small))

    if run and run.get("unmet"):
        story.append(Paragraph("Unmet need", styles["Heading2"]))
        rows = [["Zone", "Item", "Shortfall", "Reason"]]
        for u in run["unmet"]:
            rows.append([u["name"], u["sku"], u["shortfall"], u["reason"]])
        story.append(table(rows, [30 * mm, 28 * mm, 18 * mm, 98 * mm], colors.HexColor("#7f1d1d")))

    if run and run.get("parameters"):
        story.append(Paragraph("Planning parameters used", styles["Heading2"]))
        rows = [["Parameter", "Value", "Kind", "Adjusted by agent", "Refs"]]
        for p in run["parameters"].values():
            rows.append(
                [
                    p["name"],
                    f"{p['value']:g} {p['unit']}",
                    p["kind"],
                    p.get("note") or ("yes" if p.get("adjusted_by_agent") else "no"),
                    ", ".join(f"[{ref(c)}]" for c in p.get("citations", [])) or "-",
                ]
            )
        story.append(table(rows, [34 * mm, 34 * mm, 18 * mm, 72 * mm, 16 * mm]))

    if refs:
        story.append(Paragraph("References", styles["Heading2"]))
        by_n = {}
        for d in data["dispatches"]:
            for c in d["citations"]:
                by_n[ref(c)] = c
        for p in (run or {}).get("parameters", {}).values():
            for c in p.get("citations", []):
                by_n[ref(c)] = c
        for n in sorted(by_n):
            c = by_n[n]
            story.append(
                Paragraph(
                    f"[{n}] {c['claim']} <i>{c['source_title']}</i>"
                    f"{', ' + c['published'] if c.get('published') else ''}. {c['source_url']}",
                    small,
                )
            )

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"Uddhar report - {ev['name']}",
    ).build(story)
    return buf.getvalue()
