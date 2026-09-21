"""PDF field report generation (reportlab).

Renders the branded two-page UDDHAR field report: a header/footer band drawn on
every page, an executive summary with stat cards and a colour legend, the ranked
zone table with density highlights, a schematic zone grid, and a deterministic
priority recommendation section. The layout is generated entirely from the
``AnalysisResult`` so it looks identical for every report; the AI brief is used
only to supply the short "Current Assessment" paragraph.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.schemas import AnalysisResult, Zone

MAX_ZONE_ROWS = 16
GRID_COLS = 4

# --- Brand palette (print-friendly, light theme) -------------------------------
NAVY = HexColor("#1B2A4A")
ORANGE = HexColor("#F08A1D")
RED = HexColor("#DC3B2A")
GREEN = HexColor("#1E9E57")
MUTED = HexColor("#64748B")
LINE = HexColor("#CBD5E1")
LIGHT_BG = HexColor("#F1F5F9")
ZEBRA = HexColor("#F5F8FC")

# Tier tints used by the grid, table highlights, and recommendation badges.
TIER = {
    "red": {"main": RED, "tint": HexColor("#FCE9E7")},
    "orange": {"main": ORANGE, "tint": HexColor("#FDEFDD")},
    "green": {"main": GREEN, "tint": HexColor("#E7F6EE")},
}


_PAGE_W, _PAGE_H = LETTER
_MARGIN = 40
_CONTENT_W = _PAGE_W - 2 * _MARGIN
_PANEL_PAD = 14  # horizontal padding inside bordered panels


# --- Small helpers -------------------------------------------------------------
def _hex(color) -> str:
    """reportlab Color -> `#rrggbb` string for inline <font color> markup."""
    return "#" + color.hexval()[2:]


def _zone_coords(zone: Zone) -> str:
    if zone.centroid_lat is None or zone.centroid_lng is None:
        return "—"
    return f"{zone.centroid_lat:.4f}, {zone.centroid_lng:.4f}"


def _total_buildings(zone: Zone) -> int:
    b = zone.building_counts
    return b.none + b.minor + b.major + b.destroyed


def _event_title(pair_id: str | None) -> str:
    """`mexico-earthquake_00000005` -> `Mexico Earthquake #5`."""
    if not pair_id:
        return "Uploaded Imagery"
    base = pair_id
    suffix = ""
    if "_" in pair_id:
        base, tail = pair_id.rsplit("_", 1)
        if tail.isdigit():
            suffix = f" #{int(tail)}"
    name = re.sub(r"[-_]+", " ", base).strip().title()
    return f"{name}{suffix}" if name else pair_id


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.replace("*", "").replace("`", "").replace("#", "").strip()


def _extract_assessment(brief: str, fallback: str) -> str:
    """Pull the first prose paragraph out of the (markdown) AI brief."""
    if not brief:
        return fallback
    for raw in brief.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip markdown headings, tables, list bullets, and bold-only headers.
        if line[0] in "#|>-":
            continue
        if line.startswith("**") and line.endswith("**"):
            continue
        text = _strip_md(line)
        if len(text) >= 40:
            return text
    return fallback


def _overall_assessment(destroyed_pct: float, major_pct: float, minor_pct: float):
    """Return (label, tier) for the overall damage banner."""
    severe = destroyed_pct + major_pct
    if severe >= 15:
        return "SEVERE STRUCTURAL DAMAGE", "red"
    if severe >= 5 or minor_pct >= 20:
        return "MODERATE DAMAGE", "orange"
    if severe > 0 or minor_pct > 0:
        return "LIMITED DAMAGE", "orange"
    return "LOW OBSERVED DAMAGE", "green"


def _zone_tiers(zones: list[Zone], has_damage: bool):
    """Assign each zone rank a tier; return (rank->tier, focus_ranks_by_tier)."""
    density_order = sorted(zones, key=_total_buildings, reverse=True)
    # Grid colour scale: top-3 density = red accents, the next density band =
    # orange (secondary), everything else green (routine). The orange band is
    # kept tighter than the recommendation's mid band so only the genuinely
    # higher-density zones read warm.
    top = {z.rank for z in density_order[:3]}
    mid = {z.rank for z in density_order[3:8]}

    tiers: dict[int, str] = {}
    for zone in zones:
        if has_damage:
            dc = zone.damage_counts
            if dc.destroyed > 0 or dc.major > 0:
                tier = "red"
            elif dc.minor > 0:
                tier = "orange"
            else:
                tier = "green"
        else:
            if zone.rank in top:
                tier = "red"
            elif zone.rank in mid:
                tier = "orange"
            else:
                tier = "green"
        tiers[zone.rank] = tier
    return tiers, density_order


# --- Compass rose flowable (page 2 grid) ---------------------------------------
class _Compass(Flowable):
    def __init__(self, size: float = 44):
        super().__init__()
        self.width = size
        self.height = size

    def draw(self):
        c = self.canv
        s = self.width
        x = s / 2
        c.setStrokeColor(NAVY)
        c.setFillColor(NAVY)
        c.setLineWidth(1.4)
        c.line(x, 4, x, s - 16)  # shaft
        c.line(x, s - 12, x - 5, s - 20)  # arrow head
        c.line(x, s - 12, x + 5, s - 20)  # arrow head
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(x, s - 10, "N")


# --- Page decoration (header band, footer, watermark) --------------------------
def _make_page_decorator(subtitle: str, pair_line: str):
    def decorate(canvas, doc):
        w, h = _PAGE_W, _PAGE_H

        # Watermark (behind everything).
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 88)
        canvas.setFillColor(HexColor("#EEF2F7"))
        canvas.translate(w / 2, h / 2)
        canvas.rotate(32)
        canvas.drawCentredString(0, 0, "UDDHAR")
        canvas.restoreState()

        # Top accent bar: navy with an orange block on the right.
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 6, w, 6, fill=1, stroke=0)
        canvas.setFillColor(ORANGE)
        canvas.rect(w * 0.72, h - 6, w * 0.28, 6, fill=1, stroke=0)

        # Wordmark only — the title block starts at the margin now that there is
        # no logo to indent past.
        tx = _MARGIN
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 19)
        canvas.drawString(tx, h - 40, "UDDHAR Field Damage Report")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 9.5)
        canvas.drawString(tx, h - 56, subtitle)
        canvas.setFont("Helvetica", 8.5)
        canvas.drawString(tx, h - 69, pair_line)

        # Header rule.
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.8)
        canvas.line(_MARGIN, h - 84, w - _MARGIN, h - 84)

        # Footer.
        canvas.setStrokeColor(LINE)
        canvas.line(_MARGIN, 42, w - _MARGIN, 42)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            _MARGIN,
            31,
            "UDDHAR  |  AI-assisted satellite damage triage  |  Preliminary assessment",
        )
        canvas.drawRightString(w - _MARGIN, 31, f"Page {doc.page}")

    return decorate


# --- Styles --------------------------------------------------------------------
_H2 = ParagraphStyle(
    "Section", fontName="Helvetica-Bold", fontSize=13, textColor=NAVY,
    spaceBefore=11, spaceAfter=6, leading=15,
)
_BODY = ParagraphStyle(
    "Body", fontName="Helvetica", fontSize=9.5, textColor=HexColor("#334155"),
    leading=14,
)
_SMALL = ParagraphStyle(
    "Small", fontName="Helvetica", fontSize=8, textColor=MUTED, leading=11,
)
_LABEL = ParagraphStyle(
    "Label", fontName="Helvetica-Bold", fontSize=7.5, textColor=MUTED, leading=10,
)
_STAT_LABEL = ParagraphStyle(
    "StatLabel", fontName="Helvetica-Bold", fontSize=7.5, textColor=MUTED, leading=9,
)
_STAT_VALUE = ParagraphStyle(
    "StatValue", fontName="Helvetica-Bold", fontSize=21, textColor=NAVY, leading=22,
)
_ASSESS_VALUE = ParagraphStyle(
    "AssessValue", fontName="Helvetica-Bold", fontSize=17, textColor=NAVY, leading=20,
)


def _cell(html: str, style: ParagraphStyle = _BODY) -> Paragraph:
    return Paragraph(html, style)


# --- Section builders ----------------------------------------------------------
def _status_strip() -> Table:
    left = _cell(
        '<font size=7.5 color="#B45309"><b>REPORT STATUS:</b></font><br/>'
        '<font size=10 color="#C2410C"><b>PRELIMINARY AI ASSESSMENT</b></font>',
        _BODY,
    )
    right = _cell(
        "Human verification is required before field deployment or "
        "emergency-response decisions.",
        _BODY,
    )
    table = Table([[left, right]], colWidths=[2.15 * inch, _CONTENT_W - 2.15 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FEF6EC")),
                ("LINEBEFORE", (0, 0), (0, 0), 3.5, ORANGE),
                ("BOX", (0, 0), (-1, -1), 0.6, HexColor("#F5D9B5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _stat_cards(summary) -> Table:
    def card(label: str, value: str, color: HexColor) -> list:
        return [
            Paragraph(label, _STAT_LABEL),
            Spacer(1, 4),
            Paragraph(f'<font color="{_hex(color)}">{value}</font>', _STAT_VALUE),
        ]

    dest_color = RED if summary.destroyed_pct > 0 else NAVY
    major_color = ORANGE if summary.major_pct > 0 else NAVY
    minor_color = HexColor("#CA8A04") if summary.minor_pct > 0 else NAVY

    cards = [
        card("TOTAL BUILDINGS", str(summary.total_buildings), NAVY),
        card("DESTROYED", f"{summary.destroyed_pct}%", dest_color),
        card("MAJOR", f"{summary.major_pct}%", major_color),
        card("MINOR", f"{summary.minor_pct}%", minor_color),
    ]
    col = _CONTENT_W / 4
    table = Table([cards], colWidths=[col] * 4)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), white),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
            ]
        )
    )
    return table


def _assessment_and_legend(label: str, tier: str) -> Table:
    main = TIER[tier]["main"]
    left = [
        Paragraph("OVERALL ASSESSMENT", _STAT_LABEL),
        Spacer(1, 10),
        Paragraph(f'<font color="{_hex(main)}">{label}</font>', _ASSESS_VALUE),
    ]

    def legend_row(color: HexColor, name: str, desc: str):
        badge = _cell(f'<font color="#FFFFFF"><b>{name}</b></font>', _LABEL)
        badge_tbl = Table([[badge]], colWidths=[0.7 * inch])
        badge_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), color),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return [badge_tbl, _cell(desc, _SMALL)]

    legend = Table(
        [
            [_cell('<font color="#64748B"><b>COLOR LEGEND</b></font>', _LABEL), ""],
            legend_row(RED, "RED", "Immediate verification"),
            legend_row(ORANGE, "ORANGE", "Secondary assessment"),
            legend_row(GREEN, "GREEN", "Routine monitoring / low observed risk"),
        ],
        colWidths=[0.85 * inch, 2.6 * inch],
    )
    legend.setStyle(
        TableStyle(
            [
                ("SPAN", (0, 0), (1, 0)),
                ("BACKGROUND", (0, 0), (-1, -1), white),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 1), (-1, -1), 0.4, LIGHT_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    outer = Table(
        [[left, legend]], colWidths=[_CONTENT_W - 3.65 * inch, 3.65 * inch]
    )
    outer.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), LIGHT_BG),
                ("BOX", (0, 0), (0, 0), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 14),
                ("RIGHTPADDING", (0, 0), (0, 0), 14),
                ("TOPPADDING", (0, 0), (0, 0), 10),
                ("BOTTOMPADDING", (0, 0), (0, 0), 10),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
            ]
        )
    )
    return outer


def _zone_table(zones: list[Zone], highlight: set[int]) -> Table:
    header = ["Rank", "Coordinates (lat, lng)", "Destroyed", "Major", "Minor", "None", "Priority"]
    rows = [header]
    for zone in zones[:MAX_ZONE_ROWS]:
        rows.append(
            [
                str(zone.rank),
                _zone_coords(zone),
                str(zone.building_counts.destroyed),
                str(zone.building_counts.major),
                str(zone.building_counts.minor),
                str(zone.building_counts.none),
                f"{zone.priority_score:.1f}",
            ]
        )

    col_widths = [
        0.55 * inch, 1.95 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch, 0.75 * inch, 0.85 * inch,
    ]
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, ZEBRA]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    for idx, zone in enumerate(zones[:MAX_ZONE_ROWS], start=1):
        if zone.rank in highlight:
            style.append(("BACKGROUND", (0, idx), (-1, idx), TIER["orange"]["tint"]))
            style.append(("FONTNAME", (0, idx), (-1, idx), "Helvetica-Bold"))
            style.append(("TEXTCOLOR", (0, idx), (-1, idx), HexColor("#9A3412")))
    table.setStyle(TableStyle(style))
    return table


def _zone_grid(zones: list[Zone], tiers: dict[int, str]) -> Table:
    visible = zones[:MAX_ZONE_ROWS]
    cells: list[list] = []
    row: list = []
    grid_style: list = []

    for i, zone in enumerate(visible):
        tier = tiers.get(zone.rank, "green")
        main = _hex(TIER[tier]["main"])
        cell = _cell(
            f'<font size=9 color="{main}"><b>Zone {i + 1}</b></font><br/>'
            f'<font size=7.5 color="#475569">{_total_buildings(zone)} buildings</font>',
            _BODY,
        )
        r, c = divmod(i, GRID_COLS)
        grid_style.append(("BOX", (c, r), (c, r), 1.2, TIER[tier]["main"]))
        grid_style.append(("BACKGROUND", (c, r), (c, r), TIER[tier]["tint"]))
        row.append(cell)
        if len(row) == GRID_COLS:
            cells.append(row)
            row = []
    if row:
        while len(row) < GRID_COLS:
            row.append("")
        cells.append(row)

    if not cells:
        cells = [[""] * GRID_COLS]

    # The grid + compass sit inside a panel that has 14pt of horizontal padding
    # on each side, so the available width is _CONTENT_W minus that padding.
    compass_w = 0.7 * inch
    inner_w = _CONTENT_W - 2 * _PANEL_PAD - compass_w
    grid = Table(
        cells,
        colWidths=[inner_w / GRID_COLS] * GRID_COLS,
        rowHeights=[34] * len(cells),
    )
    grid_style.extend(
        [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]
    )
    grid.setStyle(TableStyle(grid_style))

    body = Table([[_Compass(), grid]], colWidths=[compass_w, inner_w])
    body.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
                ("VALIGN", (1, 0), (1, 0), "TOP"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
            ]
        )
    )

    caption = _cell(
        '<para align="right"><font size=7 color="#94A3B8">Approx. grid overview</font></para>',
        _SMALL,
    )
    panel = Table([[body], [caption]], colWidths=[_CONTENT_W])
    panel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), _PANEL_PAD),
                ("RIGHTPADDING", (0, 0), (-1, -1), _PANEL_PAD),
                ("TOPPADDING", (0, 0), (0, 0), 14),
                ("BOTTOMPADDING", (0, 1), (0, 1), 8),
                ("TOPPADDING", (0, 1), (0, 1), 0),
            ]
        )
    )
    return panel


def _current_assessment_box(text: str) -> Table:
    inner = [
        _cell('<font color="#64748B"><b>CURRENT ASSESSMENT</b></font>', _LABEL),
        Spacer(1, 6),
        _cell(escape(text), _BODY),
    ]
    box = Table([[inner]], colWidths=[_CONTENT_W])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("LINEBEFORE", (0, 0), (0, 0), 3.5, ORANGE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return box


def _recommendations(zones: list[Zone], has_damage: bool, density_order: list[Zone]):
    """Deterministic three-tier priority recommendation rows."""
    order = density_order if not has_damage else sorted(
        zones, key=lambda z: z.priority_score, reverse=True
    )
    metric = "priority" if has_damage else "buildings"

    def count(z: Zone) -> int:
        return _total_buildings(z)

    top = order[:3]
    if len(top) >= 3:
        a, b, c = top
        if metric == "buildings":
            tail = (
                f"followed by Ranks {b.rank} and {c.rank} "
                f"({count(b)} buildings each)"
                if count(b) == count(c)
                else f"followed by Rank {b.rank} ({count(b)} buildings) and "
                f"Rank {c.rank} ({count(c)} buildings)"
            )
            body1 = (
                f"Prioritize Rank {a.rank} ({count(a)} buildings), {tail}. These "
                "zones contain the greatest number of structures and offer the "
                "highest value for early verification."
            )
        else:
            body1 = (
                f"Prioritize Rank {a.rank}, followed by Ranks {b.rank} and "
                f"{c.rank}. These zones carry the highest model priority scores "
                "and should receive the first assessment teams."
            )
    elif top:
        ranks = ", ".join(f"Rank {z.rank}" for z in top)
        body1 = (
            f"Prioritize {ranks} for the first assessment teams based on the "
            "current ranking."
        )
    else:
        body1 = "No zones were available for ranking."

    mid = order[3:9]
    if mid:
        mid_ranks = ", ".join(str(z.rank) for z in sorted(mid, key=lambda z: z.rank))
        counts = [count(z) for z in mid]
        body2 = (
            f"Proceed to Ranks {mid_ranks}, where each zone contains approximately "
            f"{min(counts)}-{max(counts)} buildings."
        )
    else:
        body2 = "No mid-tier zones require secondary assessment at this time."

    # Reference the contiguous bottom band of the ranked table (lowest-priority
    # rows), e.g. Ranks 11-16 for a 16-zone grid.
    all_ranks = sorted(z.rank for z in zones)
    if all_ranks:
        cut = (2 * len(all_ranks)) // 3
        low_band = all_ranks[cut:] or all_ranks[-1:]
        detail = "the current absence of visible damage continues" if not has_damage \
            else "conditions remain stable"
        body3 = (
            f"Conduct scheduled monitoring in the lower-priority zones, especially "
            f"Ranks {low_band[0]}-{low_band[-1]}, to confirm that {detail}."
        )
    else:
        body3 = "Maintain routine monitoring across the remaining zones."

    specs = [
        ("01", "Deploy rapid-assessment teams first", body1, "HIGHEST", "red"),
        ("02", "Cover the mid-density zones next", body2, "MEDIUM", "orange"),
        ("03", "Maintain periodic checks elsewhere", body3, "LOW RISK", "green"),
    ]

    rows = []
    for num, title, body, badge, tier in specs:
        main = TIER[tier]["main"]
        num_cell = _cell(
            f'<font size=17 color="{_hex(main)}"><b>{num}</b></font>',
            _BODY,
        )
        body_cell = _cell(
            f'<font color="#0F172A"><b>{escape(title)}</b></font><br/>'
            f'<font size=8.5 color="#475569">{escape(body)}</font>',
            _BODY,
        )
        badge_cell = _cell(f'<font color="#FFFFFF"><b>{badge}</b></font>', _LABEL)
        badge_tbl = Table([[badge_cell]], colWidths=[1.0 * inch])
        badge_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), main),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        row = Table(
            [[num_cell, body_cell, badge_tbl]],
            colWidths=[0.55 * inch, _CONTENT_W - 0.55 * inch - 1.3 * inch, 1.3 * inch],
        )
        row.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), white),
                    ("LINEBEFORE", (0, 0), (0, 0), 3.5, main),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("VALIGN", (0, 0), (1, 0), "MIDDLE"),
                    ("VALIGN", (2, 0), (2, 0), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("LEFTPADDING", (1, 0), (1, 0), 10),
                    ("RIGHTPADDING", (2, 0), (2, 0), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        rows.append(row)
        rows.append(Spacer(1, 6))
    return rows


def _operational_note(has_damage: bool) -> Table:
    text = (
        "<b>Operational note:</b> Green indicates low observed risk, not guaranteed "
        "safety. Satellite imagery cannot rule out hidden, interior, or delayed "
        "structural damage."
    )
    note = Table([[_cell(text, _SMALL)]], colWidths=[_CONTENT_W])
    note.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FEF6EC")),
                ("BOX", (0, 0), (-1, -1), 0.6, HexColor("#F5D9B5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return note


# --- Entry point ---------------------------------------------------------------
def generate_report_pdf(analysis: AnalysisResult, brief: str) -> bytes:
    buffer = io.BytesIO()

    now = datetime.now()
    subtitle = f"{_event_title(analysis.pair_id)}   |   {now.day} {now:%B %Y}"
    pair_line = (
        f"Pair ID: {analysis.pair_id or 'Uploaded imagery'}   |   "
        f"Inference mode: {analysis.inference_mode}"
    )

    doc = BaseDocTemplate(
        buffer,
        pagesize=LETTER,
        title="UDDHAR Field Report",
        author="UDDHAR",
    )
    frame = Frame(
        _MARGIN, 52, _CONTENT_W, _PAGE_H - 52 - 96, id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    decorate = _make_page_decorator(subtitle, pair_line)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])

    zones = analysis.zones
    summary = analysis.summary
    has_damage = (
        summary.destroyed_pct > 0
        or summary.major_pct > 0
        or summary.minor_pct > 0
        or any(z.priority_score > 0 for z in zones)
    )
    tiers, density_order = _zone_tiers(zones, has_damage)

    if has_damage:
        highlight = {z.rank for z in sorted(zones, key=lambda z: z.priority_score, reverse=True)[:3]}
    else:
        highlight = {z.rank for z in density_order[:3]}

    overall_label, overall_tier = _overall_assessment(
        summary.destroyed_pct, summary.major_pct, summary.minor_pct
    )

    fallback_assessment = (
        f"No visible structural damage was detected across the {len(zones)} examined "
        'zones. Every zone currently reports only the "none" damage class. The '
        "operational focus should therefore shift to verification, early-stage "
        "triage, and continued monitoring."
        if not has_damage
        else (
            f"Damage was detected across the analyzed zones, with {summary.destroyed_pct}% "
            f"destroyed and {summary.major_pct}% major damage among assessed buildings. "
            "Rank-ordered zones should receive assessment teams in priority order, "
            "starting with the highest-priority clusters."
        )
    )
    assessment_text = _extract_assessment(brief, fallback_assessment)

    story: list = []

    # --- Page 1 ---
    story.append(_status_strip())
    story.append(Spacer(1, 10))
    story.append(Paragraph("EXECUTIVE SUMMARY", _H2))
    story.append(_stat_cards(summary))
    story.append(Spacer(1, 9))
    story.append(_assessment_and_legend(overall_label, overall_tier))
    story.append(Paragraph("RANKED PRIORITY ZONES", _H2))
    story.append(_zone_table(zones, highlight))
    story.append(Spacer(1, 5))
    caption = (
        "Highlighted rows identify the highest "
        f"{'priority' if has_damage else 'building-density'} zones for rapid field "
        "verification."
    )
    if not has_damage:
        caption += " All current model priority scores are 0.0."
    if not analysis.geo_available and analysis.geo_message:
        caption += f" {analysis.geo_message}"
    story.append(_cell(caption, _SMALL))

    # --- Page 2 ---
    story.append(PageBreak())
    story.append(Paragraph("GEOGRAPHIC OVERVIEW", _H2))
    story.append(
        _cell(
            "The schematic below shows the analyzed zone grid and applies the "
            "operational color scale. It is a report overview, not a substitute for "
            "a georeferenced field map.",
            _BODY,
        )
    )
    story.append(Spacer(1, 10))
    story.append(_zone_grid(zones, tiers))

    story.append(Paragraph("SITUATION BRIEF", _H2))
    story.append(_current_assessment_box(assessment_text))
    story.append(Paragraph("PRIORITY RECOMMENDATION", _H2))
    story.extend(_recommendations(zones, has_damage, density_order))
    story.append(Spacer(1, 4))
    story.append(_operational_note(has_damage))

    doc.build(story)
    return buffer.getvalue()
