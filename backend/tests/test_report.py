from __future__ import annotations

from app.schemas import AnalysisResult, AnalysisSummary, BuildingCounts, DamageCounts, Zone
from app.services.report import generate_report_pdf


def _sample_zone(rank: int, *, with_geo: bool) -> Zone:
    return Zone(
        rank=rank,
        bbox=[0, 0, 10, 10],
        damage_counts=DamageCounts(none=1, minor=2, major=3, destroyed=4),
        building_counts=BuildingCounts(none=1, minor=1, major=1, destroyed=1),
        priority_score=42.5,
        centroid_lat=34.796123 if with_geo else None,
        centroid_lng=-92.356789 if with_geo else None,
    )


def test_generate_report_pdf_returns_valid_pdf_bytes():
    analysis = AnalysisResult(
        zones=[_sample_zone(1, with_geo=True), _sample_zone(2, with_geo=True)],
        summary=AnalysisSummary(
            total_building_pixels=1000,
            total_buildings=2,
            destroyed_pct=50.0,
            major_pct=50.0,
            minor_pct=0.0,
        ),
        pair_id="midwest-flooding_00000000",
        inference_mode="stub",
        geo_available=True,
    )

    pdf_bytes = generate_report_pdf(analysis, "Deploy teams to zone 1 first.")

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_generate_report_pdf_handles_missing_geo_and_special_chars():
    analysis = AnalysisResult(
        zones=[_sample_zone(1, with_geo=False)],
        summary=AnalysisSummary(
            total_building_pixels=100,
            total_buildings=1,
            destroyed_pct=100.0,
            major_pct=0.0,
            minor_pct=0.0,
        ),
        pair_id=None,
        inference_mode="stub",
        geo_available=False,
        geo_message="Geographic coordinates available for demo pairs with xBD metadata only.",
    )

    brief = "Zone 1 <critical> & unassessed\nSecond line with \"quotes\""
    pdf_bytes = generate_report_pdf(analysis, brief)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500
