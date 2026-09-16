"""The real-layer path: OSM wards/buildings/facilities + census-anchored WorldPop.

Skipped when the layers have not been fetched (`python scripts/fetch_osm.py --all`), so the
suite still passes on a clean clone.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.scenario import build_real_assessment, layers_available

pytestmark = pytest.mark.skipif(
    not layers_available(), reason="OSM layers not fetched; run scripts/fetch_osm.py --all"
)


@pytest.fixture(scope="module")
def assessment():
    return build_real_assessment(assessment_id="test", created_at="2026-01-01T00:00:00")


def test_uses_real_kathmandu_wards(assessment):
    assert assessment.geometry_source.value == "osm"
    assert len(assessment.wards) > 20
    assert any(w.ward_name == "Ward 8" for w in assessment.wards)


def test_buildings_are_real_osm_footprints(assessment):
    assert len(assessment.buildings) > 1000
    assert all(b.id.startswith("osm-") for b in assessment.buildings[:50])
    # Real footprints are irregular; the synthetic grid emits 5-point rectangles.
    assert any(len(b.geometry["coordinates"][0]) != 5 for b in assessment.buildings)


def test_damage_decays_with_distance_from_the_epicentre(assessment):
    """The whole point of the shaking field: priority must come out of geography.

    Wards near the epicentre should be hit harder than wards far from it.
    """
    from backend.app.services.score import count_damaged

    cards = assessment.ward_cards()
    by_id = {w.ward_id: w for w in assessment.wards}
    rates = [
        (by_id[c.ward_id].distance_from_staging_km, c.damaged_buildings / max(1, c.total_buildings))
        for c in cards
    ]
    near = [r for d, r in rates if d < 2.0]
    far = [r for d, r in rates if d > 5.0]
    assert near and far, "need wards both near and far from the epicentre"
    assert sum(near) / len(near) > sum(far) / len(far)
    assert count_damaged(cards[0].damage_breakdown) > 0


def test_moving_the_epicentre_re_ranks_the_wards():
    """If the top ward does not change when the earthquake moves, the score is not
    actually responding to geography."""
    west = build_real_assessment("w", "2026-01-01T00:00:00", epicentre=(85.29, 27.69), radius_km=2.0)
    east = build_real_assessment("e", "2026-01-01T00:00:00", epicentre=(85.35, 27.73), radius_km=2.0)
    assert west.ward_cards()[0].ward_id != east.ward_cards()[0].ward_id


def test_damage_is_plausible_not_apocalyptic(assessment):
    """A model that flattens half of Kathmandu is not credible to anyone who was there."""
    cards = assessment.ward_cards()
    total = sum(c.total_buildings for c in cards)
    damaged = sum(c.damaged_buildings for c in cards)
    assert 0.02 < damaged / total < 0.35


def test_population_is_census_anchored(assessment):
    """Raw WorldPop reads ~3x high over Kathmandu (see config/priority.yaml). After
    calibration the AOI total must land near the 2021 census figure of ~846k."""
    cards = assessment.ward_cards()
    by_id = {w.ward_id: w for w in assessment.wards}
    ward_totals = sum(
        by_id[c.ward_id].people_per_building * c.total_buildings for c in cards
    )
    assert 500_000 < ward_totals < 1_200_000


def test_facility_counts_are_not_saturated(assessment):
    """Every ward having the same facility score means the term is doing no work."""
    cards = assessment.ward_cards()
    scores = {round(c.breakdown.infrastructure, 3) for c in cards}
    assert len(scores) > 3


def test_api_serves_the_real_layers():
    client = TestClient(app)
    summary = client.post("/api/assessments/demo").json()
    assert summary["geometry_source"] == "osm"
    assert summary["aoi_name"] == "Kathmandu Metropolitan City"
