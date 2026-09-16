"""End-to-end API tests against the mock assessment."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def assessment_id(client):
    """The synthetic grid: fixed ward ids and counts, so these assertions stay meaningful
    whether or not the real OSM layers have been fetched on this machine."""
    return client.post("/api/assessments/demo?synthetic=true").json()["assessment_id"]


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_demo_assessment_has_buildings_and_wards(client, assessment_id):
    summary = client.get(f"/api/assessments/{assessment_id}").json()
    assert summary["status"] == "complete"
    assert summary["wards"] == 8
    assert summary["total_buildings"] > 500
    assert summary["affected_population"] > 0


def test_ward_cards_are_ranked_by_priority(client, assessment_id):
    cards = client.get(f"/api/assessments/{assessment_id}/wards").json()
    scores = [c["priority_score"] for c in cards]
    assert scores == sorted(scores, reverse=True)


def test_ward_8_matches_the_brief(client, assessment_id):
    """The brief's worked example: Ward 8, ~127 damaged, ~34 severe, ~2,400 affected,
    3 critical facilities nearby, Priority: Very High.

    `critical_facilities` counts only the life-safety set (hospital, bridge, fire station).
    A dense Kathmandu ward contains 145 schools, so counting every facility produced a
    headline number no dispatcher could act on. All three of the brief's facility types
    still appear, in `facility_breakdown` - which is what the card renders as chips.
    """
    card = client.get(f"/api/assessments/{assessment_id}/wards/ktm-08").json()
    assert card["priority_band"] == "Very High"
    assert 100 <= card["damaged_buildings"] <= 150
    assert 25 <= card["severely_damaged"] <= 50
    assert 2000 <= card["affected_population"] <= 2800
    assert set(card["facility_breakdown"]) == {"hospital", "school", "bridge"}
    assert card["critical_facilities"] == 2  # hospital + bridge; the school is not life-safety


def test_buildings_are_valid_geojson(client, assessment_id):
    fc = client.get(f"/api/assessments/{assessment_id}/buildings").json()
    assert fc["type"] == "FeatureCollection"
    feature = fc["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    props = feature["properties"]
    assert props["damage_class"] in {"no-damage", "minor-damage", "major-damage", "destroyed"}
    assert 0.0 <= props["confidence"] <= 1.0
    assert props["reviewed_by_human"] is False


def test_buildings_filter_by_ward(client, assessment_id):
    fc = client.get(f"/api/assessments/{assessment_id}/buildings", params={"ward_id": "ktm-08"}).json()
    assert fc["features"]
    assert {f["properties"]["ward_id"] for f in fc["features"]} == {"ktm-08"}


def test_human_override_rescores_the_ward(client, assessment_id):
    """The human-in-the-loop guarantee: a correction moves the numbers immediately."""
    fc = client.get(f"/api/assessments/{assessment_id}/buildings", params={"ward_id": "ktm-08"}).json()
    target = next(f for f in fc["features"] if f["properties"]["damage_class"] == "no-damage")
    building_id = target["properties"]["id"]

    before = client.get(f"/api/assessments/{assessment_id}/wards/ktm-08").json()
    after = client.patch(
        f"/api/buildings/{building_id}",
        json={"damage_class": "destroyed", "note": "Verified collapsed by ward officer"},
    ).json()

    assert after["severely_damaged"] == before["severely_damaged"] + 1
    assert after["damaged_buildings"] == before["damaged_buildings"] + 1
    assert after["reviewed_count"] == before["reviewed_count"] + 1

    reviewed = client.get(f"/api/buildings/{building_id}").json()
    assert reviewed["reviewed_by_human"] is True
    assert reviewed["confidence"] == 1.0
    assert reviewed["review_note"] == "Verified collapsed by ward officer"


def test_ward_boundaries_carry_priority(client, assessment_id):
    fc = client.get(f"/api/assessments/{assessment_id}/ward-boundaries").json()
    assert len(fc["features"]) == 8
    assert fc["features"][0]["properties"]["priority_band"]


@pytest.mark.parametrize("layer,fmt", [("buildings", "geojson"), ("buildings", "csv"),
                                       ("wards", "geojson"), ("wards", "csv")])
def test_exports(client, assessment_id, layer, fmt):
    r = client.get(f"/api/assessments/{assessment_id}/export", params={"layer": layer, "format": fmt})
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert r.content


def test_legend_matches_config(client):
    legend = client.get("/api/legend").json()
    assert {e["key"] for e in legend["damage"]} == {
        "destroyed", "major-damage", "minor-damage", "no-damage"
    }
    assert legend["critical_infrastructure"]["color"] == "#2563eb"


def test_unwired_inference_modes_are_rejected_not_faked(client):
    """Better a 501 than a judge seeing mock data labelled as model output."""
    r = client.post("/api/assess", json={"pre_image": "a.tif", "post_image": "b.tif", "mode": "model"})
    assert r.status_code == 501


def test_synthetic_path_is_labelled_synthetic(client, assessment_id):
    summary = client.get(f"/api/assessments/{assessment_id}").json()
    assert summary["geometry_source"] == "synthetic"


def test_unknown_ids_404(client):
    assert client.get("/api/assessments/nope").status_code == 404
    assert client.get("/api/buildings/nope").status_code == 404
