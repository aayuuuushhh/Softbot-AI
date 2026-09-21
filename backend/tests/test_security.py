from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import security
from app.config import settings
from app.main import app

client = TestClient(app)

BRIEF_BODY = {
    "analysis": {
        "zones": [],
        "summary": {
            "total_building_pixels": 0,
            "total_buildings": 0,
            "destroyed_pct": 0.0,
            "major_pct": 0.0,
            "minor_pct": 0.0,
        },
        "inference_mode": "stub",
    },
    "context": None,
}


@pytest.fixture(autouse=True)
def reset_security_state():
    """Snapshot and restore mutable security config + limiter state per test."""
    orig_token = settings.access_token
    orig_limit = settings.rate_limit_requests
    orig_window = settings.rate_limit_window_seconds
    security._hits.clear()
    yield
    settings.access_token = orig_token
    settings.rate_limit_requests = orig_limit
    settings.rate_limit_window_seconds = orig_window
    security._hits.clear()


def test_rate_limit_allows_then_blocks():
    settings.access_token = ""
    settings.rate_limit_requests = 3
    settings.rate_limit_window_seconds = 60

    for _ in range(3):
        assert client.post("/brief", json=BRIEF_BODY).status_code == 200

    blocked = client.post("/brief", json=BRIEF_BODY)
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers


def test_rate_limit_disabled_when_zero():
    settings.access_token = ""
    settings.rate_limit_requests = 0

    for _ in range(20):
        assert client.post("/brief", json=BRIEF_BODY).status_code == 200


def test_access_token_enforced_when_set():
    settings.access_token = "s3cret"
    settings.rate_limit_requests = 0

    assert client.post("/brief", json=BRIEF_BODY).status_code == 401
    assert (
        client.post("/brief", json=BRIEF_BODY, headers={"X-API-Key": "wrong"}).status_code
        == 401
    )
    assert (
        client.post("/brief", json=BRIEF_BODY, headers={"X-API-Key": "s3cret"}).status_code
        == 200
    )


def test_access_token_unequal_length_returns_401():
    settings.access_token = "s3cret"
    settings.rate_limit_requests = 0
    assert (
        client.post("/brief", json=BRIEF_BODY, headers={"X-API-Key": "x"}).status_code
        == 401
    )


def test_open_access_by_default():
    settings.access_token = ""
    settings.rate_limit_requests = 0
    assert client.post("/brief", json=BRIEF_BODY).status_code == 200


def test_health_is_unprotected():
    settings.access_token = "s3cret"
    settings.rate_limit_requests = 1
    # health has no guards, so it stays reachable even with token + tight limit
    for _ in range(5):
        assert client.get("/health").status_code == 200
