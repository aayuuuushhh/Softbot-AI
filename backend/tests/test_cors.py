"""CORS allowlist regression — Vercel production must be able to call the API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

VERCEL_ORIGIN = "https://disasteriq.vercel.app"


def test_default_cors_origins_include_production_vercel():
    assert VERCEL_ORIGIN in settings.cors_origins


def test_health_allows_disasteriq_vercel_origin():
    resp = client.get("/health", headers={"Origin": VERCEL_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == VERCEL_ORIGIN


def test_health_rejects_unlisted_origin():
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example"
