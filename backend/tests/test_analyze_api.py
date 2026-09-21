from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.main import app

client = TestClient(app)


def _png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    rng = np.random.default_rng(seed=7)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _upload_files() -> dict:
    return {
        "pre_image": ("pre.png", _png_bytes(), "image/png"),
        "post_image": ("post.png", _png_bytes(), "image/png"),
    }


def test_health_reports_mode_and_pair_count():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["inference_mode"] == "stub"
    assert body["demo_pairs"] >= 1


def test_demo_pairs_are_listed():
    resp = client.get("/demo/pairs")
    assert resp.status_code == 200
    pairs = resp.json()
    assert pairs
    assert {"id", "disaster_type", "pre_image", "post_image"} <= set(pairs[0])


def test_analyze_demo_pair_returns_ranked_zones():
    pair_id = client.get("/demo/pairs").json()[0]["id"]

    resp = client.post("/analyze", data={"demo_pair_id": pair_id})

    assert resp.status_code == 200
    body = resp.json()
    assert body["pair_id"] == pair_id
    assert body["inference_mode"].startswith("stub")

    zones = body["zones"]
    assert zones, "stub inference should produce at least one damaged zone"
    assert [z["rank"] for z in zones] == list(range(1, len(zones) + 1))

    scores = [z["priority_score"] for z in zones]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 100.0 for s in scores)


def test_analyze_accepts_uploads():
    resp = client.post("/analyze", files=_upload_files())
    assert resp.status_code == 200
    assert resp.json()["geo_available"] is False


def test_analyze_requires_both_images():
    resp = client.post("/analyze", files={"pre_image": ("pre.png", _png_bytes(), "image/png")})
    assert resp.status_code == 400
    assert "demo_pair_id" in resp.json()["detail"]


def test_analyze_rejects_unknown_demo_pair():
    resp = client.post("/analyze", data={"demo_pair_id": "no-such-pair"})
    assert resp.status_code == 404


def test_analyze_rejects_non_image_content_type():
    files = {
        "pre_image": ("pre.pdf", b"%PDF-1.4", "application/pdf"),
        "post_image": ("post.png", _png_bytes(), "image/png"),
    }
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 400
    assert "Unsupported image type" in resp.json()["detail"]


def test_analyze_rejects_bytes_that_are_not_an_image():
    files = {
        "pre_image": ("pre.png", b"definitely not a png", "image/png"),
        "post_image": ("post.png", _png_bytes(), "image/png"),
    }
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 400
    assert "not a valid image" in resp.json()["detail"]


def test_analyze_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 128)

    resp = client.post("/analyze", files=_upload_files())

    assert resp.status_code == 400
    assert "25 MB" in resp.json()["detail"] or "limit" in resp.json()["detail"]


def test_rejected_upload_leaves_no_job_directory(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 128)
    before = set(main.settings.upload_dir.iterdir())

    client.post("/analyze", files=_upload_files())

    assert set(main.settings.upload_dir.iterdir()) == before


def test_inference_outage_returns_503(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("Docker inference failed: daemon not running")

    monkeypatch.setattr(main, "run_inference", _boom)

    resp = client.post("/analyze", files=_upload_files())

    assert resp.status_code == 503
    assert "Docker inference failed" in resp.json()["detail"]


def test_demo_image_rejects_path_traversal():
    resp = client.get("/demo/images/..%2F..%2F.env")
    assert resp.status_code in (400, 404)


def test_demo_image_does_not_serve_uploads(tmp_path):
    """Uploads are private: /demo/images must not expose them by filename."""
    planted = main.settings.upload_dir / "secret_pre_disaster.png"
    planted.write_bytes(_png_bytes())
    try:
        resp = client.get(f"/demo/images/{planted.name}")
        assert resp.status_code == 404
    finally:
        planted.unlink()


def test_report_pdf_round_trip():
    pair_id = client.get("/demo/pairs").json()[0]["id"]
    analysis = client.post("/analyze", data={"demo_pair_id": pair_id}).json()

    resp = client.post("/report/pdf", json={"analysis": analysis, "brief": "Deploy to zone 1."})

    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF")
    assert "attachment" in resp.headers["content-disposition"]


@pytest.mark.parametrize("endpoint", ["/analyze", "/brief", "/report/pdf"])
def test_expensive_endpoints_require_token_when_configured(monkeypatch, endpoint):
    monkeypatch.setattr(main.settings, "access_token", "s3cret")

    resp = client.post(endpoint)

    assert resp.status_code == 401
