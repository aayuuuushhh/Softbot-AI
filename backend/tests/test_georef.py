from __future__ import annotations

import json

import pytest

from app.services import georef

TRUE_COEFFS_LNG = (0.0001, 0.00005, -90.5)  # a*x + b*y + c
TRUE_COEFFS_LAT = (-0.00003, 0.00012, 14.6)


def _true_transform(x: float, y: float) -> tuple[float, float]:
    a, b, c = TRUE_COEFFS_LNG
    d, e, f = TRUE_COEFFS_LAT
    return a * x + b * y + c, d * x + e * y + f


def _square_corners(cx: float, cy: float, half: float = 2.0) -> list[tuple[float, float]]:
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
        (cx - half, cy - half),
    ]


def _wkt(points: list[tuple[float, float]]) -> str:
    return "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in points) + "))"


def _write_synthetic_label(labels_dir, pair_id: str, centers: list[tuple[float, float]]) -> None:
    xy_features = []
    lnglat_features = []

    for i, (cx, cy) in enumerate(centers):
        uid = f"bldg-{i}"
        xy_corners = _square_corners(cx, cy)
        lnglat_corners = [_true_transform(x, y) for x, y in xy_corners]

        xy_features.append(
            {"properties": {"feature_type": "building", "uid": uid}, "wkt": _wkt(xy_corners)}
        )
        lnglat_features.append(
            {"properties": {"feature_type": "building", "uid": uid}, "wkt": _wkt(lnglat_corners)}
        )

    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{pair_id}_post_disaster.json"
    label_path.write_text(
        json.dumps({"features": {"xy": xy_features, "lng_lat": lnglat_features}}),
        encoding="utf-8",
    )


def test_parse_wkt_polygon():
    verts = georef._parse_wkt_polygon("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))")
    assert verts.shape == (5, 2)
    assert verts[0].tolist() == [0.0, 0.0]
    assert verts[2].tolist() == [4.0, 4.0]


def test_polygon_centroid_square():
    verts = georef._parse_wkt_polygon("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))")
    cx, cy = georef._polygon_centroid(verts)
    assert cx == pytest.approx(2.0)
    assert cy == pytest.approx(2.0)


def test_fit_geo_transform_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(georef.settings, "demo_data_dir", tmp_path)

    centers = [(10, 10), (500, 10), (10, 500), (500, 500), (250, 250)]
    _write_synthetic_label(tmp_path / "labels", "synthetic_pair", centers)

    transform = georef.fit_geo_transform("synthetic_pair")
    assert transform is not None

    lat, lng = transform.pixel_to_latlng(250, 250)
    expected_lng, expected_lat = _true_transform(250, 250)
    assert lat == pytest.approx(expected_lat, abs=1e-4)
    assert lng == pytest.approx(expected_lng, abs=1e-4)


def test_fit_geo_transform_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(georef.settings, "demo_data_dir", tmp_path)
    assert georef.fit_geo_transform("nonexistent_pair") is None
