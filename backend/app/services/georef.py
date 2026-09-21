"""Georeferencing: maps zone pixel centroids to lat/lng.

xBD label files carry every building twice: once as a WKT polygon in image
pixel space ("xy") and once in geographic space ("lng_lat"), keyed by the
same building uid. Treating matched building centroids as control points,
we fit a least-squares affine transform (pixel -> lng/lat) and apply it to
zone bounding-box centroids.

Uploaded imagery has no xBD label file, so it never gets geo enrichment —
callers should check for a `None` result and surface NO_GEO_MESSAGE.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from app.config import settings

NO_GEO_MESSAGE = "Geographic coordinates available for demo pairs with xBD metadata only."

MIN_CORRESPONDENCES = 3

_POINT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def _parse_wkt_polygon(wkt: str) -> np.ndarray:
    """Parse a WKT POLYGON string into an (N, 2) array of vertices."""
    points = [(float(x), float(y)) for x, y in _POINT_RE.findall(wkt)]
    if len(points) < 3:
        raise ValueError(f"WKT polygon has fewer than 3 vertices: {wkt!r}")
    return np.array(points, dtype=np.float64)


def _polygon_centroid(vertices: np.ndarray) -> tuple[float, float]:
    """Shoelace-formula centroid; falls back to a vertex average for
    degenerate (near-zero-area) polygons."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0

    if abs(area) < 1e-9:
        return float(x.mean()), float(y.mean())

    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def _label_path_candidates(pair_id: str) -> list[Path]:
    clean_id = Path(pair_id).name
    labels_dir = settings.demo_data_dir / "labels"
    return [
        labels_dir / f"{clean_id}_post_disaster.json",
        labels_dir / f"{clean_id}_pre_disaster.json",
    ]


def _load_correspondences(pair_id: str) -> list[tuple[float, float, float, float]]:
    """Return [(pixel_x, pixel_y, lng, lat), ...] for buildings present in
    both the xy and lng_lat feature sets of a demo pair's label file."""
    for label_path in _label_path_candidates(pair_id):
        if not label_path.exists():
            continue

        try:
            data = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        features = data.get("features", {})
        xy_features = features.get("xy", [])
        lnglat_features = features.get("lng_lat", [])

        lnglat_by_uid = {
            f["properties"]["uid"]: f["wkt"]
            for f in lnglat_features
            if "uid" in f.get("properties", {})
        }

        correspondences: list[tuple[float, float, float, float]] = []
        for f in xy_features:
            uid = f.get("properties", {}).get("uid")
            lnglat_wkt = lnglat_by_uid.get(uid)
            if not lnglat_wkt:
                continue

            try:
                px, py = _polygon_centroid(_parse_wkt_polygon(f["wkt"]))
                lng, lat = _polygon_centroid(_parse_wkt_polygon(lnglat_wkt))
            except (ValueError, KeyError):
                continue

            correspondences.append((px, py, lng, lat))

        if correspondences:
            return correspondences

    return []


class AffineGeoTransform:
    """Least-squares affine fit: pixel (x, y) -> (lat, lng)."""

    def __init__(self, coeffs_lng: np.ndarray, coeffs_lat: np.ndarray):
        self._coeffs_lng = coeffs_lng
        self._coeffs_lat = coeffs_lat

    def pixel_to_latlng(self, x: float, y: float) -> tuple[float, float]:
        row = np.array([x, y, 1.0])
        lng = float(row @ self._coeffs_lng)
        lat = float(row @ self._coeffs_lat)
        return lat, lng


def fit_geo_transform(pair_id: str) -> AffineGeoTransform | None:
    """Fit a pixel -> lat/lng affine transform for a demo pair from xBD
    building label correspondences.

    Returns None when fewer than MIN_CORRESPONDENCES buildings are
    available — e.g. uploads, or demo pairs without a labels/ file.
    """
    correspondences = _load_correspondences(pair_id)

    if len(correspondences) < MIN_CORRESPONDENCES:
        return None

    points = np.array(correspondences, dtype=np.float64)
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    lng_target = points[:, 2]
    lat_target = points[:, 3]

    coeffs_lng, *_ = np.linalg.lstsq(design, lng_target, rcond=None)
    coeffs_lat, *_ = np.linalg.lstsq(design, lat_target, rcond=None)

    return AffineGeoTransform(coeffs_lng, coeffs_lat)
