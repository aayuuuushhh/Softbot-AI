"""Visual and GeoJSON output for the perception layer."""

from __future__ import annotations

import io

import cv2
import numpy as np

from core.schemas import DamageDetection, Severity

# Severity palette, RGB. Shared with the dashboard so the map and the overlay
# agree; keep in sync with web/lib/severity.ts.
SEVERITY_RGB: dict[str, tuple[int, int, int]] = {
    Severity.NONE.value: (56, 161, 105),       # green
    Severity.MINOR.value: (214, 158, 46),      # amber
    Severity.MAJOR.value: (221, 107, 32),      # orange
    Severity.DESTROYED.value: (197, 48, 48),   # red
}


def detections_to_geojson(detections: list[DamageDetection]) -> dict:
    """FeatureCollection of damage polygons, ready for MapLibre."""
    features = []
    for i, d in enumerate(detections):
        geometry = (
            d.geometry.model_dump()
            if d.geometry is not None
            else d.centroid.model_dump()
        )
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": geometry,
                "properties": {
                    "severity": str(d.severity),
                    "confidence": round(d.confidence, 3),
                    "source": d.source,
                    "area_m2": round(d.area_m2, 1) if d.area_m2 else None,
                    "model_version": d.model_version,
                    "color": "#{:02x}{:02x}{:02x}".format(*SEVERITY_RGB[str(d.severity)]),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def render_overlay(
    post_image: np.ndarray,
    detections: list[DamageDetection],
    alpha: float = 0.45,
) -> np.ndarray:
    """Tint each detected region by severity over the post-event image."""
    base = post_image.copy()
    tint = np.zeros_like(base)
    touched = np.zeros(base.shape[:2], bool)

    for d in detections:
        if not d.bbox_px:
            continue
        x0, y0, x1, y1 = d.bbox_px
        tint[y0:y1, x0:x1] = SEVERITY_RGB[str(d.severity)]
        touched[y0:y1, x0:x1] = True

    out = base.astype(np.float32)
    out[touched] = out[touched] * (1 - alpha) + tint[touched].astype(np.float32) * alpha
    out = out.astype(np.uint8)

    for d in detections:
        if not d.bbox_px:
            continue
        x0, y0, x1, y1 = d.bbox_px
        cv2.rectangle(out, (x0, y0), (x1, y1), SEVERITY_RGB[str(d.severity)], 1)
    return out


def render_triptych(
    pre_image: np.ndarray, post_image: np.ndarray, detections: list[DamageDetection]
) -> np.ndarray:
    """pre | post | overlay, side by side with labels — the field-report view."""
    overlay = render_overlay(post_image, detections)
    panels, labels = [pre_image, post_image, overlay], ["PRE", "POST", "DAMAGE"]

    height = min(p.shape[0] for p in panels)
    resized = []
    for panel, label in zip(panels, labels, strict=True):
        scale = height / panel.shape[0]
        img = cv2.resize(panel, (int(panel.shape[1] * scale), height))
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (18, 18, 18), -1)
        cv2.putText(img, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (245, 245, 245), 1, cv2.LINE_AA)
        resized.append(img)

    gap = np.full((height, 4, 3), 30, np.uint8)
    return np.hstack([resized[0], gap, resized[1], gap, resized[2]])


def to_png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return io.BytesIO(buf.tobytes()).getvalue()
