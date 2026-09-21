"""Deterministic zone scoring from xView2 damage masks (pixel values 0-4).

Building counts are derived via scipy connected-component labeling per
damage class within a region — an approximation of "how many distinct
buildings" carry that severity, since the mask is a per-pixel semantic
segmentation rather than a per-building instance segmentation.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from app.schemas import AnalysisResult, AnalysisSummary, BuildingCounts, DamageCounts, Zone

# xView2 class pixel values
CLASS_NAMES = {
    0: "background",
    1: "none",
    2: "minor",
    3: "major",
    4: "destroyed",
}

# Overlay colors (RGBA). These must stay in step with the dashboard's damage
# legend, the summary cards and the zone canvas — the overlay is read against
# that legend, so a colour that disagrees with it is simply wrong.
OVERLAY_COLORS = {
    0: (0, 0, 0, 0),
    1: (34, 197, 94, 120),    # green  - no damage  (tailwind green-500)
    2: (234, 179, 8, 140),    # yellow - minor      (tailwind yellow-500)
    3: (249, 115, 22, 160),   # orange - major      (tailwind orange-500)
    4: (239, 68, 68, 180),    # red    - destroyed  (tailwind red-500)
}

# Undamaged buildings carry no weight — they must not raise a zone's priority.
WEIGHTS = {2: 2.0, 3: 3.5, 4: 5.0}
_MAX_WEIGHT = WEIGHTS[4]

# 8-connectivity: two same-class pixels count as one building if they touch
# on an edge or a corner.
_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.uint8)


def load_confidence(path: Path, mask_shape: tuple[int, ...]) -> np.ndarray:
    """Load the per-pixel confidence map emitted alongside a pytorch mask."""
    confidence = np.load(path)
    if confidence.shape != mask_shape:
        raise ValueError(
            f"Confidence shape {confidence.shape} does not match mask shape {mask_shape}"
        )
    return confidence


def confidence_for_region(confidence: np.ndarray, mask_region: np.ndarray) -> float | None:
    """Mean predicted-class probability across a zone's building pixels.

    Background dominates most tiles, so averaging over the whole region would
    report the model's confidence that empty ground is empty.
    """
    building = mask_region > 0
    if not building.any():
        return None
    return round(float(confidence[building].mean()), 4)


def counts_for_region(mask: np.ndarray) -> DamageCounts:
    building = mask > 0
    if not building.any():
        return DamageCounts()
    vals, cnts = np.unique(mask[building], return_counts=True)
    mapping = dict(zip(vals.tolist(), cnts.tolist()))
    return DamageCounts(
        none=int(mapping.get(1, 0)),
        minor=int(mapping.get(2, 0)),
        major=int(mapping.get(3, 0)),
        destroyed=int(mapping.get(4, 0)),
    )


def building_counts_for_region(mask: np.ndarray) -> BuildingCounts:
    """Count connected components (individual buildings) per damage class."""
    counts: dict[int, int] = {}
    for cls in (1, 2, 3, 4):
        _, num_components = ndimage.label(mask == cls, structure=_CONNECTIVITY)
        counts[cls] = int(num_components)
    return BuildingCounts(
        none=counts[1],
        minor=counts[2],
        major=counts[3],
        destroyed=counts[4],
    )


def priority_score(counts: BuildingCounts) -> float:
    """Zone severity on a 0-100 scale.

    The denominator is the worst case the zone could have reached — every
    building destroyed — so a zone of intact structures scores 0, a totally
    destroyed zone scores 100, and scores stay comparable across zones
    holding different numbers of buildings.
    """
    total = counts.none + counts.minor + counts.major + counts.destroyed
    if total == 0:
        return 0.0
    weighted = (
        counts.minor * WEIGHTS[2]
        + counts.major * WEIGHTS[3]
        + counts.destroyed * WEIGHTS[4]
    )
    return round((weighted / (total * _MAX_WEIGHT)) * 100, 2)


def score_mask(
    mask_path: Path,
    grid_rows: int = 4,
    grid_cols: int = 4,
    confidence_path: Path | None = None,
) -> AnalysisResult:
    mask = load_mask(mask_path)
    h, w = mask.shape

    confidence = load_confidence(confidence_path, mask.shape) if confidence_path else None
    cell_h = max(1, h // grid_rows)
    cell_w = max(1, w // grid_cols)

    zones: list[Zone] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = row * cell_h
            x0 = col * cell_w
            y1 = h if row == grid_rows - 1 else (row + 1) * cell_h
            x1 = w if col == grid_cols - 1 else (col + 1) * cell_w
            region = mask[y0:y1, x0:x1]
            counts = counts_for_region(region)
            total_building_px = counts.none + counts.minor + counts.major + counts.destroyed
            if total_building_px == 0:
                continue
            building_counts = building_counts_for_region(region)
            zone_confidence = (
                confidence_for_region(confidence[y0:y1, x0:x1], region)
                if confidence is not None
                else None
            )
            zones.append(
                Zone(
                    rank=0,
                    bbox=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                    damage_counts=counts,
                    building_counts=building_counts,
                    priority_score=priority_score(building_counts),
                    confidence=zone_confidence,
                )
            )

    zones.sort(key=lambda z: z.priority_score, reverse=True)
    for i, zone in enumerate(zones, start=1):
        zone.rank = i

    all_counts = counts_for_region(mask)
    total_building_px = (
        all_counts.none + all_counts.minor + all_counts.major + all_counts.destroyed
    )

    all_building_counts = building_counts_for_region(mask)
    total_buildings = (
        all_building_counts.none
        + all_building_counts.minor
        + all_building_counts.major
        + all_building_counts.destroyed
    )

    summary = AnalysisSummary(
        total_building_pixels=total_building_px,
        total_buildings=total_buildings,
        destroyed_pct=round(all_building_counts.destroyed / total_buildings * 100, 2) if total_buildings else 0.0,
        major_pct=round(all_building_counts.major / total_buildings * 100, 2) if total_buildings else 0.0,
        minor_pct=round(all_building_counts.minor / total_buildings * 100, 2) if total_buildings else 0.0,
    )

    overlay = _build_overlay(mask)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return AnalysisResult(
        zones=zones,
        summary=summary,
        # Never expose server filesystem paths in API responses.
        mask_path=None,
        mask_base64=mask_b64,
        inference_mode="scoring",
        geo_mode="image",
        image_size=[int(w), int(h)],
    )


def _build_overlay(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in OVERLAY_COLORS.items():
        if cls == 0:
            continue
        rgba[mask == cls] = color
    return Image.fromarray(rgba, mode="RGBA")
