from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import base64
import io

from app.schemas import BuildingCounts
from app.services.scoring import (
    OVERLAY_COLORS,
    building_counts_for_region,
    counts_for_region,
    priority_score,
    score_mask,
)

# The dashboard's damage legend, summary cards and zone canvas all key off these
# exact Tailwind colours. The overlay is read against that legend, so a mismatch
# means the picture contradicts the key beside it — minor damage once painted
# blue while the legend called it yellow.
LEGEND_RGB = {
    1: (34, 197, 94),    # green-500  - no damage
    2: (234, 179, 8),    # yellow-500 - minor
    3: (249, 115, 22),   # orange-500 - major
    4: (239, 68, 68),    # red-500    - destroyed
}


def test_counts_for_region_basic():
    mask = np.array(
        [
            [0, 1, 1],
            [2, 2, 3],
            [4, 4, 4],
        ],
        dtype=np.uint8,
    )
    counts = counts_for_region(mask)
    assert counts.none == 2
    assert counts.minor == 2
    assert counts.major == 1
    assert counts.destroyed == 3


def test_building_counts_for_region_single_component():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:4, 1:4] = 2  # one contiguous "minor" blob
    counts = building_counts_for_region(mask)
    assert counts.minor == 1
    assert counts.none == 0
    assert counts.major == 0
    assert counts.destroyed == 0


def test_building_counts_for_region_multiple_components():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:2, 0:2] = 3  # major blob #1
    mask[7:9, 7:9] = 3  # major blob #2, disconnected
    counts = building_counts_for_region(mask)
    assert counts.major == 2


def test_priority_score_weighting():
    mostly_destroyed = BuildingCounts(none=0, minor=0, major=0, destroyed=10)
    mostly_undamaged = BuildingCounts(none=10, minor=0, major=0, destroyed=0)
    assert priority_score(mostly_destroyed) > priority_score(mostly_undamaged)


def test_priority_score_empty():
    assert priority_score(BuildingCounts()) == 0.0


def test_priority_score_spans_zero_to_one_hundred():
    assert priority_score(BuildingCounts(none=10)) == 0.0
    assert priority_score(BuildingCounts(destroyed=10)) == 100.0


def test_priority_score_ignores_undamaged_buildings():
    """Intact structures must dilute, never inflate, a zone's priority."""
    damage_only = BuildingCounts(major=2)
    with_intact_neighbours = BuildingCounts(none=8, major=2)
    assert priority_score(with_intact_neighbours) < priority_score(damage_only)


@pytest.mark.parametrize(
    "counts",
    [
        BuildingCounts(),
        BuildingCounts(none=5),
        BuildingCounts(minor=3, major=1),
        BuildingCounts(none=1, minor=1, major=1, destroyed=1),
        BuildingCounts(destroyed=7),
    ],
)
def test_priority_score_always_in_range(counts):
    assert 0.0 <= priority_score(counts) <= 100.0


def test_score_mask_end_to_end(tmp_path):
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:10, 2:10] = 2  # minor buildings, top-left grid cell
    mask[30:38, 30:38] = 4  # destroyed buildings, bottom-right grid cell

    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)

    result = score_mask(mask_path, grid_rows=4, grid_cols=4)

    assert result.summary.total_buildings == 2
    assert result.summary.destroyed_pct == pytest.approx(50.0)

    ranks = [zone.rank for zone in result.zones]
    assert ranks == sorted(ranks)
    assert result.zones[0].priority_score >= result.zones[-1].priority_score
    assert result.zones[0].building_counts.destroyed == 1
    assert result.mask_path is None


def test_overlay_colors_match_the_dashboard_legend():
    for cls, expected_rgb in LEGEND_RGB.items():
        assert OVERLAY_COLORS[cls][:3] == expected_rgb, (
            f"class {cls} is painted {OVERLAY_COLORS[cls][:3]} but the legend shows "
            f"{expected_rgb}"
        )


def test_overlay_paints_each_class_with_its_legend_color(tmp_path):
    """The rendered overlay, not just the table, must agree with the legend."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 2
    mask[2, 2] = 3
    mask[3, 3] = 4

    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)

    result = score_mask(mask_path, grid_rows=2, grid_cols=2)

    overlay = np.array(
        Image.open(io.BytesIO(base64.b64decode(result.mask_base64))).convert("RGBA")
    )

    assert tuple(overlay[0, 0][:3]) == LEGEND_RGB[1]
    assert tuple(overlay[1, 1][:3]) == LEGEND_RGB[2]
    assert tuple(overlay[2, 2][:3]) == LEGEND_RGB[3]
    assert tuple(overlay[3, 3][:3]) == LEGEND_RGB[4]

    # Background stays fully transparent so the after-image shows through.
    assert overlay[0, 3][3] == 0
