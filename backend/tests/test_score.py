"""Unit tests for the Response Priority Score."""

import pytest

from backend.app.config import priority_config
from backend.app.schemas import DamageClass
from backend.app.services.score import (
    WardInputs,
    accessibility_penalty,
    band_for,
    count_damaged,
    count_severe,
    raw_infrastructure_score,
    raw_damage_score,
    score_wards,
)


def counts(no=0, minor=0, major=0, destroyed=0):
    return {
        DamageClass.NO_DAMAGE: no,
        DamageClass.MINOR: minor,
        DamageClass.MAJOR: major,
        DamageClass.DESTROYED: destroyed,
    }


def test_weights_sum_to_one():
    assert sum(priority_config()["weights"].values()) == pytest.approx(1.0)


def test_undamaged_buildings_contribute_nothing():
    assert raw_damage_score(counts(no=500)) == 0.0


def test_damage_score_weights_severity():
    assert raw_damage_score(counts(destroyed=10)) > raw_damage_score(counts(minor=10))


def test_counts():
    c = counts(no=100, minor=20, major=10, destroyed=5)
    assert count_damaged(c) == 35
    assert count_severe(c) == 15  # major + destroyed


def test_infrastructure_score_ranks_by_facility_importance():
    assert raw_infrastructure_score([]) == 0.0
    assert raw_infrastructure_score(["hospital"]) > raw_infrastructure_score(["school"])
    assert raw_infrastructure_score(["hospital"] * 10) > raw_infrastructure_score(["hospital"])


def test_infrastructure_still_discriminates_when_every_ward_is_facility_rich():
    """Kathmandu wards routinely have 150+ facilities. Clipping the term at 1.0 made every
    ward identical on it; normalizing against the AOI maximum keeps it informative."""
    counts = counts_helper()
    scored = score_wards([
        WardInputs("few", "Few", counts, 1000, ["school"] * 20),
        WardInputs("many", "Many", counts, 1000, ["hospital"] * 60),
    ])
    assert scored["many"][1].infrastructure > scored["few"][1].infrastructure


def counts_helper():
    return counts(no=100, minor=20, major=10, destroyed=5)


def test_accessibility_penalty_bounds():
    assert accessibility_penalty(0.0, 0.0, 10.0) == 0.0
    assert accessibility_penalty(1.0, 10.0, 10.0) == 1.0


def test_bands():
    assert band_for(0.9)[0] == "Very High"
    assert band_for(0.6)[0] == "High"
    assert band_for(0.3)[0] == "Medium"
    assert band_for(0.0)[0] == "Low"


def test_worse_ward_outranks_better_ward():
    wards = [
        WardInputs("a", "Ward A", counts(no=180, minor=15, major=8, destroyed=2), 300, []),
        WardInputs("b", "Ward B", counts(no=80, minor=50, major=40, destroyed=35), 2400,
                   ["hospital", "school", "bridge"], road_disruption=0.5, distance_from_staging_km=5),
    ]
    scored = score_wards(wards)
    assert scored["b"][0] > scored["a"][0]
    assert band_for(scored["b"][0])[0] == "Very High"


def test_population_breaks_a_damage_tie():
    """Two wards, identical damage - the denser one must rank higher. This is the whole
    point of the Nepal-specific score."""
    same = counts(no=100, minor=20, major=10, destroyed=5)
    scored = score_wards([
        WardInputs("sparse", "Sparse", same, 200, []),
        WardInputs("dense", "Dense", same, 2000, []),
    ])
    assert scored["dense"][0] > scored["sparse"][0]


def test_empty_input():
    assert score_wards([]) == {}


# --- Heuristic fallback ---------------------------------------------------


def test_identical_chips_read_as_undamaged():
    import numpy as np

    from backend.app.services.classify import classify_heuristic

    chip = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
    assert classify_heuristic(chip, chip).damage_class == DamageClass.NO_DAMAGE


def test_flattened_building_reads_as_damaged():
    """A structured roof replaced by flat rubble must not come back 'no-damage'."""
    import numpy as np

    from backend.app.services.classify import classify_heuristic, heuristic_change_score

    rng = np.random.default_rng(1)
    roof = np.zeros((64, 64, 3), dtype="uint8")
    roof[::4, :, :] = 200  # regular roof lines
    rubble = (rng.random((64, 64, 3)) * 255).astype("uint8")

    assert heuristic_change_score(roof, rubble) > heuristic_change_score(roof, roof)
    assert classify_heuristic(roof, rubble).damage_class != DamageClass.NO_DAMAGE


def test_heuristic_never_sounds_as_sure_as_a_human():
    import numpy as np

    from backend.app.services.classify import classify_heuristic

    rng = np.random.default_rng(2)
    a = (rng.random((64, 64, 3)) * 255).astype("uint8")
    b = (rng.random((64, 64, 3)) * 255).astype("uint8")
    assert classify_heuristic(a, b).confidence < 1.0
