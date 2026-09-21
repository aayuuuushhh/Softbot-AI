"""CPU tests for post-softmax confidence extraction in infer_pair."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

INFER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFER_DIR))

from infer_pair import probs_to_mask_and_confidence  # noqa: E402


def test_probs_to_mask_and_confidence_matches_max_along_class_axis() -> None:
    arr = np.zeros((3, 4, 4), dtype=np.float32)
    arr[0, 0, 0] = 0.1
    arr[1, 0, 0] = 0.2
    arr[2, 0, 0] = 0.7
    arr[0, 1, 1] = 0.8
    arr[1, 1, 1] = 0.15
    arr[2, 1, 1] = 0.05
    arr[0, 2, 3] = 0.33
    arr[1, 2, 3] = 0.33
    arr[2, 2, 3] = 0.34

    mask, confidence = probs_to_mask_and_confidence(arr)

    assert mask[0, 0] == 2
    assert confidence is not None
    assert confidence[0, 0] == pytest.approx(0.7)
    assert confidence[1, 1] == pytest.approx(0.8)
    assert confidence[2, 3] == pytest.approx(0.34)
    assert np.allclose(confidence, arr.max(axis=0))


def test_probs_to_mask_and_confidence_2d_returns_no_confidence() -> None:
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    mask, confidence = probs_to_mask_and_confidence(arr)
    assert confidence is None
    assert mask.shape == (2, 2)


def test_probs_values_remain_in_unit_interval_without_double_softmax() -> None:
    arr = np.array(
        [
            [[0.2, 0.5], [0.1, 0.25]],
            [[0.3, 0.3], [0.6, 0.25]],
            [[0.5, 0.2], [0.3, 0.5]],
        ],
        dtype=np.float32,
    )
    _, confidence = probs_to_mask_and_confidence(arr)
    assert confidence is not None
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)
    assert np.allclose(arr.sum(axis=0), 1.0)
