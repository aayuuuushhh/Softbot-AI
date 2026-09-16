"""Damage classification: the trained siamese model, and the heuristic that covers for it.

The heuristic exists because a hackathon model can fail to converge at 3am. It is mediocre and
honest, and it guarantees the map is never empty at demo time - the mode is reported all the way
to the dashboard so nobody mistakes one for the other.

See PROJECT_PLAN.md section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import DamageClass

# Structural change thresholds, tuned on chip pairs. Above each cut-off, a building moves up
# a severity class. Deliberately conservative: over-calling damage sends teams to the wrong ward.
HEURISTIC_THRESHOLDS = {
    DamageClass.DESTROYED: 0.55,
    DamageClass.MAJOR: 0.35,
    DamageClass.MINOR: 0.18,
}


@dataclass
class Prediction:
    damage_class: DamageClass
    confidence: float


def heuristic_change_score(pre_chip: np.ndarray, post_chip: np.ndarray) -> float:
    """How much did this building change? 0 = identical, 1 = unrecognisable.

    Three signals, because none alone survives real imagery:
      - brightness delta  : rubble and exposed concrete are brighter than intact roofs
      - texture delta     : a collapsed building loses its regular roof edges
      - structural drop   : simplified SSIM over the chip
    """
    pre = _to_gray(pre_chip)
    post = _to_gray(post_chip)

    brightness = float(np.abs(post.mean() - pre.mean()))
    texture = float(abs(_edge_density(post) - _edge_density(pre)))
    structural = 1.0 - _similarity(pre, post)

    score = 0.25 * brightness + 0.35 * texture + 0.40 * structural
    return float(np.clip(score, 0.0, 1.0))


def classify_heuristic(pre_chip: np.ndarray, post_chip: np.ndarray) -> Prediction:
    score = heuristic_change_score(pre_chip, post_chip)
    for damage_class, threshold in HEURISTIC_THRESHOLDS.items():
        if score >= threshold:
            # Confidence rises with distance past the threshold, and is capped well below 1.0 -
            # the heuristic should never look as sure as a human reviewer.
            margin = (score - threshold) / max(1e-6, 1.0 - threshold)
            return Prediction(damage_class, round(float(min(0.75, 0.45 + 0.3 * margin)), 2))
    return Prediction(DamageClass.NO_DAMAGE, round(float(min(0.8, 0.55 + (0.18 - score))), 2))


def _to_gray(chip: np.ndarray) -> np.ndarray:
    array = chip.astype(np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    if array.ndim == 3:
        array = array[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return array


def _edge_density(gray: np.ndarray) -> float:
    """Mean gradient magnitude - a proxy for how much roof structure survives."""
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean()
    dx = np.abs(np.diff(gray, axis=1)).mean()
    return float((dx + dy) / 2.0)


def _similarity(pre: np.ndarray, post: np.ndarray) -> float:
    """Global SSIM over the chip. 1.0 = identical."""
    mu_pre, mu_post = pre.mean(), post.mean()
    var_pre, var_post = pre.var(), post.var()
    covariance = ((pre - mu_pre) * (post - mu_post)).mean()

    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_pre * mu_post + c1) * (2 * covariance + c2)
    denominator = (mu_pre**2 + mu_post**2 + c1) * (var_pre + var_post + c2)
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0
