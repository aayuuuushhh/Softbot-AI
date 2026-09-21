"""Evaluation metrics shared by the training and evaluation scripts.

Never bare accuracy: on xBD a model that paints everything `background`
scores >95% pixel accuracy and finds nothing.

Two granularities:

* pixel  - per-class F1 over the segmentation mask (what training optimises)
* building - each labelled footprint takes the majority predicted class inside
  it, then per-class F1 over buildings. This is what xView2 scored and what the
  CLAUDE.md gate refers to: F1 > 0.80 on `destroyed`.
"""

from __future__ import annotations

import numpy as np

from core.vision.models import CHANGE_CLASSES

DESTROYED = CHANGE_CLASSES.index("destroyed")
DESTROYED_F1_GATE = 0.80


def confusion(
    pred: np.ndarray, target: np.ndarray, n_classes: int, ignore_index: int = 255
) -> np.ndarray:
    """(n, n) confusion matrix, rows = truth, cols = prediction."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    keep = target != ignore_index
    pred, target = pred[keep], target[keep]
    idx = target.astype(np.int64) * n_classes + pred.astype(np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    return f1


def report(cm: np.ndarray, names: list[str]) -> dict:
    f1 = per_class_f1(cm)
    support = cm.sum(axis=1)
    per_class = {
        name: {
            "f1": None if np.isnan(f1[i]) else round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, name in enumerate(names)
    }
    present = [f for f in f1 if not np.isnan(f)]
    return {
        "per_class": per_class,
        "macro_f1": round(float(np.mean(present)), 4) if present else None,
        "confusion": cm.tolist(),
    }


def building_votes(
    pred_mask: np.ndarray, polygons: list[tuple[np.ndarray, int]]
) -> tuple[list[int], list[int]]:
    """Majority predicted class inside each labelled footprint.

    Background pixels do not vote: a building the model partly missed is judged
    on the pixels it did classify. A footprint with no non-background
    prediction counts as `none` (detected nothing = claimed no damage).
    """
    import cv2

    from core.vision.datasets import IGNORE_INDEX

    none_cls = CHANGE_CLASSES.index("none")
    preds, truths = [], []
    for coords, cls in polygons:
        if cls == IGNORE_INDEX:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        if w <= 0 or h <= 0:
            continue
        local = np.zeros((h, w), np.uint8)
        cv2.fillPoly(local, [coords - np.array([x, y], np.int32)], 1)
        region = pred_mask[y : y + h, x : x + w]
        inside = region[local[: region.shape[0], : region.shape[1]] > 0]
        inside = inside[inside > 0]
        vote = int(np.bincount(inside).argmax()) if inside.size else none_cls
        preds.append(vote)
        truths.append(int(cls))
    return preds, truths


def passes_gate(metrics: dict) -> bool:
    f1 = metrics["per_class"]["destroyed"]["f1"]
    return f1 is not None and f1 > DESTROYED_F1_GATE
