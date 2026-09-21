"""S1 (overhead) and S2 (ground) inference wrappers.

Two interchangeable backends per stage:

* ``stub``  - deterministic OpenCV. Needs no weights, always available.
* ``torch`` - the nets in ``models.py``, batched on the GPU under AMP.

`torch` falls back to `stub` with a loud warning when its checkpoint is
missing, so a fresh clone runs end-to-end and a missing `.pt` never takes down
the API.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from core.config import get_settings
from core.schemas import DamageDetection, Point, Polygon, Severity
from core.vision.preprocess import (
    Raster,
    align_pair,
    load_ground_image,
    normalize,
    pixel_area_m2,
    pixel_to_lonlat,
    tile_windows,
)

log = logging.getLogger(__name__)

# Minimum connected-component size, in pixels, to count as a structure rather
# than sensor noise.
MIN_COMPONENT_PX = 12


def score_to_severity(score: float) -> Severity:
    """Map a normalised change score in [0, 1] onto the damage scale."""
    if score < 0.15:
        return Severity.NONE
    if score < 0.35:
        return Severity.MINOR
    if score < 0.60:
        return Severity.MAJOR
    return Severity.DESTROYED


# --------------------------------------------------------------------------
# Satellite (S1)
# --------------------------------------------------------------------------


@runtime_checkable
class SatelliteBackend(Protocol):
    name: str

    def detect(self, pre: Raster, post: Raster) -> list[DamageDetection]: ...


def _components_to_detections(
    mask: np.ndarray,
    score_map: np.ndarray,
    raster: Raster,
    model_version: str,
) -> list[DamageDetection]:
    """Turn a binary change mask into geotagged, severity-scored detections."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    px_area = pixel_area_m2(raster.transform, raster.crs)
    detections: list[DamageDetection] = []

    for i in range(1, count):  # 0 is background
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < MIN_COMPONENT_PX:
            continue

        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[i, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[i, cv2.CC_STAT_HEIGHT])

        component = labels[y0:y1, x0:x1] == i
        score = float(score_map[y0:y1, x0:x1][component].mean())
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        lon, lat = pixel_to_lonlat(raster.transform, raster.crs, cx, cy)
        corners = [
            pixel_to_lonlat(raster.transform, raster.crs, x, y)
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))
        ]

        detections.append(
            DamageDetection(
                bbox_px=[x0, y0, x1, y1],
                geometry=Polygon(coordinates=[[list(c) for c in corners]]),
                centroid=Point(coordinates=[lon, lat]),
                severity=score_to_severity(score),
                # A detection is never fully certain from overhead alone;
                # fusion will raise this when a ground photo agrees.
                confidence=float(np.clip(0.35 + 0.6 * score, 0.0, 0.95)),
                source="satellite",
                area_m2=area_px * px_area,
                model_version=model_version,
            )
        )

    detections.sort(key=lambda d: d.area_m2 or 0.0, reverse=True)
    return detections


class StubChangeDetector:
    """Deterministic change detection: align, difference in Lab, threshold.

    CIELAB rather than RGB because its L channel isolates the brightness
    collapse of a roof becoming rubble, while a*/b* catch the colour shift —
    both of which survive the illumination differences that wreck a plain RGB
    difference.
    """

    name = "stub"

    def detect(self, pre: Raster, post: Raster) -> list[DamageDetection]:
        aligned, shift = align_pair(pre.image, post.image)
        log.info("stub S1: corrected shift dx=%.2f dy=%.2f", *shift)

        pre_lab = cv2.cvtColor(pre.image, cv2.COLOR_RGB2LAB).astype(np.float32)
        post_lab = cv2.cvtColor(aligned, cv2.COLOR_RGB2LAB).astype(np.float32)

        delta = np.abs(pre_lab - post_lab)
        # Weight luminance highest: structural collapse shows up there first.
        magnitude = 0.6 * delta[..., 0] + 0.2 * delta[..., 1] + 0.2 * delta[..., 2]
        magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)

        norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)

        _thresh, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Brightness collapse (roof -> rubble) is the strongest single cue, so
        # fold it into the score rather than relying on magnitude alone.
        brightness_drop = np.clip(
            (pre_lab[..., 0] - post_lab[..., 0]) / 100.0, 0.0, 1.0
        )
        score_map = np.clip(
            0.6 * (magnitude / max(magnitude.max(), 1e-6)) + 0.4 * brightness_drop, 0.0, 1.0
        )

        return _components_to_detections(mask > 0, score_map, pre, self.name)


class TorchChangeDetector:
    """SiameseChangeNet over overlapping tiles, batched on the GPU under AMP.

    Tile logits are accumulated and averaged where windows overlap, so a
    structure split across a tile boundary is resolved once rather than twice.
    """

    name = "torch"

    def __init__(self) -> None:
        import torch

        from core.vision.models import CHANGE_CLASSES, SiameseChangeNet, load_checkpoint

        settings = get_settings()
        self.device = settings.resolved_device()
        self.tile_size = settings.tile_size
        self.overlap = settings.tile_overlap
        self.batch_size = settings.batch_size
        self.classes = CHANGE_CLASSES
        self._torch = torch

        self.model = SiameseChangeNet(pretrained=True)
        self.has_weights = load_checkpoint(
            self.model, settings.satellite_weights, self.device
        )
        self.model.eval().to(self.device)

    def detect(self, pre: Raster, post: Raster) -> list[DamageDetection]:
        torch = self._torch
        aligned, shift = align_pair(pre.image, post.image)
        log.info("torch S1: corrected shift dx=%.2f dy=%.2f", *shift)

        h, w = pre.image.shape[:2]
        windows = tile_windows(h, w, self.tile_size, self.overlap)
        n_classes = len(self.classes)
        logit_sum = np.zeros((n_classes, h, w), np.float32)
        counts = np.zeros((h, w), np.float32)

        use_amp = self.device == "cuda"
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            pre_stack = np.stack([normalize(pre.image[y0:y1, x0:x1]) for x0, y0, x1, y1 in batch])
            post_stack = np.stack([normalize(aligned[y0:y1, x0:x1]) for x0, y0, x1, y1 in batch])

            pre_t = torch.from_numpy(pre_stack).to(self.device)
            post_t = torch.from_numpy(post_stack).to(self.device)

            with torch.inference_mode(), torch.autocast("cuda", enabled=use_amp):
                logits = self.model(pre_t, post_t).float().cpu().numpy()

            for (x0, y0, x1, y1), tile_logits in zip(batch, logits, strict=True):
                logit_sum[:, y0:y1, x0:x1] += tile_logits[:, : y1 - y0, : x1 - x0]
                counts[y0:y1, x0:x1] += 1.0

        logit_sum /= np.maximum(counts, 1.0)[None, ...]

        exp = np.exp(logit_sum - logit_sum.max(axis=0, keepdims=True))
        probs = exp / exp.sum(axis=0, keepdims=True)
        predicted = probs.argmax(axis=0)

        # Classes 2..4 are minor/major/destroyed; 0/1 are background/no-damage.
        damaged = predicted >= 2
        severity_weight = np.array([0.0, 0.0, 0.3, 0.6, 1.0], np.float32)
        score_map = np.tensordot(severity_weight, probs, axes=(0, 0))

        version = "siamese-resnet18" + ("" if self.has_weights else "-untrained")
        return _components_to_detections(damaged, score_map, pre, version)


# --------------------------------------------------------------------------
# Ground (S2)
# --------------------------------------------------------------------------


@runtime_checkable
class GroundBackend(Protocol):
    name: str

    def classify(self, image_path: str) -> tuple[Severity, float]: ...


class StubGroundClassifier:
    """Heuristic structural-integrity score for a field photo.

    Intact buildings present long, straight, high-contrast edges (walls, roof
    lines, window rows). Collapse replaces that with high-frequency, randomly
    oriented texture and a duller, greyer palette. We score the ratio of
    coherent linear structure to overall edge energy.
    """

    name = "stub"

    def classify(self, image_path: str) -> tuple[Severity, float]:
        image = load_ground_image(image_path, size=384)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(edges.mean() / 255.0)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=55, minLineLength=45, maxLineGap=12
        )
        n_lines = 0 if lines is None else len(lines)

        # Rubble is desaturated and has high local variance.
        saturation = float(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1].mean() / 255.0)
        local_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        structure = min(n_lines / 45.0, 1.0)           # 1.0 = very intact
        chaos = min(edge_density / 0.14, 1.0)          # 1.0 = very rubbled
        texture = min(local_var / 2600.0, 1.0)
        dullness = 1.0 - min(saturation / 0.34, 1.0)

        damage = float(
            np.clip(0.42 * chaos + 0.28 * texture + 0.20 * dullness - 0.38 * structure + 0.22,
                    0.0, 1.0)
        )
        # Heuristics deserve modest confidence; the torch backend and fusion
        # are what should raise it.
        confidence = float(np.clip(0.40 + 0.25 * abs(damage - 0.5) * 2.0, 0.0, 0.72))
        return score_to_severity(damage), confidence


class TorchGroundClassifier:
    """GroundDamageNet over a single field photo."""

    name = "torch"

    def __init__(self) -> None:
        import torch

        from core.vision.models import GROUND_CLASSES, GroundDamageNet, load_checkpoint

        settings = get_settings()
        self.device = settings.resolved_device()
        self.classes = GROUND_CLASSES
        self._torch = torch

        self.model = GroundDamageNet(pretrained=True)
        self.has_weights = load_checkpoint(self.model, settings.ground_weights, self.device)
        self.model.eval().to(self.device)

    def classify(self, image_path: str) -> tuple[Severity, float]:
        torch = self._torch
        array = normalize(load_ground_image(image_path, size=224))
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)

        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device == "cuda"):
            probs = torch.softmax(self.model(tensor).float(), dim=1)[0].cpu().numpy()

        index = int(probs.argmax())
        confidence = float(probs[index])
        if not self.has_weights:
            # An untrained head produces near-uniform softmax; reporting that
            # as confidence would mislead fusion.
            confidence = min(confidence, 0.30)
        return Severity(self.classes[index]), confidence


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

_sat_cache: dict[str, SatelliteBackend] = {}
_ground_cache: dict[str, GroundBackend] = {}


def get_satellite_backend(name: str | None = None) -> SatelliteBackend:
    settings = get_settings()
    name = name or settings.sat_backend

    if name == "torch":
        if not settings.satellite_weights.exists():
            log.warning(
                "UDDHAR_SAT_BACKEND=torch but %s is missing - falling back to "
                "the stub detector. Train with scripts/train_satellite.py.",
                settings.satellite_weights,
            )
            name = "stub"

    if name not in _sat_cache:
        _sat_cache[name] = TorchChangeDetector() if name == "torch" else StubChangeDetector()
    return _sat_cache[name]


def get_ground_backend(name: str | None = None) -> GroundBackend:
    settings = get_settings()
    name = name or settings.ground_backend

    if name == "torch" and not settings.ground_weights.exists():
        log.warning(
            "UDDHAR_GROUND_BACKEND=torch but %s is missing - falling back to "
            "the stub classifier. Train with scripts/train_ground.py.",
            settings.ground_weights,
        )
        name = "stub"

    if name not in _ground_cache:
        _ground_cache[name] = (
            TorchGroundClassifier() if name == "torch" else StubGroundClassifier()
        )
    return _ground_cache[name]


def reset_backends() -> None:
    """Drop cached backends so a config change takes effect (used by tests)."""
    _sat_cache.clear()
    _ground_cache.clear()
