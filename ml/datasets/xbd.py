"""xBD building chips.

Pairs each building polygon with its pre- and post-disaster crop. Chip extraction runs once
offline (`extract_chips`) and writes a manifest; training then just reads PNGs, which keeps the
GPU fed without re-opening 1024x1024 rasters every batch.

See PROJECT_PLAN.md section 2.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..models.siamese import DAMAGE_CLASSES

CHIP_SIZE = 128
PADDING_PX = 10

# Chip paths are stored in the manifest relative to this, so a manifest extracted on one
# machine still resolves after the chips are copied to whatever box does the training.
REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_TO_INDEX = {name: i for i, name in enumerate(DAMAGE_CLASSES)}


@dataclass
class ChipRecord:
    chip_id: str
    pre_path: str
    post_path: str
    damage_class: str
    disaster: str


# --- Offline extraction ---------------------------------------------------


def _polygon_bounds(wkt: str) -> tuple[float, float, float, float]:
    """Minimal WKT POLYGON parser - avoids a shapely dependency in the ML path."""
    body = wkt[wkt.index("((") + 2 : wkt.index("))")]
    xs, ys = [], []
    for point in body.split(","):
        x, y = point.strip().split(" ")[:2]
        xs.append(float(x))
        ys.append(float(y))
    return min(xs), min(ys), max(xs), max(ys)


def extract_chips(
    xbd_root: Path,
    out_dir: Path,
    disasters: list[str] | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Crop every labelled building from its pre/post tile pair.

    Returns the path to the manifest CSV. Labels come from the *post* JSON, which is the only
    one carrying `subtype`.

    `disasters` filters by event name, which is how the earthquake hold-out is built: the
    training manifest excludes mexico-earthquake so it can be scored as unseen seismic terrain.
    """
    images_dir, labels_dir = xbd_root / "images", xbd_root / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pre").mkdir(exist_ok=True)
    (out_dir / "post").mkdir(exist_ok=True)

    if manifest_path is None:
        manifest_path = out_dir.parent / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[ChipRecord] = []

    for post_label in sorted(labels_dir.glob("*_post_disaster.json")):
        stem = post_label.stem.replace("_post_disaster", "")
        disaster = stem.rsplit("_", 1)[0]
        if disasters and not any(d in disaster for d in disasters):
            continue

        pre_image_path = _find_image(images_dir, f"{stem}_pre_disaster")
        post_image_path = _find_image(images_dir, f"{stem}_post_disaster")
        if pre_image_path is None or post_image_path is None:
            continue

        pre_image = Image.open(pre_image_path).convert("RGB")
        post_image = Image.open(post_image_path).convert("RGB")
        label_data = json.loads(post_label.read_text())

        for index, feature in enumerate(label_data.get("features", {}).get("xy", [])):
            damage = feature.get("properties", {}).get("subtype")
            if damage not in CLASS_TO_INDEX:
                continue  # skips `un-classified`

            min_x, min_y, max_x, max_y = _polygon_bounds(feature["wkt"])
            box = (
                max(0, int(min_x) - PADDING_PX),
                max(0, int(min_y) - PADDING_PX),
                min(pre_image.width, int(max_x) + PADDING_PX),
                min(pre_image.height, int(max_y) + PADDING_PX),
            )
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue  # degenerate footprint

            chip_id = f"{stem}_{index:04d}"
            pre_out = out_dir / "pre" / f"{chip_id}.png"
            post_out = out_dir / "post" / f"{chip_id}.png"
            pre_image.crop(box).resize((CHIP_SIZE, CHIP_SIZE)).save(pre_out)
            post_image.crop(box).resize((CHIP_SIZE, CHIP_SIZE)).save(post_out)

            rows.append(
                ChipRecord(chip_id, _repo_relative(pre_out), _repo_relative(post_out),
                           damage, disaster)
            )

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "pre_path", "post_path", "damage_class", "disaster"])
        for r in rows:
            writer.writerow([r.chip_id, r.pre_path, r.post_path, r.damage_class, r.disaster])

    return manifest_path


def _repo_relative(path: Path) -> str:
    """Store `data/processed/...` rather than `/home/someone/...`."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)  # chips written outside the repo stay absolute


def chip_path(stored: str) -> Path:
    """Inverse of `_repo_relative` - what the Dataset opens."""
    path = Path(stored)
    return path if path.is_absolute() else REPO_ROOT / path


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in (".png", ".tif", ".tiff", ".jpg"):
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def class_counts(manifest_path: Path) -> dict[str, int]:
    counts = {name: 0 for name in DAMAGE_CLASSES}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["damage_class"] in counts:
                counts[row["damage_class"]] += 1
    return counts


# --- Dataset --------------------------------------------------------------


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1))


def to_uint8_tensor(image: Image.Image) -> torch.Tensor:
    """Un-normalised CHW uint8 - a quarter the bytes of `to_tensor`.

    Training hands these to the GPU and normalises there (`normalise_batch`). At batch 256 a
    float32 pre/post pair is 96 MB in every worker's prefetch queue; uint8 makes it 24 MB,
    which is the difference between fitting in RAM and meeting the OOM killer.
    """
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def normalise_batch(batch: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,H,W) -> ImageNet-normalised float. Mirrors `to_tensor` exactly."""
    mean = torch.as_tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch.float().div_(255.0) - mean) / std


class XBDChipDataset(Dataset):
    """Augmentations are applied *identically* to the pre and post chip - flip one and not the
    other and you have invented damage that isn't there."""

    def __init__(self, manifest_path: Path, chip_ids: list[str] | None = None,
                 augment: bool = False, as_uint8: bool = False):
        with open(manifest_path) as f:
            rows = [ChipRecord(**row) for row in csv.DictReader(f)]
        allowed = set(chip_ids) if chip_ids is not None else None
        self.records = [r for r in rows if allowed is None or r.chip_id in allowed]
        self.augment = augment
        # Callers that normalise on the GPU (training) set this; evaluate.py and infer.py
        # want ready-to-use float tensors and leave it off.
        self.as_uint8 = as_uint8

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        pre = Image.open(chip_path(record.pre_path)).convert("RGB")
        post = Image.open(chip_path(record.post_path)).convert("RGB")

        if self.augment:
            pre, post = self._augment(pre, post)

        convert = to_uint8_tensor if self.as_uint8 else to_tensor
        return convert(pre), convert(post), CLASS_TO_INDEX[record.damage_class]

    def _augment(self, pre: Image.Image, post: Image.Image):
        import random

        if random.random() < 0.5:
            pre, post = pre.transpose(Image.FLIP_LEFT_RIGHT), post.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            pre, post = pre.transpose(Image.FLIP_TOP_BOTTOM), post.transpose(Image.FLIP_TOP_BOTTOM)
        turns = random.randint(0, 3)
        if turns:
            pre, post = pre.rotate(90 * turns), post.rotate(90 * turns)
        return pre, post

    def labels(self) -> list[int]:
        return [CLASS_TO_INDEX[r.damage_class] for r in self.records]
