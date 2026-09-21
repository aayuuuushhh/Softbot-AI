"""Training datasets for S1 (xBD tiles -> segmentation masks) and S2 (field photos).

Only the training scripts import this module; the API never does.

xBD layout, as written by scripts/fetch_xbd.sh (xBD's own train/test split kept):

    data/raw/xbd/{train,test}/images/<event>_<id>_{pre,post}_disaster.png
    data/raw/xbd/{train,test}/labels/<event>_<id>_{pre,post}_disaster.json

Mask classes follow core/vision/models.py CHANGE_CLASSES exactly:

    0 background   1 none   2 minor   3 major   4 destroyed   255 ignore

`un-classified` buildings become 255 - excluded from the loss rather than
guessed at.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely import wkt as shapely_wkt
from torch.utils.data import Dataset

from core.vision.models import CHANGE_CLASSES, GROUND_CLASSES

IGNORE_INDEX = 255

XBD_SUBTYPE_TO_CLASS = {
    "no-damage": CHANGE_CLASSES.index("none"),
    "minor-damage": CHANGE_CLASSES.index("minor"),
    "major-damage": CHANGE_CLASSES.index("major"),
    "destroyed": CHANGE_CLASSES.index("destroyed"),
    "un-classified": IGNORE_INDEX,
}

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# --------------------------------------------------------------------------
# xBD
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class XBDTile:
    stem: str  # <event>_<id>
    event: str
    pre_image: Path
    post_image: Path
    post_label: Path


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def list_xbd_tiles(
    root: Path,
    split: str = "train",
    events: list[str] | None = None,
    exclude_events: list[str] | None = None,
) -> list[XBDTile]:
    """Every complete pre/post tile pair in one xBD split, optionally filtered by event."""
    images_dir, labels_dir = root / split / "images", root / split / "labels"
    tiles: list[XBDTile] = []
    for label in sorted(labels_dir.glob("*_post_disaster.json")):
        stem = label.name.removesuffix("_post_disaster.json")
        event = stem.rsplit("_", 1)[0]
        if events and event not in events:
            continue
        if exclude_events and event in exclude_events:
            continue
        pre = _find_image(images_dir, f"{stem}_pre_disaster")
        post = _find_image(images_dir, f"{stem}_post_disaster")
        if pre is None or post is None:
            continue
        tiles.append(XBDTile(stem, event, pre, post, label))
    return tiles


def split_tiles(
    tiles: list[XBDTile], val_fraction: float, seed: int
) -> tuple[list[XBDTile], list[XBDTile]]:
    """Split on whole tiles. Neighbouring buildings in one tile look alike, so
    splitting below the tile would leak validation into training."""
    shuffled = list(tiles)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[cut:], shuffled[:cut]


def building_polygons(label_path: Path) -> list[tuple[np.ndarray, int]]:
    """(pixel polygon (N, 2) int32, class index) for every building in a label file."""
    data = json.loads(label_path.read_text())
    out: list[tuple[np.ndarray, int]] = []
    for feature in data.get("features", {}).get("xy", []):
        subtype = feature.get("properties", {}).get("subtype", "no-damage")
        cls = XBD_SUBTYPE_TO_CLASS.get(subtype, IGNORE_INDEX)
        try:
            geom = shapely_wkt.loads(feature["wkt"])
        except Exception:  # noqa: BLE001 - one bad polygon must not kill a tile
            continue
        polygons = getattr(geom, "geoms", [geom])
        for poly in polygons:
            if poly.is_empty or not hasattr(poly, "exterior"):
                continue
            coords = np.asarray(poly.exterior.coords, dtype=np.float64)[:, :2]
            out.append((np.round(coords).astype(np.int32), cls))
    return out


def rasterize_label(label_path: Path, height: int, width: int) -> np.ndarray:
    """Burn building polygons into an (H, W) uint8 mask.

    Draw order is ascending severity so that where footprints overlap, the
    more severe label wins - the conservative choice for triage. Ignored
    (un-classified) buildings are drawn last so they are never trained on.
    """
    mask = np.zeros((height, width), np.uint8)
    polygons = building_polygons(label_path)
    order = sorted(polygons, key=lambda p: (p[1] == IGNORE_INDEX, p[1]))
    for coords, cls in order:
        cv2.fillPoly(mask, [coords], int(cls))
    return mask


def read_rgb(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"could not decode image: {path}")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def augment_pair(
    pre: np.ndarray, post: np.ndarray, mask: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric augmentation applied *identically* to pre, post and mask.

    Flip one image and not the other and you have invented damage. Photometric
    jitter, by contrast, is applied independently: real acquisitions differ in
    exposure, and the model has to learn to ignore that.
    """
    if rng.random() < 0.5:
        pre, post, mask = pre[:, ::-1], post[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        pre, post, mask = pre[::-1], post[::-1], mask[::-1]
    k = rng.randint(0, 3)
    if k:
        pre, post, mask = np.rot90(pre, k), np.rot90(post, k), np.rot90(mask, k)

    def jitter(img: np.ndarray) -> np.ndarray:
        gain = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-12, 12)
        return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    return jitter(pre), jitter(post), np.ascontiguousarray(mask)


class XBDTileDataset(Dataset):
    """Random `crop`-px windows from xBD tile pairs, with their damage masks.

    Returns uint8 CHW pre/post tensors (normalised on the GPU by the training
    loop - a quarter of the bytes through the loader) and an int64 mask.
    """

    def __init__(
        self,
        tiles: list[XBDTile],
        crop: int = 512,
        augment: bool = False,
        crops_per_tile: int = 1,
        seed: int = 0,
    ) -> None:
        self.tiles = tiles
        self.crop = crop
        self.augment = augment
        self.crops_per_tile = max(1, crops_per_tile)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.tiles) * self.crops_per_tile

    def load_full(self, tile: XBDTile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pre = read_rgb(tile.pre_image)
        post = read_rgb(tile.post_image)
        if post.shape != pre.shape:
            post = cv2.resize(post, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_AREA)
        mask = rasterize_label(tile.post_label, pre.shape[0], pre.shape[1])
        return pre, post, mask

    def __getitem__(self, index: int):
        tile = self.tiles[index // self.crops_per_tile]
        # Deterministic per-item RNG for validation, fresh randomness for training.
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + index)
        pre, post, mask = self.load_full(tile)

        h, w = mask.shape
        c = min(self.crop, h, w)
        if self.augment:
            y0, x0 = rng.randint(0, h - c), rng.randint(0, w - c)
        else:
            y0, x0 = (h - c) // 2, (w - c) // 2
        pre = pre[y0 : y0 + c, x0 : x0 + c]
        post = post[y0 : y0 + c, x0 : x0 + c]
        mask = mask[y0 : y0 + c, x0 : x0 + c]

        if self.augment:
            pre, post, mask = augment_pair(pre, post, mask, rng)

        return (
            torch.from_numpy(np.ascontiguousarray(pre.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(post.transpose(2, 0, 1))),
            torch.from_numpy(mask.astype(np.int64)),
        )


def pixel_class_counts(tiles: list[XBDTile], max_tiles: int = 200, seed: int = 0) -> np.ndarray:
    """Per-class pixel counts over a sample of tiles, for loss weighting."""
    sample = list(tiles)
    random.Random(seed).shuffle(sample)
    counts = np.zeros(len(CHANGE_CLASSES), np.int64)
    from PIL import Image

    for tile in sample[:max_tiles]:
        with Image.open(tile.pre_image) as img:  # header only, no pixel decode
            w, h = img.size
        mask = rasterize_label(tile.post_label, h, w)
        valid = mask[mask != IGNORE_INDEX]
        counts += np.bincount(valid.ravel(), minlength=len(CHANGE_CLASSES))[: len(CHANGE_CLASSES)]
    return counts


# --------------------------------------------------------------------------
# Ground photos
# --------------------------------------------------------------------------


def list_ground_photos(root: Path) -> list[tuple[Path, int]]:
    """`root/<class>/*.jpg` for class in GROUND_CLASSES. Unknown folders are ignored."""
    items: list[tuple[Path, int]] = []
    for index, name in enumerate(GROUND_CLASSES):
        folder = root / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, index))
    return items


def stratified_split(
    items: list[tuple[Path, int]], val_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    train, val = [], []
    for cls in sorted({c for _, c in items}):
        group = [it for it in items if it[1] == cls]
        rng.shuffle(group)
        cut = int(round(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:cut])
        train.extend(group[cut:])
    return train, val


class GroundPhotoDataset(Dataset):
    """Field photos as normalised 224 px tensors (float32, CHW)."""

    def __init__(self, items: list[tuple[Path, int]], size: int = 224, augment: bool = False):
        from torchvision.transforms import v2

        self.items = items
        self.size = size
        if augment:
            self.transform = v2.Compose(
                [
                    v2.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
                    v2.RandomHorizontalFlip(),
                    v2.ColorJitter(0.3, 0.3, 0.2, 0.02),
                    v2.RandomRotation(8),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize(size, antialias=True),
                    v2.CenterCrop(size),
                ]
            )

    def __len__(self) -> int:
        return len(self.items)

    def labels(self) -> list[int]:
        return [c for _, c in self.items]

    def __getitem__(self, index: int):
        from core.vision.preprocess import IMAGENET_MEAN, IMAGENET_STD

        path, cls = self.items[index]
        image = torch.from_numpy(read_rgb(path).transpose(2, 0, 1).copy())
        image = self.transform(image).float() / 255.0
        mean = torch.as_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD).view(3, 1, 1)
        return (image - mean) / std, cls
