"""Raster I/O and normalisation for the overhead pipeline (S1's front half).

Everything here is deterministic and torch-free, so it is equally usable by the
`stub` and `torch` backends.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine, xy

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"


@dataclass(slots=True)
class Raster:
    """An 8-bit RGB image plus enough geospatial context to geotag a pixel."""

    image: np.ndarray  # (H, W, 3) uint8
    transform: Affine
    crs: str | None
    path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _to_uint8_rgb(bands: np.ndarray) -> np.ndarray:
    """(C, H, W) of any dtype -> (H, W, 3) uint8.

    Satellite products are routinely uint16 or float reflectance, and often
    carry more than three bands. We take the first three (or replicate a single
    band) and percentile-stretch, which is far more robust than assuming a
    fixed dynamic range.
    """
    if bands.ndim == 2:
        bands = bands[None, ...]

    if bands.shape[0] >= 3:
        arr = bands[:3]
    else:
        arr = np.repeat(bands[:1], 3, axis=0)

    # Decide on the SOURCE dtype, before the float cast below. Checking after
    # the cast would stretch every image, including 8-bit RGB that is already
    # correctly scaled - which silently alters the pixel values the detector
    # sees and wrecks the scene's colour balance.
    needs_stretch = bands.dtype != np.uint8

    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)

    if needs_stretch:
        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            band = arr[..., c]
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                out[..., c] = 0.0
                continue
            lo, hi = np.percentile(finite, (2.0, 98.0))
            if hi <= lo:
                lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                out[..., c] = 0.0
                continue
            out[..., c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
        arr = out

    return np.nan_to_num(arr).astype(np.uint8)


def read_geotiff(path: str | Path) -> Raster:
    """Read a GeoTIFF (or any GDAL-readable raster) as 8-bit RGB."""
    path = Path(path)
    with rasterio.open(path) as src:
        bands = src.read()
        crs = str(src.crs) if src.crs else None
        transform = src.transform
    if crs is None:
        log.warning("%s has no CRS; detections will not be geotagged", path.name)
    return Raster(image=_to_uint8_rgb(bands), transform=transform, crs=crs, path=path)


# --------------------------------------------------------------------------
# Geotagging
# --------------------------------------------------------------------------


def pixel_to_lonlat(
    transform: Affine, crs: str | None, col: float, row: float
) -> tuple[float, float]:
    """Pixel (col, row) -> (lon, lat) in WGS84.

    Without this every detection is just a box in an image; with it, a
    detection is something a field team can be sent to.
    """
    x, y = xy(transform, row, col, offset="center")
    if crs is None or crs.upper() == WGS84:
        return float(x), float(y)
    lon, lat = Transformer.from_crs(crs, WGS84, always_xy=True).transform(x, y)
    return float(lon), float(lat)


def pixel_area_m2(transform: Affine, crs: str | None) -> float:
    """Ground area of one pixel, in m^2. Approximate for geographic CRSs."""
    sx, sy = abs(transform.a), abs(transform.e)
    if crs is not None and crs.upper() != WGS84:
        return sx * sy  # projected CRS: units are already metres
    # Degrees: 1 deg latitude ~ 111,320 m; longitude shrinks with latitude, but
    # at the scale of a single tile the error is immaterial.
    return (sx * 111_320.0) * (sy * 111_320.0)


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def align_pair(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Register `post` onto `pre` and match its illumination.

    Two separate corrections, both of which otherwise dominate the change
    signal:

    * **Sub-pixel shift** via FFT phase correlation. Satellite revisits are
      never pixel-aligned, and an uncorrected shift lights up every building
      edge as "change".
    * **Illumination** via per-channel mean/std matching. A pre-monsoon scene
      and a post-monsoon scene differ in sun angle and haze, which a raw
      difference reads as damage everywhere.

    Returns the corrected `post` and the (dx, dy) shift that was applied.
    """
    if pre.shape != post.shape:
        post = cv2.resize(post, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_AREA)

    pre_g = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
    post_g = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # A Hann window suppresses the edge discontinuity that would otherwise
    # dominate the correlation peak.
    window = cv2.createHanningWindow((pre_g.shape[1], pre_g.shape[0]), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(pre_g * window, post_g * window)

    if abs(dx) > 0.05 or abs(dy) > 0.05:
        matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
        post = cv2.warpAffine(
            post,
            matrix,
            (pre.shape[1], pre.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return match_illumination(pre, post), (float(dx), float(dy))


def match_illumination(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shift `target`'s per-channel mean and spread onto `reference`'s."""
    out = np.empty_like(target, dtype=np.float32)
    for c in range(3):
        ref_c, tgt_c = reference[..., c].astype(np.float32), target[..., c].astype(np.float32)
        tgt_std = tgt_c.std()
        if tgt_std < 1e-6:
            out[..., c] = tgt_c
            continue
        out[..., c] = (tgt_c - tgt_c.mean()) * (ref_c.std() / tgt_std) + ref_c.mean()
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Cloud cover
# --------------------------------------------------------------------------


def estimate_cloud_fraction(image: np.ndarray) -> float:
    """Fraction of the scene that is probably cloud: bright and desaturated.

    This is what makes the multi-modal constraint real rather than decorative.
    Fusion scales satellite confidence by (1 - cloud_fraction), so a monsoon
    scene automatically defers to ground photos instead of reporting confident
    nonsense.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation, value = hsv[..., 1], hsv[..., 2]
    cloud = (value > 200) & (saturation < 40)
    return float(cloud.mean())


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    array: np.ndarray

    @property
    def offset(self) -> tuple[int, int]:
        return self.x0, self.y0


def tile_windows(
    height: int, width: int, size: int = 512, overlap: int = 64
) -> list[tuple[int, int, int, int]]:
    """Cover (height, width) with overlapping windows, clipped at the edges.

    Overlap matters: a building bisected by a tile boundary is otherwise seen
    twice, each time as a partial structure.
    """
    if size <= 0:
        raise ValueError("tile size must be positive")
    overlap = max(0, min(overlap, size - 1))
    stride = size - overlap

    def starts(extent: int) -> list[int]:
        if extent <= size:
            return [0]
        pos = list(range(0, extent - size + 1, stride))
        if pos[-1] + size < extent:
            pos.append(extent - size)
        return pos

    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in starts(height)
        for x in starts(width)
    ]


def tiles(image: np.ndarray, size: int = 512, overlap: int = 64) -> Iterator[Tile]:
    h, w = image.shape[:2]
    for x0, y0, x1, y1 in tile_windows(h, w, size, overlap):
        yield Tile(x0, y0, x1, y1, image[y0:y1, x0:x1])


# --------------------------------------------------------------------------
# Tensor conversion
# --------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32, ImageNet-normalised.

    The encoders are ImageNet-pretrained, so they expect these statistics even
    before any fine-tuning.
    """
    arr = image.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


def load_ground_image(path: str | Path, size: int = 224) -> np.ndarray:
    """Read a field photo as RGB, centre-cropped to a square and resized."""
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"could not decode image: {path}")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    square = image[top : top + side, left : left + side]
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
