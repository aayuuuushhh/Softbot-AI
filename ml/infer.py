"""Turn a pre/post tile pair into per-building damage classes.

    python -m ml.infer --pre data/raw/pre.tif --post data/raw/post.tif \
        --footprints data/nepal/buildings.geojson --out out.geojson

Two paths, same output shape:
  --mode model      trained siamese classifier (needs a checkpoint)
  --mode heuristic  image-difference fallback, no checkpoint, always available

The chosen mode travels with the result so the dashboard can label it. Never let a heuristic
result be presented as a model result.

See PROJECT_PLAN.md section 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.services.classify import classify_heuristic

CHIP_SIZE = 128
PADDING_PX = 10


def load_image(path: Path) -> np.ndarray:
    """RGB array. Uses rasterio for GeoTIFFs so georeferencing survives; PIL otherwise."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read([1, 2, 3]).transpose(1, 2, 0)
        return array.astype(np.uint8)
    return np.asarray(Image.open(path).convert("RGB"))


def crop(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    min_x, min_y = max(0, min_x - PADDING_PX), max(0, min_y - PADDING_PX)
    max_x = min(image.shape[1], max_x + PADDING_PX)
    max_y = min(image.shape[0], max_y + PADDING_PX)
    chip = image[min_y:max_y, min_x:max_x]
    if chip.size == 0:
        return np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(chip).resize((CHIP_SIZE, CHIP_SIZE)))


def pixel_bounds(geometry: dict, transform) -> tuple[int, int, int, int]:
    """Polygon lon/lat ring -> pixel bounding box, via the raster's affine transform."""
    ring = geometry["coordinates"][0]
    if transform is None:
        xs = [int(p[0]) for p in ring]
        ys = [int(p[1]) for p in ring]
    else:
        inverse = ~transform
        pixels = [inverse * (p[0], p[1]) for p in ring]
        xs = [int(p[0]) for p in pixels]
        ys = [int(p[1]) for p in pixels]
    return min(xs), min(ys), max(xs), max(ys)


def classify_all(
    pre: np.ndarray,
    post: np.ndarray,
    footprints: list[dict],
    transform,
    mode: str,
    checkpoint: Path | None,
    device: str,
) -> list[dict]:
    chips = [(crop(pre, pixel_bounds(f["geometry"], transform)),
              crop(post, pixel_bounds(f["geometry"], transform))) for f in footprints]

    if mode == "heuristic":
        predictions = [classify_heuristic(pre_chip, post_chip) for pre_chip, post_chip in chips]
        results = [(p.damage_class.value, p.confidence) for p in predictions]
    else:
        results = _classify_with_model(chips, checkpoint, device)

    output = []
    for feature, (damage_class, confidence) in zip(footprints, results):
        properties = dict(feature.get("properties", {}))
        properties.update({
            "damage_class": damage_class,
            "confidence": round(float(confidence), 2),
            "reviewed_by_human": False,
        })
        output.append({"type": "Feature", "geometry": feature["geometry"], "properties": properties})
    return output


def _classify_with_model(chips, checkpoint: Path | None, device: str):
    import torch

    from ml.datasets.xbd import to_tensor
    from ml.models.siamese import DAMAGE_CLASSES, load_checkpoint

    if checkpoint is None or not Path(checkpoint).exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Train one with `python -m ml.train`, "
            "or run with --mode heuristic."
        )

    model = load_checkpoint(str(checkpoint), device)
    results = []
    batch_size = 64
    for start in range(0, len(chips), batch_size):
        batch = chips[start : start + batch_size]
        pre_batch = torch.stack([to_tensor(Image.fromarray(c[0])) for c in batch]).to(device)
        post_batch = torch.stack([to_tensor(Image.fromarray(c[1])) for c in batch]).to(device)
        predicted, confidence = model.predict(pre_batch, post_batch)
        results.extend(
            (DAMAGE_CLASSES[i], float(c)) for i, c in zip(predicted.cpu().tolist(), confidence.cpu().tolist())
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--footprints", type=Path, required=True,
                        help="GeoJSON of building polygons (OSM, Open Buildings or xBD)")
    parser.add_argument("--out", type=Path, default=Path("data/processed/assessment.geojson"))
    parser.add_argument("--mode", choices=["model", "heuristic"], default="heuristic")
    parser.add_argument("--checkpoint", type=Path, default=Path("ml/checkpoints/siamese_resnet18.pt"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if _cuda_available() else "cpu")

    pre = load_image(args.pre)
    post = load_image(args.post)
    if pre.shape != post.shape:
        raise SystemExit(
            f"Pre {pre.shape} and post {post.shape} differ. Co-register them first "
            "(data/README.md, preprocessing step 2)."
        )

    transform = _read_transform(args.pre)
    footprints = json.loads(args.footprints.read_text())["features"]
    print(f"{len(footprints)} footprints, mode={args.mode}, device={device}")

    features = classify_all(pre, post, footprints, transform, args.mode, args.checkpoint, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    counts: dict[str, int] = {}
    for feature in features:
        cls = feature["properties"]["damage_class"]
        counts[cls] = counts.get(cls, 0) + 1
    print("predicted:", counts)
    print(f"wrote {args.out}")


def _read_transform(path: Path):
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    import rasterio

    with rasterio.open(path) as src:
        return src.transform


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
