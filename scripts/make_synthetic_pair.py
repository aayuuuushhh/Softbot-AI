#!/usr/bin/env python3
"""Generate a synthetic pre/post GeoTIFF pair with known planted damage.

The pair carries a real geotransform over the Rasuwa demo bbox, so detections
come out geotagged and land inside the seeded zones. A `truth.json` records
exactly which building clusters were destroyed, which is what lets the
perception tests assert recall instead of just "it ran".

The post image also gets a sub-pixel shift and an illumination change, so the
alignment and illumination-matching steps in preprocess.py are genuinely
exercised rather than passed a perfectly registered pair.

    python scripts/make_synthetic_pair.py --out data/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

BBOX = (85.20, 27.95, 85.55, 28.25)  # matches scripts/seed_demo.py
SIZE = 1024
CRS = "EPSG:4326"

# Zone centroids from the demo seed; clusters are planted near these so the
# detections fall inside real zones.
ZONE_SITES = [
    ("Dhunche",      85.297, 28.112, "destroyed"),
    ("Syaphru Besi", 85.339, 28.161, "major"),
    ("Ramche",       85.256, 28.036, "none"),
    ("Laharepauwa",  85.283, 27.988, "destroyed"),
    ("Kalikasthan",  85.245, 27.973, "none"),
    ("Haku",         85.312, 28.070, "destroyed"),
    ("Gatlang",      85.238, 28.187, "minor"),
    ("Thuman",       85.375, 28.205, "major"),
]

DAMAGE_FRACTION = {"none": 0.0, "minor": 0.25, "major": 0.55, "destroyed": 0.9}


def lonlat_to_px(lon: float, lat: float) -> tuple[int, int]:
    minlon, minlat, maxlon, maxlat = BBOX
    col = int((lon - minlon) / (maxlon - minlon) * SIZE)
    row = int((maxlat - lat) / (maxlat - minlat) * SIZE)
    return col, row


def make_terrain(rng: np.random.Generator) -> np.ndarray:
    """Low-frequency green/brown hillside with a river."""
    coarse = rng.normal(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    terrain = np.kron(coarse, np.ones((SIZE // 32, SIZE // 32, 1), np.float32))
    terrain = terrain * 12.0 + np.array([96.0, 116.0, 74.0], np.float32)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    river = np.abs(yy - (SIZE * 0.55 + 70 * np.sin(xx / 130.0))) < 11
    terrain[river] = np.array([120.0, 128.0, 140.0], np.float32)

    terrain += rng.normal(0.0, 3.0, terrain.shape).astype(np.float32)
    return np.clip(terrain, 0, 255)


def make_ground_photo(rng: np.random.Generator, destroyed: bool, size: int = 512) -> np.ndarray:
    """Synthesise a field photograph of an intact or collapsed building.

    The two cases differ along exactly the axes StubGroundClassifier measures:
    an intact facade has long straight high-contrast lines and saturated
    colour; rubble has chaotic high-frequency texture and a dull grey palette.
    """
    import cv2

    img = np.zeros((size, size, 3), np.float32)
    img[:, :] = np.array([150.0, 170.0, 195.0])  # sky
    ground_y = int(size * 0.78)
    img[ground_y:, :] = np.array([110.0, 100.0, 88.0])

    if not destroyed:
        # Facade: a saturated block with regular window rows and roof line.
        x0, x1 = int(size * 0.14), int(size * 0.86)
        y0 = int(size * 0.24)
        wall = np.array([196.0, 172.0, 138.0])
        img[y0:ground_y, x0:x1] = wall
        cv2.rectangle(img, (x0, y0), (x1, ground_y), (92, 74, 56), 3)
        for row in range(4):
            wy = y0 + int((row + 0.6) * (ground_y - y0) / 4.6)
            for col in range(5):
                wx = x0 + int((col + 0.5) * (x1 - x0) / 5.4)
                cv2.rectangle(img, (wx, wy), (wx + 34, wy + 44), (58, 66, 78), -1)
                cv2.rectangle(img, (wx, wy), (wx + 34, wy + 44), (30, 34, 40), 2)
        cv2.line(img, (x0 - 16, y0), (x1 + 16, y0), (78, 60, 44), 6)   # eaves
        cv2.line(img, (x0, y0), (x0, ground_y), (120, 96, 72), 2)
        cv2.line(img, (x1, y0), (x1, ground_y), (120, 96, 72), 2)
    else:
        # Rubble: overlapping dull polygons, no coherent long edges.
        pile_top = int(size * 0.46)
        for _ in range(180):
            cx = int(rng.uniform(size * 0.08, size * 0.92))
            cy = int(rng.uniform(pile_top, size * 0.96))
            n = int(rng.integers(3, 6))
            pts = np.array([
                [cx + int(rng.normal(0, 22)), cy + int(rng.normal(0, 16))] for _ in range(n)
            ], np.int32)
            grey = float(rng.uniform(70, 145))
            colour = (grey, grey * rng.uniform(0.94, 1.03), grey * rng.uniform(0.88, 0.98))
            cv2.fillPoly(img, [pts], colour)
        img[pile_top:] += rng.normal(0.0, 26.0, img[pile_top:].shape).astype(np.float32)

    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shift", type=float, default=1.7,
                        help="pixels of translation applied to the post image")
    parser.add_argument("--ground", action="store_true", default=True,
                        help="also emit synthetic field photographs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    terrain = make_terrain(rng)
    pre = terrain.copy()

    clusters: list[dict] = []
    for zone, lon, lat, severity in ZONE_SITES:
        cx, cy = lonlat_to_px(lon, lat)
        n_buildings = int(rng.integers(14, 24))
        buildings = []
        for _ in range(n_buildings):
            bx = int(np.clip(cx + rng.normal(0, 16), 6, SIZE - 20))
            by = int(np.clip(cy + rng.normal(0, 16), 6, SIZE - 20))
            bw, bh = int(rng.integers(5, 11)), int(rng.integers(5, 11))
            # Bright rooftops: corrugated metal reads far brighter than terrain
            roof = rng.uniform(175, 235)
            pre[by : by + bh, bx : bx + bw] = np.array([roof, roof * 0.97, roof * 0.9])
            buildings.append((bx, by, bw, bh))
        clusters.append(
            {"zone": zone, "lon": lon, "lat": lat, "severity": severity,
             "center_px": [cx, cy], "buildings": buildings}
        )

    # --- post image: destroy a fraction of each cluster's buildings ---
    post = pre.copy()
    truth = []
    for cluster in clusters:
        frac = DAMAGE_FRACTION[cluster["severity"]]
        buildings = cluster["buildings"]
        n_hit = int(round(len(buildings) * frac))
        hit_idx = rng.choice(len(buildings), size=n_hit, replace=False) if n_hit else []
        for i in hit_idx:
            bx, by, bw, bh = buildings[i]
            # Rubble: dark, desaturated, high-variance
            rubble = rng.normal(78, 18, (bh, bw, 3)).astype(np.float32)
            post[by : by + bh, bx : bx + bw] = np.clip(rubble, 0, 255)
        truth.append({
            "zone": cluster["zone"],
            "lon": cluster["lon"],
            "lat": cluster["lat"],
            "severity": cluster["severity"],
            "center_px": cluster["center_px"],
            "buildings_total": len(buildings),
            "buildings_destroyed": int(n_hit),
        })

    # --- realism: sub-pixel misregistration + illumination change + noise ---
    if args.shift:
        import cv2

        matrix = np.array([[1.0, 0.0, args.shift], [0.0, 1.0, args.shift * 0.6]], np.float32)
        post = cv2.warpAffine(post, matrix, (SIZE, SIZE),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    post = post * 0.88 + 22.0  # hazier, flatter post-monsoon light
    post += rng.normal(0.0, 3.0, post.shape).astype(np.float32)
    post = np.clip(post, 0, 255)

    transform = from_bounds(*BBOX, SIZE, SIZE)
    profile = dict(driver="GTiff", height=SIZE, width=SIZE, count=3,
                   dtype="uint8", crs=CRS, transform=transform, compress="deflate")

    for name, arr in (("pre.tif", pre), ("post.tif", post)):
        with rasterio.open(args.out / name, "w", **profile) as dst:
            dst.write(np.transpose(arr.astype(np.uint8), (2, 0, 1)))
        print(f"wrote {args.out / name}")

    # --- synthetic field photographs, for S2 and the fusion override ---
    ground_manifest = []
    if args.ground:
        import cv2

        ground_dir = args.out / "ground"
        ground_dir.mkdir(parents=True, exist_ok=True)
        # One photo per cluster: destroyed clusters get rubble, the rest intact.
        for t in truth:
            destroyed = t["severity"] in ("destroyed", "major")
            photo = make_ground_photo(rng, destroyed=destroyed)
            name = f"{t['zone'].lower().replace(' ', '_')}.jpg"
            cv2.imwrite(str(ground_dir / name), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
            ground_manifest.append({
                "file": f"ground/{name}", "zone": t["zone"],
                "lon": t["lon"], "lat": t["lat"],
                "expected": "destroyed" if destroyed else "intact",
            })
        print(f"wrote {len(ground_manifest)} ground photos to {ground_dir}")

    truth_path = args.out / "truth.json"
    truth_path.write_text(json.dumps(
        {"bbox": list(BBOX), "size": SIZE, "crs": CRS,
         "applied_shift_px": [args.shift, args.shift * 0.6], "clusters": truth,
         "ground_photos": ground_manifest},
        indent=2,
    ))
    print(f"wrote {truth_path}")

    destroyed = [t["zone"] for t in truth if t["severity"] == "destroyed"]
    total = sum(t["buildings_destroyed"] for t in truth)
    print(f"\n{total} buildings destroyed across {len(truth)} clusters")
    print(f"destroyed zones: {', '.join(destroyed)}")


if __name__ == "__main__":
    main()
