"""Generate index.csv for a train subset (DisasterIQ hackathon).

Ports michal2409/xView2 utils/generate_idx.py with configurable paths so
idx values match the sorted image list in --data-dir (not the full 8k train set).

Usage:
  python scripts/generate_subset_index.py --data-dir data/train_subset \\
    --out ml/pytorch-xview2/utils/index.csv
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_foreground(img_pre: np.ndarray, img_post: np.ndarray):
    h_pre, w_pre, _ = np.where(img_pre > 0)
    h_post, w_post, _ = np.where(img_post > 0)
    if h_pre.size == 0 or h_post.size == 0:
        return None
    min_h = max(int(min(h_pre)), int(min(h_post)))
    max_h = min(int(max(h_pre)), int(max(h_post)))
    min_w = max(int(min(w_pre)), int(min(w_post)))
    max_w = min(int(max(w_pre)), int(max(w_post)))
    return np.s_[min_h:max_h, min_w:max_w]


def get_row(
    idx: int,
    imgs_post: list[str],
    imgs_pre: list[str],
    lbls_post: list[str],
    exclude_idx: set[int],
) -> dict | None:
    if idx in exclude_idx:
        return None
    img_post = cv2.imread(imgs_post[idx])
    img_pre = cv2.imread(imgs_pre[idx])
    if img_post is None or img_pre is None:
        return None
    fg = get_foreground(img_pre, img_post)
    if fg is None:
        return None
    crop = img_post[fg]
    if crop.shape[0] < 512 or crop.shape[1] < 512:
        return None
    row: dict = {"idx": idx, "1": 0, "2": 0, "3": 0, "4": 0}
    lbl = cv2.imread(lbls_post[idx], cv2.IMREAD_UNCHANGED)
    if lbl is None:
        return None
    for cls_ in (1, 2, 3, 4):
        if cls_ in np.unique(lbl):
            row[str(cls_)] = 1
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate index.csv for xView2 PyTorch training")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "train_subset",
        help="Train directory with images/ and targets/ subfolders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "ml" / "pytorch-xview2" / "utils" / "index.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="Optional exclude list JSON (upstream exclude.txt format)",
    )
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for joblib")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    imgs_post = sorted(glob.glob(str(data_dir / "images" / "*post*")))
    imgs_pre = sorted(glob.glob(str(data_dir / "images" / "*pre*")))
    lbls_post = sorted(glob.glob(str(data_dir / "targets" / "*post*")))
    if not imgs_post:
        raise SystemExit(f"No post images in {data_dir / 'images'}")
    if len(imgs_post) != len(imgs_pre) or len(imgs_post) != len(lbls_post):
        raise SystemExit(
            f"Mismatched counts: pre={len(imgs_pre)} post={len(imgs_post)} targets={len(lbls_post)}"
        )

    exclude_idx: set[int] = set()
    if args.exclude and args.exclude.exists():
        exclude_idx = set(json.load(args.exclude.open(encoding="utf-8")))

    n = len(imgs_post)
    rows = Parallel(n_jobs=args.jobs)(
        delayed(get_row)(idx, imgs_post, imgs_pre, lbls_post, exclude_idx)
        for idx in tqdm(range(n), total=n, desc="index.csv")
    )
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No valid rows — check data-dir layout and image sizes")

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    max_idx = int(df["idx"].max())
    print(f"Wrote {len(df)} rows -> {out}")
    print(f"max(idx)={max_idx}, n_images={n} (max must be < {n})")
    if max_idx >= n:
        raise SystemExit("BUG: max idx exceeds image count")


if __name__ == "__main__":
    main()
