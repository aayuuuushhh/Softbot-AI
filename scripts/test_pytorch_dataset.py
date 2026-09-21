"""CPU smoke test: PyTorch train datasets load without index.csv crash.

Requires patched ml/pytorch-xview2 and generated utils/index.csv.
Needs xView2 deps: torch, cv2, albumentations, pandas (install in pytorch env).

Usage:
  python ml/finetune/patch_pytorch_xview2.py
  python scripts/generate_subset_index.py
  python scripts/test_pytorch_dataset.py --data-dir data/train_subset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"


def validate_index_csv(data_dir: Path, index_csv: Path) -> None:
    import pandas as pd

    n_pre = len(list((data_dir / "images").glob("*pre*")))
    df = pd.read_csv(index_csv)
    if df.empty:
        raise SystemExit("index.csv is empty")
    max_idx = int(df["idx"].max())
    print(f"index.csv: {len(df)} rows, max(idx)={max_idx}, n_pre_images={n_pre}")
    if max_idx >= n_pre:
        raise SystemExit(f"max idx {max_idx} >= n_images {n_pre}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "train_subset")
    args = parser.parse_args()

    if not XVIEW2_ROOT.exists():
        raise SystemExit(f"Missing {XVIEW2_ROOT}")

    index_csv = XVIEW2_ROOT / "utils" / "index.csv"
    if not index_csv.exists():
        raise SystemExit(f"Missing {index_csv} — run scripts/generate_subset_index.py first")

    loader_path = XVIEW2_ROOT / "data_loading" / "pytorch_loader.py"
    if "_load_index_csv" not in loader_path.read_text(encoding="utf-8"):
        raise SystemExit("pytorch_loader.py not patched — run ml/finetune/patch_pytorch_xview2.py")

    os.environ.setdefault("XVIEW2_INDEX_CSV", str(index_csv.resolve()))
    data_dir = args.data_dir.resolve()
    validate_index_csv(data_dir, index_csv)

    sys.path.insert(0, str(XVIEW2_ROOT))
    try:
        from data_loading.pytorch_loader import TrainPostDataset, TrainPreDataset
    except ImportError as exc:
        print(f"Skip full dataset test ({exc}). Index validation passed.")
        return

    data_dir_str = str(data_dir)
    print(f"TrainPreDataset({data_dir_str})...")
    pre_ds = TrainPreDataset(data_dir_str, "pre", autoaugment=False)
    print(f"  len={len(pre_ds)}, max_idx={max(pre_ds.idx)}")

    print(f"TrainPostDataset({data_dir_str})...")
    post_ds = TrainPostDataset(data_dir_str, "post", autoaugment=False)
    print(f"  len={len(post_ds)}, max_idx={max(post_ds.idx) if post_ds.idx else 'n/a'}")

    n_pre = len(sorted((data_dir / "images").glob("*pre*")))
    if max(pre_ds.idx) >= n_pre:
        raise SystemExit(f"TrainPreDataset max idx {max(pre_ds.idx)} >= n_images {n_pre}")

    print("Fetching sample batch item 0...")
    _ = pre_ds[0]
    if post_ds.idx:
        _ = post_ds[0]
    print("OK — dataset construction and __getitem__ succeeded.")


if __name__ == "__main__":
    main()
