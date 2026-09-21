"""Prepare train subset for fine-tuning (run after train archive verified and extracted).

Usage:
  python scripts/prepare_train_subset.py --train-dir D:/AMD/data/train

Keeps only earthquake, flood, and wildfire disaster folders for hackathon scope.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

KEEP_PREFIXES = (
    "mexico-earthquake",
    "midwest-flooding",
    "nepal-flooding",
    "socal-fire",
    "santa-rosa-wildfire",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "train_subset")
    args = parser.parse_args()

    xbd_root = args.train_dir
    if (xbd_root / "images").exists():
        # Challenge format: filter by filename prefix in flat dirs
        out = args.output_dir
        for sub in ("images", "labels", "targets"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        for prefix in KEEP_PREFIXES:
            for sub in ("images", "labels", "targets"):
                src = xbd_root / sub
                if not src.exists():
                    continue
                for f in src.glob(f"{prefix}*"):
                    shutil.copy2(f, out / sub / f.name)
        print(f"Filtered flat train layout -> {out}")
        return

    # Disaster-folder layout from split_into_disasters.py
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for d in xbd_root.iterdir():
        if d.is_dir() and any(d.name.startswith(p) or d.name == p for p in KEEP_PREFIXES):
            dest = out / d.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(d, dest)
            print(f"Copied {d.name}")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
