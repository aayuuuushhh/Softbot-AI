"""Merge train_subset + tier3_subset into combined_subset for local or Kaggle training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage_kaggle_data import merge_subsets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-subset",
        type=Path,
        default=REPO_ROOT / "data" / "train_subset",
    )
    parser.add_argument(
        "--tier3-subset",
        type=Path,
        default=REPO_ROOT / "data" / "tier3_subset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "combined_subset",
    )
    args = parser.parse_args()

    for src in (args.train_subset, args.tier3_subset):
        if not (src / "images").is_dir():
            raise SystemExit(f"Missing {src / 'images'} — extract/build subset first")

    if args.output_dir.exists():
        import shutil

        shutil.rmtree(args.output_dir)

    pairs = merge_subsets([args.train_subset, args.tier3_subset], args.output_dir)
    print(f"combined_subset -> {args.output_dir} ({pairs} post-disaster pairs)")


if __name__ == "__main__":
    main()
