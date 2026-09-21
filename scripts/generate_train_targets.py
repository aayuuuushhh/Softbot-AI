"""Generate xBD target PNG masks from label JSON (wraps michal2409/xView2 convert2png)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = REPO_ROOT / "ml" / "pytorch-xview2" / "utils" / "convert2png.py"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "train",
        help="Directory with images/ and labels/ subfolders",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    if not CONVERTER.exists():
        print(f"Missing converter: {CONVERTER}")
        print("Run: git clone https://github.com/michal2409/xView2.git ml/pytorch-xview2")
        sys.exit(1)

    labels = args.data_dir / "labels"
    images = args.data_dir / "images"
    if not labels.exists() or not images.exists():
        print(f"Expected {images} and {labels}")
        sys.exit(1)

    targets = args.data_dir / "targets"
    targets.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(CONVERTER),
        "--data",
        str(args.data_dir),
        "--n_jobs",
        str(args.n_jobs),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    count = len(list(targets.glob("*.png")))
    print(f"Generated {count} target masks in {targets}")


if __name__ == "__main__":
    main()
