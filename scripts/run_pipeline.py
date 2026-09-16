#!/usr/bin/env python3
"""Sprint driver: prepare data, train, or run a full assessment end to end.

    python scripts/run_pipeline.py --prepare              # chips + class balance
    python scripts/run_pipeline.py --assess --pre a.tif --post b.tif --footprints f.geojson
    python scripts/run_pipeline.py --check                # what is ready, what is missing

See PROJECT_PLAN.md section 6 for where each of these sits in the schedule.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import paths_config, resolve  # noqa: E402


def check() -> int:
    """Readiness report. Run this before you start debugging why nothing works."""
    paths = paths_config()
    required = [
        ("xBD images", resolve(paths["xbd"]["images"]), "data/README.md section 1"),
        ("xBD labels", resolve(paths["xbd"]["labels"]), "data/README.md section 1"),
        ("Ward boundaries", resolve(paths["nepal"]["wards"]), "data/README.md section 2"),
        ("Facilities", resolve(paths["nepal"]["facilities"]), "data/README.md section 3"),
        ("Population raster", resolve(paths["nepal"]["population"]), "data/README.md section 4"),
        ("Chip manifest", resolve(paths["processed"]["manifest"]), "run with --prepare"),
        ("Model checkpoint", resolve(paths["model"]["checkpoint"]), "python -m ml.train"),
    ]

    missing = 0
    for name, path, hint in required:
        if path.exists():
            print(f"  [ok]      {name}")
        else:
            missing += 1
            print(f"  [missing] {name:<18} -> {hint}")

    print()
    if missing:
        print(f"{missing} missing. The dashboard still runs on mock data in the meantime:")
        print("  uvicorn backend.app.main:app --reload   +   cd frontend && npm run dev")
    else:
        print("Everything present.")
    return 0 if missing == 0 else 1


def prepare() -> int:
    from ml.datasets.xbd import class_counts, extract_chips

    paths = paths_config()
    xbd_root = resolve(paths["xbd"]["root"])
    if not xbd_root.exists():
        print(f"No xBD data at {xbd_root}. See data/README.md section 1.")
        return 1

    out_dir = resolve(paths["processed"]["chips"])
    print(f"Extracting chips from {xbd_root} -> {out_dir}")
    manifest = extract_chips(xbd_root, out_dir)

    counts = class_counts(manifest)
    total = sum(counts.values()) or 1
    print(f"\n{total} chips -> {manifest}")
    print("class balance:")
    for name, count in counts.items():
        print(f"  {name:<14} {count:>7}  ({100 * count / total:5.1f}%)")
    print("\nThis imbalance is why train.py uses class-weighted loss and an oversampling sampler.")
    return 0


def assess(args) -> int:
    command = [
        sys.executable, "-m", "ml.infer",
        "--pre", str(args.pre), "--post", str(args.post),
        "--footprints", str(args.footprints), "--mode", args.mode,
    ]
    if args.out:
        command += ["--out", str(args.out)]
    print(" ".join(command))
    return subprocess.call(command, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Report which inputs are present")
    parser.add_argument("--prepare", action="store_true", help="Extract chips and print class balance")
    parser.add_argument("--assess", action="store_true", help="Run inference on a tile pair")
    parser.add_argument("--pre", type=Path)
    parser.add_argument("--post", type=Path)
    parser.add_argument("--footprints", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=["model", "heuristic"], default="heuristic")
    args = parser.parse_args()

    if args.prepare:
        return prepare()
    if args.assess:
        if not (args.pre and args.post and args.footprints):
            parser.error("--assess needs --pre, --post and --footprints")
        return assess(args)
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
