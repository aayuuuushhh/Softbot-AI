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
        ("xBD train images", resolve(paths["xbd"]["train"]) / "images", "data/README.md section 1"),
        ("xBD train labels", resolve(paths["xbd"]["train"]) / "labels", "data/README.md section 1"),
        ("xBD test images", resolve(paths["xbd"]["test"]) / "images", "data/README.md section 1"),
        ("xBD test labels", resolve(paths["xbd"]["test"]) / "labels", "data/README.md section 1"),
        ("Ward boundaries", resolve(paths["nepal"]["wards"]), "data/README.md section 2"),
        ("Facilities", resolve(paths["nepal"]["facilities"]), "data/README.md section 3"),
        ("Population raster", resolve(paths["nepal"]["population"]), "data/README.md section 4"),
        ("Chip manifest (train)", resolve(paths["processed"]["manifest"]), "run with --prepare"),
        ("Chip manifest (test)", resolve(paths["processed"]["manifest_test"]), "run with --prepare"),
        ("Chip manifest (holdout)", resolve(paths["processed"]["manifest_holdout"]), "run with --prepare"),
        ("Model checkpoint", resolve(paths["model"]["checkpoint"]), "python -m ml.train"),
    ]

    missing = 0
    for name, path, hint in required:
        if path.exists():
            print(f"  [ok]      {name}")
        else:
            missing += 1
            print(f"  [missing] {name:<24} -> {hint}")

    print()
    if missing:
        print(f"{missing} missing. The dashboard still runs on mock data in the meantime:")
        print("  uvicorn backend.app.main:app --reload   +   cd frontend && npm run dev")
    else:
        print("Everything present.")
    return 0 if missing == 0 else 1


def _report(label: str, manifest, counts: dict) -> None:
    total = sum(counts.values()) or 1
    print(f"\n{label}: {total} chips -> {manifest}")
    for name, count in counts.items():
        print(f"  {name:<14} {count:>7}  ({100 * count / total:5.1f}%)")


def prepare() -> int:
    """Extract chips into three manifests: train, held-out test, held-out earthquake.

    The split matters more than it looks. Training and test come from xBD's own train/test
    division of the same three events, so `manifest_test` measures generalisation to unseen
    tiles. `manifest_holdout` is mexico-earthquake, an event the model never sees at all -
    it is the only earthquake in the pool and the closest proxy for Nepal, so it is the
    number to quote when asked how this performs on seismic damage.
    """
    from ml.datasets.xbd import class_counts, extract_chips

    paths = paths_config()
    chips_root = resolve(paths["processed"]["chips"])
    train_events = paths["xbd"]["train_events"]
    holdout_events = paths["xbd"]["holdout_events"]

    jobs = [
        ("train",   paths["xbd"]["train"], train_events,   paths["processed"]["manifest"]),
        ("test",    paths["xbd"]["test"],  train_events,   paths["processed"]["manifest_test"]),
        ("holdout", paths["xbd"]["train"], holdout_events, paths["processed"]["manifest_holdout"]),
    ]

    missing = [name for name, root, _, _ in jobs if not resolve(root).exists()]
    if missing:
        print(f"No xBD data at {resolve(paths['xbd']['root'])}. See data/README.md section 1.")
        return 1

    summary = {}
    for name, root, events, manifest_key in jobs:
        out_dir = chips_root / name
        manifest = resolve(manifest_key)
        print(f"\nExtracting {name} chips ({', '.join(events)}) -> {out_dir}")
        extract_chips(resolve(root), out_dir, disasters=events, manifest_path=manifest)
        summary[name] = (manifest, class_counts(manifest))

    # The holdout is train-split mexico only; add its test-split tiles so the eval set is whole.
    holdout_manifest = resolve(paths["processed"]["manifest_holdout"])
    extract_chips(
        resolve(paths["xbd"]["test"]), chips_root / "holdout",
        disasters=holdout_events, manifest_path=chips_root / "_holdout_test.csv",
    )
    _merge_manifests(holdout_manifest, chips_root / "_holdout_test.csv")
    (chips_root / "_holdout_test.csv").unlink(missing_ok=True)
    from ml.datasets.xbd import class_counts as _cc
    summary["holdout"] = (holdout_manifest, _cc(holdout_manifest))

    for name, (manifest, counts) in summary.items():
        _report(name, manifest, counts)

    print("\nTrain on the first manifest; the other two are never seen during fitting:")
    print("  python -m ml.train --manifest data/processed/manifest.csv --epochs 8")
    return 0


def _merge_manifests(base, extra) -> None:
    """Append `extra`'s rows onto `base`, keeping a single header."""
    import csv

    with open(extra) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    with open(base, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writerows(rows)


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
