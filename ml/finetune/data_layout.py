"""Build a disjoint xView2 train/test layout from a flat images/targets/labels tree.

Previously both ``train`` and ``test`` symlinked to the same folders, so
validation F1 was train-set leakage. This module keeps a holdout fraction of
pair IDs exclusively under ``test/``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_HOLDOUT_FRAC = float(os.environ.get("XVIEW2_HOLDOUT_FRAC", "0.1"))
SUBFOLDERS = ("images", "targets", "labels")


def _pair_id_from_pre(path: Path) -> str:
    name = path.name
    for marker in ("_pre_disaster.png", "_pre_disaster.jpg", "_pre_disaster.jpeg", "_pre_disaster.tif"):
        if name.lower().endswith(marker):
            return name[: -len(marker)]
    stem = path.stem
    if stem.endswith("_pre_disaster"):
        return stem[: -len("_pre_disaster")]
    return stem


def _list_pair_ids(images_dir: Path) -> list[str]:
    ids: list[str] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if "_pre_disaster" in lower:
            ids.append(_pair_id_from_pre(path))
    return ids


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _populate_split(data_root: Path, split: str, pair_ids: list[str]) -> None:
    split_dir = data_root / split
    for sub in SUBFOLDERS:
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    for pair_id in pair_ids:
        for sub in SUBFOLDERS:
            src_dir = data_root / sub
            if not src_dir.is_dir():
                continue
            for src in src_dir.glob(f"{pair_id}_*"):
                if not src.is_file():
                    continue
                _link_or_copy(src, split_dir / sub / src.name)


def ensure_holdout_layout(
    data_root: Path,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
) -> None:
    """Create train/ and test/ with disjoint pair IDs (holdout in test/)."""
    images = data_root / "images"
    if not images.is_dir():
        raise FileNotFoundError(f"Missing images/ under {data_root}")

    train_images = data_root / "train" / "images"
    test_images = data_root / "test" / "images"
    if train_images.is_dir() and test_images.is_dir():
        try:
            same_target = train_images.resolve() == test_images.resolve()
        except OSError:
            same_target = False
        if not same_target:
            print(f"Disjoint train/test layout already present under {data_root}")
            return
        # Leaky directory symlinks to the same tree — rebuild.
        shutil.rmtree(data_root / "train", ignore_errors=True)
        shutil.rmtree(data_root / "test", ignore_errors=True)

    pair_ids = _list_pair_ids(images)
    if not pair_ids:
        raise FileNotFoundError(f"No *_pre_disaster images under {images}")

    holdout_frac = min(max(holdout_frac, 0.05), 0.3)
    n_test = max(1, int(round(len(pair_ids) * holdout_frac)))
    n_test = min(n_test, len(pair_ids) - 1) if len(pair_ids) > 1 else 1
    test_ids = pair_ids[-n_test:]
    train_ids = pair_ids[:-n_test] if len(pair_ids) > 1 else pair_ids

    _populate_split(data_root, "train", train_ids)
    _populate_split(data_root, "test", test_ids)
    print(
        f"xView2 holdout layout under {data_root}: "
        f"train={len(train_ids)} pairs, test={len(test_ids)} pairs "
        f"(holdout_frac={holdout_frac})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    args = parser.parse_args()
    ensure_holdout_layout(args.data_root, holdout_frac=args.holdout_frac)
