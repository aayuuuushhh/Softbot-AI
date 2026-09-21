"""Stage train_subset and/or tier3_subset from Kaggle input into a training directory.

Supports two separate dataset zips on Kaggle:
  - disasteriq-train-subset  -> .../train_subset/{images,labels,targets}
  - tier3_subset (or disasteriq-tier3-subset) -> .../tier3_subset/{images,labels,targets}

When both are attached, files are merged into combined_subset (~7379 pairs).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBFOLDERS = ("images", "labels", "targets")

# Folder names produced by zip_train_subset.ps1 / manual tier3 zip
KNOWN_NAMES = ("train_subset", "tier3_subset")
EXCLUDE_DIR_NAMES = {"demo", "test", "train", "combined_subset", "_combined_test", "_stage_test"}


def find_subset_roots(root: Path) -> list[Path]:
    """Return train_subset and/or tier3_subset directories under a Kaggle input root."""
    if not root.is_dir():
        return []

    found: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved in seen:
            return
        if (candidate / "images").is_dir() and (candidate / "targets").is_dir():
            seen.add(resolved)
            found.append(candidate)

    for name in KNOWN_NAMES:
        add(root / name)
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in EXCLUDE_DIR_NAMES:
                add(child / name)

    if found:
        return found

    # Fallback: single dataset zip with images/ at varying depth (not demo/test)
    for images in root.rglob("images"):
        parent = images.parent
        if parent.name in EXCLUDE_DIR_NAMES or any(p in EXCLUDE_DIR_NAMES for p in parent.parts):
            continue
        if (parent / "targets").is_dir():
            add(parent)
            break
    return found


def merge_subsets(sources: list[Path], dest: Path) -> int:
    """Copy images/labels/targets from each source into dest. Returns post-disaster pair count."""
    for sub in SUBFOLDERS:
        (dest / sub).mkdir(parents=True, exist_ok=True)

    pairs = 0
    for src in sources:
        print(f"Merging {src} -> {dest}")
        for sub in SUBFOLDERS:
            src_dir = src / sub
            if not src_dir.is_dir():
                raise FileNotFoundError(f"Missing {src_dir}")
            for f in src_dir.iterdir():
                if not f.is_file():
                    continue
                shutil.copy2(f, dest / sub / f.name)
                if sub == "images" and "_post_disaster" in f.name:
                    pairs += 1
    return pairs


def stage_training_data(
    input_root: Path,
    dest: Path,
    *,
    merge: bool = True,
) -> Path:
    """Pick one subset or merge all found under input_root into dest."""
    roots = find_subset_roots(input_root)
    if not roots:
        raise FileNotFoundError(
            f"No train_subset/tier3_subset layout under {input_root}. "
            "Attach Kaggle datasets with images/, labels/, targets/."
        )

    print("Found subset roots:")
    for r in roots:
        post = len(list((r / "images").glob("*_post_disaster.png")))
        print(f"  {r} ({post} post-disaster pairs)")

    if len(roots) == 1 and not merge:
        src = roots[0]
        if src.resolve() == dest.resolve():
            return dest
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"Staged single subset -> {dest}")
        return dest

    if len(roots) == 1:
        only = roots[0]
        if only.resolve() == dest.resolve():
            return dest
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(only, dest)
        print(f"Staged single subset -> {dest}")
        return dest

    # Multiple sources: merge
    if dest.exists():
        shutil.rmtree(dest)
    pairs = merge_subsets(roots, dest)
    print(f"Merged {len(roots)} subsets -> {dest} ({pairs} post-disaster pairs)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Kaggle input datasets for fine-tune")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
        / "data"
        / "train_subset",
    )
    parser.add_argument(
        "--combined-dest",
        type=Path,
        default=None,
        help="When merging, write here (default: dest parent / combined_subset)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge all found subsets (default: true when multiple inputs)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Use only the first subset found",
    )
    args = parser.parse_args()

    roots = find_subset_roots(args.input_root)
    dest = args.dest
    if len(roots) > 1 and not args.no_merge:
        dest = args.combined_dest or (args.dest.parent / "combined_subset")

    stage_training_data(
        args.input_root,
        dest,
        merge=not args.no_merge and len(roots) > 1,
    )
    print(dest)


if __name__ == "__main__":
    main()
