"""Build tier3 subset for extra fine-tune / eval (wildfire, flood, tornado, etc.)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tier3 disasters useful for hackathon scope (hazard diversity)
TIER3_KEEP_PREFIXES = (
    "portugal-wildfire",
    "pinery-bushfire",
    "woolsey-fire",
    "nepal-flooding",
    "joplin-tornado",
    "tuscaloosa-tornado",
    "moore-tornado",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier3-dir", type=Path, default=REPO_ROOT / "data" / "tier3")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "tier3_subset")
    args = parser.parse_args()

    src = args.tier3_dir
    out = args.output_dir
    if not (src / "images").exists():
        raise SystemExit(f"Missing {src / 'images'} — extract tier3 archive first")

    for sub in ("images", "labels", "targets"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    copied = 0
    for prefix in TIER3_KEEP_PREFIXES:
        for sub in ("images", "labels", "targets"):
            folder = src / sub
            if not folder.exists():
                continue
            for f in folder.glob(f"{prefix}*"):
                shutil.copy2(f, out / sub / f.name)
                if sub == "images" and "_post_disaster" in f.name:
                    copied += 1

    print(f"Tier3 subset -> {out} ({copied} post-disaster pairs)")


if __name__ == "__main__":
    main()
