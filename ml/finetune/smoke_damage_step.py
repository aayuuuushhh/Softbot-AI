#!/usr/bin/env python3
"""One-batch smoke test for damage training — surfaces loss/GPU errors in ~30s."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FINETUNE = REPO / "ml" / "finetune"
XVIEW2 = REPO / "ml" / "pytorch-xview2"
sys.path.insert(0, str(REPO))

WORKING = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
DATA = WORKING / "data" / "train_subset"
RESULTS = WORKING / "results"
CONFIG = FINETUNE / "config_subset_kaggle.yaml"
INDEX = XVIEW2 / "utils" / "index.csv"


def main() -> None:
    import subprocess

    subprocess.run([sys.executable, str(FINETUNE / "patch_pytorch_xview2.py")], check=True, cwd=REPO)
    sys.path.insert(0, str(XVIEW2))
    os.environ["XVIEW2_INDEX_CSV"] = str(INDEX)

    from ml.finetune.kaggle_train import (  # noqa: E402
        INDEX_CSV,
        load_yaml_section,
        resolve_checkpoint,
        smoke_damage_inprocess,
    )

    os.environ["XVIEW2_INDEX_CSV"] = str(INDEX_CSV)
    smoke_damage_inprocess(DATA, RESULTS, CONFIG)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
