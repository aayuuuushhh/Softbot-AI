#!/usr/bin/env python3
"""Resume damage training on Kaggle — thin wrapper around kaggle_train.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            str(REPO / "ml" / "finetune" / "kaggle_train.py"),
            "--stage",
            "dmg",
            "--skip-deps",
        ],
        check=True,
        cwd=REPO,
    )
