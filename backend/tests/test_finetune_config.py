"""Tests for finetune config loader."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOAD_CONFIG = REPO / "ml" / "finetune" / "load_config.py"


def test_load_config_localization_exports() -> None:
    result = subprocess.run(
        [sys.executable, str(LOAD_CONFIG), "localization"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'export EPOCHS="10"' in result.stdout
    assert 'export BATCH_SIZE="8"' in result.stdout


def test_load_config_damage_ckpt_pre() -> None:
    result = subprocess.run(
        [sys.executable, str(LOAD_CONFIG), "damage"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CKPT_PRE" in result.stdout
    assert 'export EPOCHS="20"' in result.stdout
