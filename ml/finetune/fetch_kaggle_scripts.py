#!/usr/bin/env python3
"""Fetch pinned ml/finetune scripts from GitHub (no git pull needed).

Use on Kaggle when local edits block `git pull` but you need updated training scripts.

  export DISASTERIQ_SCRIPTS_REF=<40-char-commit-sha>   # required for safety
  python ml/finetune/fetch_kaggle_scripts.py
  python ml/finetune/kaggle_train.py --stage dmg --skip-deps --skip-smoke-test

Unsigned ``main`` tip fetches are refused. Pin a commit SHA and optionally set
``DISASTERIQ_SCRIPTS_SHA256`` to a comma-separated ``name=hexdigest`` map.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_DIR = REPO_ROOT / "ml" / "finetune"
# Submission / canonical remote for DisasterIQ.
GITHUB_REPO = os.environ.get("DISASTERIQ_GITHUB_REPO", "DarkNem4377/DisasterIQ")
# Must be a full commit SHA — never floating "main" (supply-chain risk).
SCRIPTS_REF = os.environ.get("DISASTERIQ_SCRIPTS_REF", "").strip()

FILES = [
    "kaggle_train.py",
    "kaggle_dmg_resume.py",
    "patch_pytorch_xview2.py",
    "smoke_damage_step.py",
    "fetch_kaggle_scripts.py",
    "load_config.py",
    "data_layout.py",
    "config_subset_kaggle.yaml",
    "requirements_kaggle.txt",
    "overrides/main.py",
    "overrides/model/loss.py",
]

_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.I)


def _expected_hashes() -> dict[str, str]:
    raw = os.environ.get("DISASTERIQ_SCRIPTS_SHA256", "").strip()
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, digest = part.split("=", 1)
        out[name.strip()] = digest.strip().lower()
    return out


def fetch_file(name: str, raw_base: str, expected: dict[str, str]) -> None:
    url = f"{raw_base}/{name}"
    dest = FINETUNE_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to fetch {url}: {exc}") from exc

    digest = hashlib.sha256(data).hexdigest()
    want = expected.get(name)
    if want and digest != want.lower():
        raise SystemExit(
            f"SHA256 mismatch for {name}: got {digest}, expected {want}. "
            "Refusing to overwrite local scripts."
        )
    dest.write_bytes(data)
    print(f"Fetched {name} (sha256={digest[:12]}…)")


def main() -> None:
    if not _SHA_RE.match(SCRIPTS_REF):
        raise SystemExit(
            "Refusing unsigned fetch: set DISASTERIQ_SCRIPTS_REF to a full "
            "40-character commit SHA (not 'main'). Optional: "
            "DISASTERIQ_SCRIPTS_SHA256='kaggle_train.py=<hex>,…'."
        )
    raw_base = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{SCRIPTS_REF}/ml/finetune"
    expected = _expected_hashes()
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch_file(name, raw_base, expected)
    print(f"Done — pinned {GITHUB_REPO}@{SCRIPTS_REF[:12]}")
    print("Next: python ml/finetune/kaggle_train.py --stage dmg --skip-deps")


if __name__ == "__main__":
    main()
