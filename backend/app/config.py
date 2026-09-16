"""Loads config/priority.yaml and config/paths.yaml.

Reloadable at runtime so priority weights can be re-tuned during a demo without a restart.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def priority_config() -> dict[str, Any]:
    return _load("priority.yaml")


@lru_cache(maxsize=None)
def paths_config() -> dict[str, Any]:
    return _load("paths.yaml")


def reload_config() -> None:
    """Drop cached config so the next call re-reads the YAML from disk."""
    priority_config.cache_clear()
    paths_config.cache_clear()


def resolve(path: str) -> Path:
    """Resolve a repo-relative path from paths.yaml to an absolute path."""
    return REPO_ROOT / path
