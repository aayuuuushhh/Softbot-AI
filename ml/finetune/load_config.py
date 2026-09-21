"""Emit shell export statements from ml/finetune/config_subset.yaml.

Usage (bash):
  eval "$(python ml/finetune/load_config.py localization)"
  eval "$(python ml/finetune/load_config.py damage)"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    raise

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_subset.yaml"

SECTION_KEYS = {
    "localization": {
        "EPOCHS": "epochs",
        "BATCH_SIZE": "batch_size",
        "ENCODER": "encoder",
        "RESULTS_DIR": "results_dir",
        "NUM_WORKERS": "num_workers",
    },
    "damage": {
        "EPOCHS": "epochs",
        "BATCH_SIZE": "batch_size",
        "ENCODER": "encoder",
        "RESULTS_DIR": "results_dir",
        "CKPT_PRE": "ckpt_pre",
        "NUM_WORKERS": "num_workers",
    },
}


def _expand(value: object) -> object:
    """Expand ${VAR} in config values.

    The AMD notebook has no fixed mount to hardcode (unlike Kaggle's /kaggle),
    so config_subset_amd.yaml writes its paths relative to ${WORK_ROOT}.
    Non-string values (epochs, batch size) pass through untouched.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def resolve_config(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env_path = os.environ.get("FINETUNE_CONFIG")
    if env_path:
        return Path(env_path)
    return _DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=["localization", "damage", "data"])
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path (default: FINETUNE_CONFIG or config_subset.yaml)",
    )
    args = parser.parse_args()

    config_path = resolve_config(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.section == "data":
        data = cfg.get("data", {})
        print(f'export DATA_DIR="{_expand(data.get("train_dir", "/data/train_subset"))}"')
        print(f'export TEST_DIR="{_expand(data.get("test_dir", "/data/test"))}"')
        print(f'export RESULTS_ROOT="{_expand(data.get("results_root", "/results"))}"')
        return

    section = cfg.get(args.section, {})
    for env_key, yaml_key in SECTION_KEYS[args.section].items():
        val = section.get(yaml_key)
        if val is not None:
            # Respect caller overrides (e.g. CKPT_PRE=last.ckpt on Kaggle)
            if env_key in os.environ and os.environ[env_key]:
                continue
            print(f'export {env_key}="{_expand(val)}"')


if __name__ == "__main__":
    main()
