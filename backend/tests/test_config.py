from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, _resolve_against_repo_root


def test_rejects_unknown_inference_mode():
    with pytest.raises(ValidationError, match="Invalid INFERENCE_MODE"):
        Settings(inference_mode="tensorflow")


@pytest.mark.parametrize("mode", ["stub", "docker", "pytorch"])
def test_accepts_every_supported_mode(mode):
    assert Settings(inference_mode=mode).inference_mode == mode


def test_inference_mode_is_case_insensitive():
    assert Settings(inference_mode="STUB").inference_mode == "stub"


@pytest.mark.parametrize("field", ["grid_rows", "grid_cols"])
def test_grid_dimensions_must_be_positive(field):
    with pytest.raises(ValidationError):
        Settings(**{field: 0})


def test_relative_data_dirs_resolve_against_repo_root():
    """A relative DEMO_DATA_DIR must not depend on the process cwd."""
    settings = Settings(demo_data_dir=Path("data/demo"))
    assert settings.demo_data_dir.is_absolute()
    assert settings.demo_data_dir.parts[-2:] == ("data", "demo")


def test_absolute_data_dirs_are_left_alone(tmp_path):
    assert _resolve_against_repo_root(tmp_path) == tmp_path
