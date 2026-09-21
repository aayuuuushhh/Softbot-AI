"""Central configuration, loaded from the environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

Backend = Literal["stub", "torch"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_prefix="UDDHAR_",
        extra="ignore",
    )

    # --- Database ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "uddhar"

    # --- Perception ---
    sat_backend: Backend = "stub"
    ground_backend: Backend = "stub"
    satellite_weights: Path = REPO_ROOT / "models" / "satellite_change.pt"
    ground_weights: Path = REPO_ROOT / "models" / "ground_damage.pt"

    # --- Compute ---
    device: str = "cuda"
    tile_size: int = 512
    tile_overlap: int = 64
    batch_size: int = 4

    # --- Fusion (S3) ---
    # Ground imagery outranks satellite: it is immune to cloud cover.
    ground_weight: float = 0.7
    # How near (metres) a ground report must be to a satellite detection to be
    # considered the same structure.
    fusion_radius_m: float = 60.0

    # Severity-weighted damaged fraction of a zone at which it counts as fully
    # destroyed. Rubble never covers a whole ward, so this is well below 1.0.
    # Calibrated on the synthetic demo scene; RECALIBRATE against xBD before
    # trusting severity labels on real imagery.
    damage_saturation_fraction: float = 0.20

    # Typical footprint of one structure, m^2. Only meaningful when the imagery
    # can actually resolve individual buildings - see min_structure_gsd_m2.
    building_footprint_m2: float = 90.0

    # Ground area per pixel, m^2, above which individual structures are not
    # resolvable. Coarser than this and we report damaged area only, rather
    # than inventing a building count the imagery cannot support.
    min_structure_gsd_m2: float = 10.0

    # --- Agent ---
    llm_model: str = "claude-opus-5"
    research_cache_ttl_hours: int = 168
    agent_max_repair_rounds: int = 1

    # --- API ---
    cors_origins: str = "http://localhost:3000"
    upload_dir: Path = REPO_ROOT / "data" / "uploads"

    # --- Secrets (no UDDHAR_ prefix; these are conventional names) ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    @field_validator("cors_origins")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolved_device(self) -> str:
        """The device we can actually use, not the one that was asked for.

        Importing torch is deferred so that config stays importable in
        environments where torch is absent (e.g. a docs build).
        """
        if self.device == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
