import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"

INFERENCE_MODES = frozenset({"stub", "docker", "pytorch"})


def _resolve_against_repo_root(value: Path) -> Path:
    """Anchor relative paths to the repo root rather than the process cwd.

    The backend is normally started with cwd=backend/, so a relative path from
    .env such as ``data/demo`` would otherwise resolve to ``backend/data/demo``.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Narration/analysis LLM. Several keys may be supplied comma-separated: the
    # narrator walks the list and moves to the next one when a key is rate
    # limited or rejected, so a spent free-tier quota degrades to the next key
    # rather than to the stub brief. Kept as a plain string because
    # pydantic-settings parses a list[str] field from the environment as JSON,
    # which would reject the natural "k1,k2,k3" spelling.
    gemini_api_keys: str = ""
    gemini_model: str = "gemini-3.6-flash"
    # Free-tier quota is counted per project *per model*: gemini-3.6-flash
    # allows 20 requests a day, and once that is spent every key returns 429 on
    # that model while the same keys still answer on another one. So the
    # narrator walks these in order after the primary model is exhausted,
    # rather than dropping the whole dashboard to the deterministic stub plan.
    # Comma-separated for the same reason as the keys above.
    gemini_fallback_models: str = "gemini-2.5-flash,gemini-2.0-flash"
    demo_data_dir: Path = _REPO_ROOT / "data" / "demo"
    test_data_dir: Path = _REPO_ROOT / "data" / "test"
    inference_mode: str = "stub"  # stub | docker | pytorch
    xview2_docker_image: str = "darknem-xview2-inference"
    pytorch_checkpoint_path: Path = _REPO_ROOT / "ml" / "checkpoints" / "damage_best.ckpt"
    pytorch_inference_dir: Path = _REPO_ROOT / "ml" / "pytorch-inference"
    pytorch_repo_dir: Path = _REPO_ROOT / "ml" / "pytorch-xview2"
    upload_dir: Path = Path(__file__).resolve().parents[1] / "uploads"
    output_dir: Path = Path(__file__).resolve().parents[1] / "outputs"
    grid_rows: int = Field(default=4, gt=0)
    grid_cols: int = Field(default=4, gt=0)
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://disasteriq.vercel.app",
    ]
    # Opt-in regex for preview hosts. Leave empty in production and list exact
    # origins in cors_origins instead of allowing every *.vercel.app site.
    cors_origin_regex: str = ""

    # Endpoint protection. Both are opt-in so the demo works out of the box.
    # access_token: if set, the expensive endpoints require a matching X-API-Key
    #   header (useful only when the backend is called server-side or via a
    #   proxy — a browser-facing token would be exposed to clients).
    # rate_limit_requests per rate_limit_window_seconds, per client IP. Set the
    #   request count to 0 to disable rate limiting.
    access_token: str = ""
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    # When true, /docs and /redoc are disabled (recommended for public deploys).
    disable_openapi_docs: bool = False

    # Free hosting tiers spin the container down after ~15 minutes without
    # inbound traffic, and the next visitor pays a cold start (a minute or more)
    # during which the dashboard can only report the backend as unreachable.
    # The instance therefore pings its own public URL on a timer: the request
    # leaves the container and re-enters through the platform's router, which is
    # what the idle timer actually watches, so the container stays resident.
    #
    # keep_alive_url is normally left empty — Render injects RENDER_EXTERNAL_URL
    # and we use that, so a local or Docker run (no such variable) simply never
    # starts the pinger. Set KEEP_ALIVE=false to let the instance sleep again.
    keep_alive: bool = True
    keep_alive_url: str = ""
    # Must stay comfortably under the platform's idle window (15 min on Render).
    keep_alive_interval_seconds: int = Field(default=600, gt=0)

    @property
    def gemini_model_list(self) -> list[str]:
        """The primary model followed by its fallbacks, blanks and dupes gone."""
        seen: set[str] = set()
        models: list[str] = []
        for raw in [self.gemini_model, *self.gemini_fallback_models.split(",")]:
            model = raw.strip()
            if model and model not in seen:
                seen.add(model)
                models.append(model)
        return models

    @property
    def gemini_key_list(self) -> list[str]:
        """The configured keys, in order, with blanks and duplicates removed."""
        seen: set[str] = set()
        keys: list[str] = []
        for raw in self.gemini_api_keys.split(","):
            key = raw.strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        self.demo_data_dir = _resolve_against_repo_root(self.demo_data_dir)
        self.test_data_dir = _resolve_against_repo_root(self.test_data_dir)
        self.pytorch_checkpoint_path = _resolve_against_repo_root(self.pytorch_checkpoint_path)
        self.pytorch_inference_dir = _resolve_against_repo_root(self.pytorch_inference_dir)
        self.pytorch_repo_dir = _resolve_against_repo_root(self.pytorch_repo_dir)
        return self

    @model_validator(mode="after")
    def _default_keep_alive_url_to_platform(self) -> "Settings":
        """Fall back to the URL the host advertises for this service.

        Render sets RENDER_EXTERNAL_URL on every deploy. Using it means the
        pinger needs no configuration in production and stays off everywhere
        else, so a dev server never talks to the deployed box (or itself).
        """
        if not self.keep_alive_url.strip():
            self.keep_alive_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip()
        self.keep_alive_url = self.keep_alive_url.rstrip("/")
        return self

    @model_validator(mode="after")
    def _normalize_inference_mode(self) -> "Settings":
        mode = self.inference_mode.lower()
        if mode not in INFERENCE_MODES:
            raise ValueError(
                f"Invalid INFERENCE_MODE {self.inference_mode!r}. "
                f"Must be one of {sorted(INFERENCE_MODES)}."
            )
        self.inference_mode = mode
        return self


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
