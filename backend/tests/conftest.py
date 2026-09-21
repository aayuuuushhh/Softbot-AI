"""Test-suite bootstrap.

Environment variables are set before ``app.config`` is imported anywhere, so
the ``settings`` singleton is built from these values. pydantic-settings ranks
real environment variables above ``.env``, which keeps the suite hermetic: a
developer's local .env (Docker paths, live API keys) can never steer a test.
"""

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ["INFERENCE_MODE"] = "stub"
os.environ["DEMO_DATA_DIR"] = str(_REPO_ROOT / "data" / "demo")
os.environ["TEST_DATA_DIR"] = str(_REPO_ROOT / "data" / "test")
os.environ["GEMINI_API_KEYS"] = ""
os.environ["ACCESS_TOKEN"] = ""
os.environ["RATE_LIMIT_REQUESTS"] = "0"


@pytest.fixture
def anyio_backend():
    """Run @pytest.mark.anyio tests on asyncio only — the loop uvicorn uses."""
    return "asyncio"
