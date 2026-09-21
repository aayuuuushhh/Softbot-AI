#!/usr/bin/env python3
"""Environment preflight. Run this first whenever anything looks wrong.

Checks the things that fail silently: a CPU-only torch, a torch/torchvision
version mismatch, a missing GeoTIFF reader, an unreachable MongoDB.
"""

from __future__ import annotations

import importlib
import shutil
import socket
import sys
from urllib.parse import urlparse

# torch 2.6.0 must be paired with torchvision 0.21.0; mismatches surface as
# obscure C++ symbol errors at import time rather than a clear message.
TORCH_TORCHVISION = {
    "2.6": "0.21",
    "2.7": "0.22",
    "2.8": "0.23",
}

REQUIRED = [
    "torch", "torchvision", "torch_geometric", "cv2", "rasterio", "pyproj",
    "shapely", "numpy", "fastapi", "pydantic", "pydantic_settings", "motor",
    "pymongo", "networkx", "anthropic", "reportlab",
]

ok = True


def fail(msg: str) -> None:
    global ok
    ok = False
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def good(msg: str) -> None:
    print(f"  ok    {msg}")


print(f"python {sys.version.split()[0]}  ({sys.executable})")

print("\npackages")
mods = {}
for name in REQUIRED:
    try:
        mods[name] = importlib.import_module(name)
        good(f"{name} {getattr(mods[name], '__version__', '?')}")
    except ImportError as exc:
        fail(f"{name} missing ({exc})")

print("\ntorch / cuda")
torch = mods.get("torch")
if torch is None:
    fail("torch not importable; nothing else here can be checked")
else:
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False - the GPU path is dead, "
             "inference will silently run on CPU")
    else:
        good(f"cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        good(f"vram {vram_gb:.1f} GiB")
        if vram_gb < 8:
            warn(f"{vram_gb:.1f} GiB VRAM - keep tile_size<=512 and batch_size small")

    tv = mods.get("torchvision")
    if tv is not None:
        t_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        tv_minor = ".".join(tv.__version__.split("+")[0].split(".")[:2])
        expected = TORCH_TORCHVISION.get(t_minor)
        if expected and expected != tv_minor:
            fail(f"torch {t_minor} expects torchvision {expected}.x, found {tv_minor}.x")
        else:
            good(f"torch {t_minor} / torchvision {tv_minor} paired correctly")

print("\nmongodb")
if shutil.which("mongod") is None:
    warn("mongod not on PATH (fine if you run MongoDB in Docker)")
else:
    good(f"mongod at {shutil.which('mongod')}")

try:
    from core.config import get_settings

    uri = get_settings().mongodb_uri
    parsed = urlparse(uri)
    host, port = parsed.hostname or "localhost", parsed.port or 27017
    with socket.create_connection((host, port), timeout=2):
        good(f"reachable at {host}:{port}")
except ImportError as exc:
    fail(f"cannot import core.config - run from the repo root ({exc})")
except OSError:
    fail(f"nothing listening at {uri} - start it with scripts/run_mongo.sh")

print("\nsecrets")
try:
    from core.config import get_settings

    s = get_settings()
    (good if s.anthropic_api_key else warn)(
        "ANTHROPIC_API_KEY set - web research (Claude web search) enabled" if s.anthropic_api_key
        else "ANTHROPIC_API_KEY unset - allocation runs on the cited baseline, no web research"
    )
except Exception as exc:  # noqa: BLE001
    warn(f"could not read settings: {exc}")

print("\n" + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
