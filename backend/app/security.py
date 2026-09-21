"""Lightweight, dependency-free endpoint protection.

Two FastAPI dependencies, both opt-in via config so the demo runs unguarded
by default:

- ``rate_limit``: in-memory sliding-window limiter, per client IP.
- ``require_access_token``: shared-secret check via the ``X-API-Key`` header,
  active only when ``settings.access_token`` is set.

The rate-limit state is process-local; with multiple workers each worker keeps
its own window (fine for abuse mitigation, not a distributed quota). For a
distributed limit, back this with Redis instead.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException, Request

from app.config import settings

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def _evict_idle(cutoff: float) -> None:
    """Drop IPs whose window has fully expired.

    Without this the map keeps one entry per IP ever seen, so a long-running
    deployment leaks memory in proportion to unique visitors rather than to
    active ones.
    """
    idle = [ip for ip, hits in _hits.items() if not hits or hits[-1] < cutoff]
    for ip in idle:
        del _hits[ip]


def rate_limit(request: Request) -> None:
    limit = settings.rate_limit_requests
    window = settings.rate_limit_window_seconds
    if limit <= 0:
        return

    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - window

    with _lock:
        _evict_idle(cutoff)

        hits = _hits[ip]
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = int(hits[0] + window - now) + 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down and try again.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


def require_access_token(x_api_key: str | None = Header(default=None)) -> None:
    expected = settings.access_token
    if not expected:
        return
    # compare_digest raises ValueError on unequal lengths for str; treat that
    # as a failed auth rather than a 500.
    if (
        x_api_key is None
        or len(x_api_key) != len(expected)
        or not secrets.compare_digest(x_api_key, expected)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
