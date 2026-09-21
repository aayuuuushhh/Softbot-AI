"""Sweep stale per-job upload and output directories.

Every analysis writes a job directory that nothing ever reads again once the
response is sent. Without a sweep the disk fills silently, which on a small
deployment box takes down the whole service.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_HOURS = 24


def cleanup_old_jobs(*roots: Path, max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> int:
    """Delete subdirectories older than max_age_hours under each root.

    Returns the number of directories removed. Never raises: a failed sweep
    must not take down the request that triggered it.
    """
    if max_age_hours <= 0:
        return 0

    cutoff = time.time() - max_age_hours * 3600
    removed = 0

    for root in roots:
        if not root.is_dir():
            continue

        for child in root.iterdir():
            if not child.is_dir():
                continue

            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue

            if mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
                logger.info("Removed stale job dir: %s", child)

    return removed
