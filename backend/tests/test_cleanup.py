from __future__ import annotations

import os
import time

from app.services.cleanup import cleanup_old_jobs


def _age_dir(path, hours: float) -> None:
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_removes_only_stale_directories(tmp_path):
    fresh = tmp_path / "fresh_job"
    stale = tmp_path / "stale_job"
    fresh.mkdir()
    stale.mkdir()
    (stale / "mask.png").write_bytes(b"x")
    _age_dir(stale, hours=48)

    removed = cleanup_old_jobs(tmp_path, max_age_hours=24)

    assert removed == 1
    assert fresh.exists()
    assert not stale.exists()


def test_ignores_loose_files(tmp_path):
    loose = tmp_path / "note.txt"
    loose.write_text("keep me")
    _age_dir(loose, hours=48)

    assert cleanup_old_jobs(tmp_path, max_age_hours=24) == 0
    assert loose.exists()


def test_sweeps_every_root(tmp_path):
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    for root in (uploads, outputs):
        root.mkdir()
        job = root / "job"
        job.mkdir()
        _age_dir(job, hours=48)

    assert cleanup_old_jobs(uploads, outputs, max_age_hours=24) == 2


def test_zero_max_age_disables_sweep(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    _age_dir(job, hours=999)

    assert cleanup_old_jobs(tmp_path, max_age_hours=0) == 0
    assert job.exists()


def test_missing_root_is_not_an_error(tmp_path):
    assert cleanup_old_jobs(tmp_path / "does_not_exist") == 0
