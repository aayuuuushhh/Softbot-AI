"""Regression tests for the shipped demo dataset.

These used to assert facts about the curated xBD set (ten pairs, ground-truth
target masks carrying class-3 and class-4 pixels). That set is gone: the demo
now ships before/after imagery without xBD labels or targets, so the things
worth guarding changed. What still matters is that the manifest, the files on
disk and what /demo/pairs discovers all agree — a pair listed but missing its
post image is a dropdown entry that 404s mid-demo.
"""

from __future__ import annotations

import json

from PIL import Image

from app.config import settings
from app.services.inference import list_demo_pairs


def _manifest_pairs() -> list[str]:
    manifest = json.loads(
        (settings.demo_data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return list(manifest["pairs"])


def test_demo_manifest_matches_discovered_pairs() -> None:
    discovered = {pair["id"] for pair in list_demo_pairs()}
    assert discovered, "Demo set must ship at least one pair"
    assert discovered == set(_manifest_pairs())


def test_every_manifest_pair_has_both_images() -> None:
    images_dir = settings.demo_data_dir / "images"
    for pair_id in _manifest_pairs():
        for side in ("pre", "post"):
            path = images_dir / f"{pair_id}_{side}_disaster.png"
            assert path.exists(), f"Missing {side} image for {pair_id}"


def test_pre_and_post_images_share_dimensions() -> None:
    """Damage is the difference between the two frames, so they must describe
    the same ground at the same size. A mismatched pair still runs — inference
    resizes post onto pre — but every zone box then lands slightly off the
    overlay the dashboard draws it on."""
    images_dir = settings.demo_data_dir / "images"
    for pair_id in _manifest_pairs():
        with Image.open(images_dir / f"{pair_id}_pre_disaster.png") as pre:
            pre_size = pre.size
        with Image.open(images_dir / f"{pair_id}_post_disaster.png") as post:
            post_size = post.size
        assert pre_size == post_size, f"{pair_id}: {pre_size} != {post_size}"
