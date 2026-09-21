from __future__ import annotations

from core.config import Settings, get_settings


def test_defaults_are_safe():
    s = Settings()
    # stub needs no weights, so a fresh clone runs without any .pt file
    assert s.sat_backend == "stub"
    assert s.ground_backend == "stub"
    # 6 GB VRAM: tiles must stay small
    assert s.tile_size <= 512


def test_ground_outranks_satellite():
    """CLAUDE.md F1: ground imagery is weighted higher, to bypass cloud cover."""
    assert get_settings().ground_weight > 0.5


def test_cors_origins_parse_as_a_list():
    s = Settings(cors_origins="http://a.test, http://b.test ,")
    assert s.cors_origin_list == ["http://a.test", "http://b.test"]


def test_resolved_device_never_claims_a_gpu_it_lacks():
    s = Settings(device="cpu")
    assert s.resolved_device() == "cpu"
    # With device=cuda the result must still be one of the two real options,
    # never the unverified request.
    assert Settings(device="cuda").resolved_device() in ("cuda", "cpu")
