from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services import inference


def _write_png(path: Path, value: int = 0, size: tuple[int, int] = (32, 32)) -> Path:
    Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8)).save(path)
    return path


def test_stub_never_serves_the_ground_truth_target(tmp_path, monkeypatch):
    """A pair that ships an xBD label must still be *inferred*, not recited.

    Copying the target out as the prediction made demo pairs score perfectly
    while unseen uploads fell back to the heuristic — an answer key, not a
    model. The labels stay in the repo for ranking and offline evaluation only.
    """
    demo = tmp_path / "demo"
    (demo / "targets").mkdir(parents=True)
    monkeypatch.setattr(inference.settings, "demo_data_dir", demo)
    monkeypatch.setattr(inference.settings, "test_data_dir", tmp_path / "missing")
    monkeypatch.setattr(inference.settings, "inference_mode", "stub")

    # A target the heuristic could never invent: a solid block of class 1.
    target = _write_png(demo / "targets" / "quake_00000001_post_disaster_target.png", value=1)
    pre = _write_png(tmp_path / "quake_00000001_pre_disaster.png", value=10)
    post = _write_png(tmp_path / "quake_00000001_post_disaster.png", value=200)

    mask_path, mode, confidence = inference.run_inference(pre, post, tmp_path / "out")

    assert mode == "stub-heuristic"
    assert confidence is None
    mask = np.array(Image.open(mask_path))
    assert not np.array_equal(mask, np.array(Image.open(target)))
    # Class 1 only exists in the label file; the heuristic cannot emit it.
    assert set(np.unique(mask)) <= {0, 2, 3, 4}


def test_stub_falls_back_to_diff_heuristic(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.settings, "demo_data_dir", tmp_path / "no_demo")
    monkeypatch.setattr(inference.settings, "test_data_dir", tmp_path / "no_test")
    monkeypatch.setattr(inference.settings, "inference_mode", "stub")

    pre = _write_png(tmp_path / "pre.png", value=0)
    post = tmp_path / "post.png"
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[8:24, 8:24] = 255  # a big bright change the opening filter will keep
    Image.fromarray(arr).save(post)

    mask_path, mode, confidence = inference.run_inference(pre, post, tmp_path / "out")

    assert mode == "stub-heuristic"
    assert confidence is None
    mask = np.array(Image.open(mask_path))
    assert set(np.unique(mask)) <= {0, 2, 3, 4}
    assert (mask > 0).any(), "a 16x16 solid change must survive thresholding"


def test_diff_heuristic_survives_large_change_regions(tmp_path):
    """A scene where most of the frame changed must not threshold to nothing.

    The percentile otherwise lands inside the damaged region, so `diff > threshold`
    excludes it and a catastrophically damaged tile reports zero damage.
    """
    pre = _write_png(tmp_path / "pre.png", value=0)
    post = _write_png(tmp_path / "post.png", value=255)

    mask = np.array(Image.open(inference._diff_mask(pre, post, tmp_path / "m.png")))

    assert (mask > 0).all()


def test_diff_heuristic_reports_nothing_when_images_match(tmp_path):
    pre = _write_png(tmp_path / "pre.png", value=120)
    post = _write_png(tmp_path / "post.png", value=120)

    mask = np.array(Image.open(inference._diff_mask(pre, post, tmp_path / "m.png")))

    assert not (mask > 0).any()


def test_diff_heuristic_removes_single_pixel_speckle(tmp_path):
    """Isolated pixels are noise, not buildings — opening must erase them."""
    pre = _write_png(tmp_path / "pre.png", value=0)
    post = tmp_path / "post.png"
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[3, 3] = 255
    arr[20, 25] = 255
    Image.fromarray(arr).save(post)

    mask = np.array(Image.open(inference._diff_mask(pre, post, tmp_path / "m.png")))

    assert not (mask > 0).any()


def test_diff_heuristic_resizes_mismatched_post(tmp_path):
    pre = _write_png(tmp_path / "pre.png", size=(32, 32))
    post = _write_png(tmp_path / "post.png", size=(16, 16))

    mask = np.array(Image.open(inference._diff_mask(pre, post, tmp_path / "m.png")))

    assert mask.shape == (32, 32)


def test_diff_heuristic_aligns_shifted_pair(tmp_path):
    """A shifted capture of the same scene must not read as wall-to-wall damage.

    Registration should cancel the offset, so a translated copy of a textured
    frame produces almost no damage — versus the near-total mask a raw diff
    would yield.
    """
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "pre.png")
    shifted = np.roll(base, shift=(3, 4), axis=(0, 1))
    Image.fromarray(shifted).save(tmp_path / "post.png")

    mask = np.array(
        Image.open(inference._diff_mask(tmp_path / "pre.png", tmp_path / "post.png", tmp_path / "m.png"))
    )

    assert (mask > 0).mean() < 0.05


def test_diff_heuristic_ignores_global_illumination(tmp_path):
    """A uniform brightness lift on a textured scene is lighting, not damage."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 180, size=(96, 96), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "pre.png")
    Image.fromarray((base.astype(np.int16) + 25).astype(np.uint8)).save(tmp_path / "post.png")

    mask = np.array(
        Image.open(inference._diff_mask(tmp_path / "pre.png", tmp_path / "post.png", tmp_path / "m.png"))
    )

    assert (mask > 0).mean() < 0.02


def test_diff_heuristic_detects_real_change_under_illumination(tmp_path):
    """Illumination correction must not hide a genuine localized change."""
    rng = np.random.default_rng(2)
    base = rng.integers(0, 180, size=(96, 96), dtype=np.uint8)
    Image.fromarray(base).save(tmp_path / "pre.png")
    post = (base.astype(np.int16) + 25).astype(np.uint8).copy()
    post[30:60, 30:60] = 255  # a real 30x30 change on top of the lighting shift
    Image.fromarray(post).save(tmp_path / "post.png")

    mask = np.array(
        Image.open(inference._diff_mask(tmp_path / "pre.png", tmp_path / "post.png", tmp_path / "m.png"))
    )

    assert mask[30:60, 30:60].mean() > 0  # the patch is flagged
    # and the untouched surroundings stay mostly clear
    outside = mask.copy()
    outside[30:60, 30:60] = 0
    assert (outside > 0).mean() < 0.02


def test_diff_heuristic_drops_subbuilding_blobs(tmp_path):
    """Change regions smaller than the min-building area are removed as noise.

    On a 1200x1200 frame the min-building area is ~22 px, so a 4x4 blob (which
    survives the 3x3 opening) is filtered, while a large block is kept.
    """
    pre = _write_png(tmp_path / "pre.png", value=0, size=(1200, 1200))
    arr = np.zeros((1200, 1200), dtype=np.uint8)
    arr[10:14, 10:14] = 255  # 4x4 = 16 px, below the ~22 px min-building area
    Image.fromarray(arr).save(tmp_path / "post.png")

    mask = np.array(
        Image.open(inference._diff_mask(tmp_path / "pre.png", tmp_path / "post.png", tmp_path / "m.png"))
    )
    assert not (mask > 0).any()

    arr[100:140, 100:140] = 255  # add a 40x40 block, clearly a building-scale change
    Image.fromarray(arr).save(tmp_path / "post.png")
    mask = np.array(
        Image.open(inference._diff_mask(tmp_path / "pre.png", tmp_path / "post.png", tmp_path / "m.png"))
    )
    assert (mask > 0).any()
    assert not (mask[10:14, 10:14] > 0).any()  # small blob still filtered


def test_unsupported_mode_raises_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.settings, "inference_mode", "quantum")
    pre = _write_png(tmp_path / "pre.png")
    post = _write_png(tmp_path / "post.png")

    with pytest.raises(RuntimeError, match="Unsupported inference_mode"):
        inference.run_inference(pre, post, tmp_path / "out")


def test_pytorch_mode_requires_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.settings, "inference_mode", "pytorch")
    monkeypatch.setattr(inference.settings, "pytorch_checkpoint_path", tmp_path / "absent.ckpt")
    pre = _write_png(tmp_path / "pre.png")
    post = _write_png(tmp_path / "post.png")

    with pytest.raises(RuntimeError, match="checkpoint not found"):
        inference.run_inference(pre, post, tmp_path / "out")


def test_pytorch_mode_reports_subprocess_failure(tmp_path, monkeypatch):
    ckpt = tmp_path / "damage_best.ckpt"
    ckpt.write_bytes(b"fake")
    infer_dir = tmp_path / "pytorch-inference"
    infer_dir.mkdir()
    (infer_dir / "infer_pair.py").write_text("raise SystemExit(1)")

    monkeypatch.setattr(inference.settings, "inference_mode", "pytorch")
    monkeypatch.setattr(inference.settings, "pytorch_checkpoint_path", ckpt)
    monkeypatch.setattr(inference.settings, "pytorch_inference_dir", infer_dir)
    monkeypatch.setattr(
        inference.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "CUDA out of memory"),
    )

    pre = _write_png(tmp_path / "pre.png")
    post = _write_png(tmp_path / "post.png")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        inference.run_inference(pre, post, tmp_path / "out")


def test_pytorch_mode_returns_confidence_when_written(tmp_path, monkeypatch):
    ckpt = tmp_path / "damage_best.ckpt"
    ckpt.write_bytes(b"fake")
    infer_dir = tmp_path / "pytorch-inference"
    infer_dir.mkdir()
    (infer_dir / "infer_pair.py").write_text("")
    out_dir = tmp_path / "out"

    def _fake_run(cmd, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = out_dir / "damage_mask.png"
        _write_png(mask, value=2)
        np.save(inference.confidence_path_for_mask(mask), np.full((32, 32), 0.9, dtype=np.float32))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(inference.settings, "inference_mode", "pytorch")
    monkeypatch.setattr(inference.settings, "pytorch_checkpoint_path", ckpt)
    monkeypatch.setattr(inference.settings, "pytorch_inference_dir", infer_dir)
    monkeypatch.setattr(inference.subprocess, "run", _fake_run)

    pre = _write_png(tmp_path / "pre.png")
    post = _write_png(tmp_path / "post.png")

    mask_path, mode, confidence = inference.run_inference(pre, post, out_dir)

    assert mode == "pytorch"
    assert confidence is not None and confidence.exists()
    assert mask_path.exists()


def test_list_demo_pairs_excludes_uploads(tmp_path, monkeypatch):
    """Uploaded images must never surface as demo pairs."""
    demo = tmp_path / "demo" / "images"
    demo.mkdir(parents=True)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    _write_png(uploads / "secret_pre_disaster.png")
    _write_png(uploads / "secret_post_disaster.png")

    monkeypatch.setattr(inference.settings, "demo_data_dir", tmp_path / "demo")
    monkeypatch.setattr(inference.settings, "upload_dir", uploads)

    assert inference.list_demo_pairs() == []


def test_list_demo_pairs_ranks_damaged_first(tmp_path, monkeypatch):
    """The dashboard defaults to the first pair, so it must not be an undamaged one.

    Several real xBD scenes contain no damage at all. Alphabetical order put one
    of them first, making the default analysis return an all-zero result.
    """
    demo = tmp_path / "demo"
    images = demo / "images"
    targets = demo / "targets"
    images.mkdir(parents=True)
    targets.mkdir(parents=True)

    monkeypatch.setattr(inference.settings, "demo_data_dir", demo)
    monkeypatch.setattr(inference.settings, "test_data_dir", tmp_path / "missing")

    # "aaa" sorts first alphabetically but is entirely undamaged (class 1);
    # "zzz" is destroyed (class 4) and is the only pair worth defaulting to.
    for base, target_class in (("aaa-quake_00000001", 1), ("zzz-quake_00000002", 4)):
        _write_png(images / f"{base}_pre_disaster.png")
        _write_png(images / f"{base}_post_disaster.png")
        _write_png(targets / f"{base}_post_disaster_target.png", value=target_class)

    ids = [p["id"] for p in inference.list_demo_pairs()]

    assert ids == ["zzz-quake_00000002", "aaa-quake_00000001"]
