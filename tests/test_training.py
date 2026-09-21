"""Training-pipeline tests.

The real run needs the 3 GB xBD download and a GPU; these build a miniature xBD
tree on disk and drive the actual scripts end to end on CPU, so the data
contract, the loss, checkpointing and the checkpoint -> API loading path are
all exercised before anyone spends GPU hours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.vision import metrics  # noqa: E402
from core.vision.datasets import (  # noqa: E402
    IGNORE_INDEX,
    XBDTileDataset,
    building_polygons,
    list_ground_photos,
    list_xbd_tiles,
    rasterize_label,
    split_tiles,
)
from core.vision.models import (  # noqa: E402
    CHANGE_CLASSES,
    GroundDamageNet,
    SiameseChangeNet,
    load_checkpoint,
)

SIZE = 128


def _square_wkt(x0, y0, s):
    return f"POLYGON (({x0} {y0}, {x0 + s} {y0}, {x0 + s} {y0 + s}, {x0} {y0 + s}, {x0} {y0}))"


def _write_tile(root: Path, split: str, stem: str, buildings: list[tuple[int, int, str]]):
    images, labels = root / split / "images", root / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % 2**32)
    pre = np.full((SIZE, SIZE, 3), (60, 110, 60), np.uint8) + rng.integers(
        0, 10, (SIZE, SIZE, 3), dtype=np.uint8
    )
    post = pre.copy()
    feats = []
    for x0, y0, subtype in buildings:
        pre[y0 : y0 + 20, x0 : x0 + 20] = (200, 200, 190)
        post[y0 : y0 + 20, x0 : x0 + 20] = (
            (90, 80, 70) if subtype == "destroyed" else (200, 200, 190)
        )
        feats.append({"properties": {"subtype": subtype}, "wkt": _square_wkt(x0, y0, 20)})
    cv2.imwrite(str(images / f"{stem}_pre_disaster.png"), cv2.cvtColor(pre, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(images / f"{stem}_post_disaster.png"), cv2.cvtColor(post, cv2.COLOR_RGB2BGR))
    (labels / f"{stem}_post_disaster.json").write_text(json.dumps({"features": {"xy": feats}}))
    (labels / f"{stem}_pre_disaster.json").write_text(json.dumps({"features": {"xy": []}}))


@pytest.fixture
def fake_xbd(tmp_path) -> Path:
    root = tmp_path / "xbd"
    for i in range(4):
        _write_tile(
            root,
            "train",
            f"palu-tsunami_{i:08d}",
            [(10, 10, "destroyed"), (60, 60, "no-damage"), (90, 20, "un-classified")],
        )
    _write_tile(root, "train", "mexico-earthquake_00000000", [(30, 30, "no-damage")])
    _write_tile(
        root, "test", "palu-tsunami_00000100", [(20, 20, "destroyed"), (70, 70, "minor-damage")]
    )
    return root


# --------------------------------------------------------------------------
# dataset contract
# --------------------------------------------------------------------------


def test_masks_use_the_inference_class_indices(fake_xbd):
    label = fake_xbd / "train" / "labels" / "palu-tsunami_00000000_post_disaster.json"
    mask = rasterize_label(label, SIZE, SIZE)
    assert mask[15, 15] == CHANGE_CLASSES.index("destroyed")
    assert mask[65, 65] == CHANGE_CLASSES.index("none")
    assert mask[25, 95] == IGNORE_INDEX, "un-classified must be ignored, not trained on"
    assert mask[0, 0] == CHANGE_CLASSES.index("background")


def test_holdout_events_are_excluded_from_training_tiles(fake_xbd):
    tiles = list_xbd_tiles(fake_xbd, "train", exclude_events=["mexico-earthquake"])
    assert tiles and all(t.event == "palu-tsunami" for t in tiles)


def test_split_is_by_whole_tile_and_disjoint(fake_xbd):
    tiles = list_xbd_tiles(fake_xbd, "train")
    train, val = split_tiles(tiles, 0.4, seed=1)
    assert {t.stem for t in train}.isdisjoint({t.stem for t in val})
    assert len(train) + len(val) == len(tiles)


def test_augmentation_keeps_pre_post_and_mask_registered(fake_xbd):
    """Flip one image and not the other and you've invented damage."""
    tiles = list_xbd_tiles(fake_xbd, "train", exclude_events=["mexico-earthquake"])
    ds = XBDTileDataset(tiles, crop=SIZE, augment=True)
    for i in range(8):
        pre, _post, mask = ds[i % len(ds)]
        destroyed = (mask == CHANGE_CLASSES.index("destroyed")).numpy()
        assert destroyed.sum() > 0
        # Every destroyed pixel sits on a bright pre-event roof, wherever the
        # augmentation moved it.
        assert pre.numpy()[:, destroyed].mean() > 150


def test_building_votes_score_each_footprint_once(fake_xbd):
    label = fake_xbd / "train" / "labels" / "palu-tsunami_00000000_post_disaster.json"
    polys = building_polygons(label)
    truth = rasterize_label(label, SIZE, SIZE)
    preds, truths = metrics.building_votes(truth, polys)
    assert len(preds) == 2  # un-classified footprint skipped
    assert preds == truths


def test_f1_ignores_absent_classes_rather_than_scoring_them_zero():
    cm = metrics.confusion(np.array([4, 4, 1]), np.array([4, 4, 1]), len(CHANGE_CLASSES))
    rep = metrics.report(cm, CHANGE_CLASSES)
    assert rep["per_class"]["destroyed"]["f1"] == 1.0
    assert rep["per_class"]["major"]["f1"] is None
    assert metrics.passes_gate(rep)


# --------------------------------------------------------------------------
# end-to-end script runs
# --------------------------------------------------------------------------


def _run(module_path: str, argv: list[str], monkeypatch) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location("script_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["script", *argv])
    return module.main()


ROOT = Path(__file__).resolve().parent.parent


def test_train_satellite_writes_a_checkpoint_the_api_can_load(fake_xbd, tmp_path, monkeypatch):
    out = tmp_path / "satellite_change.pt"
    rc = _run(
        str(ROOT / "scripts" / "train_satellite.py"),
        [
            "--xbd-root",
            str(fake_xbd),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--crop",
            "128",
            "--crops-per-tile",
            "1",
            "--workers",
            "0",
            "--no-pretrained",
            "--device",
            "cpu",
            "--val-fraction",
            "0.25",
            "--out",
            str(out),
        ],
        monkeypatch,
    )
    assert rc == 0
    assert out.exists()
    report = json.loads(out.with_suffix(".metrics.json").read_text())
    assert "destroyed" in report["building"]["per_class"]

    model = SiameseChangeNet(pretrained=False)
    assert load_checkpoint(model, out, "cpu"), "the API loader must accept the trainer's format"


def test_eval_xbd_scores_test_and_holdout_and_returns_the_gate(fake_xbd, tmp_path, monkeypatch):
    ckpt = tmp_path / "sat.pt"
    torch.save({"model": SiameseChangeNet(pretrained=False).state_dict()}, ckpt)
    rc = _run(
        str(ROOT / "scripts" / "eval_xbd.py"),
        [
            "--xbd-root",
            str(fake_xbd),
            "--checkpoint",
            str(ckpt),
            "--tile-size",
            "128",
            "--device",
            "cpu",
        ],
        monkeypatch,
    )
    results = json.loads(ckpt.with_suffix(".eval.json").read_text())
    assert {"test", "holdout"} <= set(results)
    assert rc in (0, 1)  # an untrained net fails the gate; the point is it ran


def test_train_ground_writes_a_checkpoint_the_api_can_load(tmp_path, monkeypatch):
    data = tmp_path / "ground"
    rng = np.random.default_rng(0)
    for cls in ("none", "destroyed"):
        (data / cls).mkdir(parents=True)
        for i in range(4):
            img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(data / cls / f"{i}.jpg"), img)
    assert len(list_ground_photos(data)) == 8

    out = tmp_path / "ground_damage.pt"
    rc = _run(
        str(ROOT / "scripts" / "train_ground.py"),
        [
            "--data",
            str(data),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--workers",
            "0",
            "--no-pretrained",
            "--device",
            "cpu",
            "--out",
            str(out),
        ],
        monkeypatch,
    )
    assert rc == 0
    model = GroundDamageNet(pretrained=False)
    assert load_checkpoint(model, out, "cpu")
