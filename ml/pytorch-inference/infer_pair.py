"""Run PyTorch xView2 damage inference on a single pre/post pair.

Requires a trained post-disaster checkpoint (siamese, focal+dice).
Creates a temporary holdout layout and shells out to michal2409/xView2 main.py.

Per-zone confidence (mean predicted-class probability) is extracted from the
post-softmax damage probs saved by the model and written as a sibling
``{mask_stem}_confidence.npy``. This applies only to PyTorch inference; the
Docker TF1.15 baseline and stub modes emit label masks only (no probabilities).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"
MAIN_PY = XVIEW2_ROOT / "main.py"


def _dummy_mask(path: Path) -> None:
    Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8)).save(path)


def probs_to_mask_and_confidence(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert model output to class mask and per-pixel confidence map."""
    if arr.ndim == 3:
        mask = np.argmax(arr, axis=0).astype(np.uint8)
        # arr is already post-softmax (model/plt.py) — do NOT re-apply softmax
        confidence = np.take_along_axis(arr, mask[None, :, :], axis=0)[0].astype(np.float32)
        return np.clip(mask, 0, 4), confidence
    mask = np.clip(np.round(arr).astype(np.uint8), 0, 4)
    return mask, None


def _localize_checkpoint(checkpoint: Path, results_dir: Path) -> Path:
    """Rewrite Kaggle paths baked into checkpoint hyperparameters for local eval."""
    import torch
    from argparse import Namespace

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    hp = state.get("hyper_parameters")
    if not isinstance(hp, dict):
        return checkpoint

    args = hp.get("args")
    if isinstance(args, Namespace) and str(getattr(args, "results", "")).startswith("/kaggle"):
        localized_args = Namespace(**{**vars(args), "results": str(results_dir)})
        state["hyper_parameters"] = {**hp, "args": localized_args}
        localized = results_dir / "_localized_checkpoint.ckpt"
        torch.save(state, localized)
        return localized
    return checkpoint


def _run_xview2_eval(data_root: Path, results: Path, ckpt_path: Path) -> None:
    """In-process eval: load checkpoint and run one forward pass (no PL Trainer)."""
    import torch

    if str(XVIEW2_ROOT) not in sys.path:
        sys.path.insert(0, str(XVIEW2_ROOT))

    _orig = torch.load

    def _torch_load_trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _torch_load_trusted

    from data_loading.data_module import DataModule
    from model.plt import Model

    model = Model.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.args.data = str(data_root)
    model.args.results = str(results)
    model.args.gpus = 0
    model.args.precision = 32
    model.args.num_workers = 0
    model.args.val_batch_size = 1
    model.eval()

    (results / "probs").mkdir(parents=True, exist_ok=True)
    (results / "targets").mkdir(parents=True, exist_ok=True)

    loader = DataModule(model.args).test_dataloader()
    with torch.no_grad():
        for batch in loader:
            pred = model.forward(batch["image"])
            model.save(pred, batch["mask"])


def infer_pair(
    pre_path: Path,
    post_path: Path,
    checkpoint: Path,
    out_mask: Path,
    work_dir: Path | None = None,
) -> Path:
    if not MAIN_PY.exists():
        raise FileNotFoundError(f"Missing {MAIN_PY} — clone michal2409/xView2 into ml/pytorch-xview2")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    job_id = uuid.uuid4().hex[:8]
    work = work_dir or (REPO_ROOT / "backend" / "app" / "outputs" / f"pytorch_{job_id}")
    data_root = work / "data"
    holdout = data_root / "holdout"
    img_dir = holdout / "images"
    tgt_dir = holdout / "targets"
    results = work / "results"
    for d in (img_dir, tgt_dir, results):
        d.mkdir(parents=True, exist_ok=True)

    stem = post_path.stem.replace("_post_disaster", "")
    pre_name = f"{stem}_pre_disaster.png"
    post_name = f"{stem}_post_disaster.png"
    shutil.copy2(pre_path, img_dir / pre_name)
    shutil.copy2(post_path, img_dir / post_name)
    _dummy_mask(tgt_dir / pre_name)
    _dummy_mask(tgt_dir / post_name)

    ckpt_for_eval = _localize_checkpoint(checkpoint, results)

    index_src = XVIEW2_ROOT / "utils" / "index.csv"
    if index_src.exists():
        os.environ["XVIEW2_INDEX_CSV"] = str(index_src.resolve())

    _run_xview2_eval(data_root, results, ckpt_for_eval)

    probs_dir = results / "probs"
    npy_files = sorted(probs_dir.glob("test_damage_*.npy"))
    if not npy_files:
        raise RuntimeError(f"No damage predictions in {probs_dir}")

    arr = np.load(npy_files[0])
    mask, confidence = probs_to_mask_and_confidence(arr)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_mask)
    if confidence is not None:
        np.save(out_mask.parent / f"{out_mask.stem}_confidence.npy", confidence)
    return out_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = infer_pair(args.pre, args.post, args.checkpoint, args.out)
    print(f"Wrote mask: {out}")


if __name__ == "__main__":
    main()
