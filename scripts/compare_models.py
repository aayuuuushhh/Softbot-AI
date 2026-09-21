"""Compare TF docker baseline vs PyTorch fine-tuned masks (building IoU vs ground truth)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.config import settings  # noqa: E402
from app.services.inference import (  # noqa: E402
    resolve_demo_target,
    run_docker_inference,
    run_pytorch_inference,
)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    building_a = a > 0
    building_b = b > 0
    union = building_a | building_b
    if not union.any():
        return 1.0
    return float((building_a & building_b).sum() / union.sum())


def eval_pair(pair_id: str, modes: list[str]) -> dict:
    pre = settings.demo_data_dir / "images" / f"{pair_id}_pre_disaster.png"
    post = settings.demo_data_dir / "images" / f"{pair_id}_post_disaster.png"
    gt_path = resolve_demo_target(post)
    if not pre.exists() or not post.exists():
        return {"pair_id": pair_id, "error": "missing images"}
    gt = np.array(Image.open(gt_path).convert("L")) if gt_path and gt_path.exists() else None

    out_dir = settings.output_dir / "model_compare" / pair_id
    out_dir.mkdir(parents=True, exist_ok=True)
    row: dict = {"pair_id": pair_id}

    if "docker" in modes:
        try:
            pred_path = run_docker_inference(pre, post, out_dir / "docker")
            pred = np.array(Image.open(pred_path).convert("L"))
            row["docker_iou"] = round(mask_iou(pred, gt), 3) if gt is not None else None
        except Exception as exc:  # noqa: BLE001
            row["docker_error"] = str(exc)

    if "pytorch" in modes:
        try:
            pred_path, _ = run_pytorch_inference(pre, post, out_dir / "pytorch")
            pred = np.array(Image.open(pred_path).convert("L"))
            row["pytorch_iou"] = round(mask_iou(pred, gt), 3) if gt is not None else None
        except Exception as exc:  # noqa: BLE001
            row["pytorch_error"] = str(exc)

    if gt is not None:
        row["gt_destroyed_pct"] = round(float((gt == 4).sum() / max(1, (gt > 0).sum())) * 100, 2)
    return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["docker"], choices=["docker", "pytorch"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((settings.demo_data_dir / "manifest.json").read_text())
    pair_ids = manifest["pairs"] if isinstance(manifest, dict) else manifest
    if args.limit > 0:
        pair_ids = pair_ids[: args.limit]

    rows = [eval_pair(pid, args.modes) for pid in pair_ids]
    report = settings.output_dir / "model_compare" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for r in rows:
        print(r)
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
