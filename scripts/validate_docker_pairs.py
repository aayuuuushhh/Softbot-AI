"""Validate all demo pairs through docker inference and compare to ground truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.config import settings  # noqa: E402
from app.services.inference import resolve_demo_target, run_docker_inference  # noqa: E402

from compare_models import mask_iou  # noqa: E402


def main() -> None:
  manifest = json.loads((settings.demo_data_dir / "manifest.json").read_text())
  pair_ids = manifest["pairs"] if isinstance(manifest, dict) else manifest
  out_root = settings.output_dir / "docker_validation"
  out_root.mkdir(parents=True, exist_ok=True)

  rows: list[str] = []
  for pair_id in pair_ids:
    pre = settings.demo_data_dir / "images" / f"{pair_id}_pre_disaster.png"
    post = settings.demo_data_dir / "images" / f"{pair_id}_post_disaster.png"
    target = resolve_demo_target(post)
    if not pre.exists() or not post.exists():
      rows.append(f"{pair_id}: SKIP missing images")
      continue

    job_dir = out_root / pair_id
    print(f"Running docker inference: {pair_id} ...", flush=True)
    try:
      pred_path = run_docker_inference(pre, post, job_dir)
    except Exception as exc:  # noqa: BLE001
      rows.append(f"{pair_id}: FAIL {exc}")
      continue

    pred = np.array(Image.open(pred_path).convert("L"))
    if target and target.exists():
      gt = np.array(Image.open(target).convert("L"))
      iou = mask_iou(pred, gt)
      rows.append(f"{pair_id}: OK iou_building={iou:.3f} pred={pred_path.name}")
    else:
      rows.append(f"{pair_id}: OK no_gt pred={pred_path.name}")

  report = out_root / "report.txt"
  report.write_text("\n".join(rows) + "\n", encoding="utf-8")
  print("\n".join(rows))
  print(f"\nWrote {report}")


if __name__ == "__main__":
  main()
