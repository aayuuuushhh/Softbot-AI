#!/usr/bin/env python3
"""Score a satellite checkpoint on xBD data it never trained on.

    python scripts/eval_xbd.py --xbd-root data/raw/xbd

Two sets, reported separately - the gap between them is the finding:

  test     xBD's own test split, events the model has seen   -> tile generalisation
  holdout  --holdout-events (mexico-earthquake) from both splits -> seismic transfer

Exit status is the CLAUDE.md accuracy gate: 0 if building-level F1 on
`destroyed` exceeds 0.80 on the test set, 1 otherwise. The holdout number is the
one to quote for Nepal; expect it to be lower and say so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings  # noqa: E402
from core.vision import metrics  # noqa: E402
from core.vision.datasets import XBDTileDataset, list_xbd_tiles  # noqa: E402
from core.vision.models import SiameseChangeNet, load_checkpoint  # noqa: E402
from scripts.train_satellite import evaluate  # noqa: E402


def summarise(name: str, building: dict, pixel: dict) -> None:
    d = building["per_class"]["destroyed"]
    print(
        f"\n[{name}]  buildings: destroyed F1 {d['f1']} (n={d['support']})  "
        f"macro F1 {building['macro_f1']}  |  pixel macro F1 {pixel['macro_f1']}"
    )
    for cls, row in building["per_class"].items():
        if cls != "background":
            print(f"    {cls:10s} F1 {row['f1']}  support {row['support']}")


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--xbd-root", type=Path, default=Path("data/raw/xbd"))
    ap.add_argument("--checkpoint", type=Path, default=settings.satellite_weights)
    ap.add_argument("--holdout-events", nargs="*", default=["mexico-earthquake"])
    ap.add_argument("--tile-size", type=int, default=settings.tile_size)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or settings.resolved_device()
    model = SiameseChangeNet(pretrained=False)
    if not load_checkpoint(model, args.checkpoint, device):
        print(f"no usable checkpoint at {args.checkpoint}")
        return 2
    model.to(device)

    test = list_xbd_tiles(args.xbd_root, "test", exclude_events=args.holdout_events)
    holdout = list_xbd_tiles(args.xbd_root, "train", events=args.holdout_events) + list_xbd_tiles(
        args.xbd_root, "test", events=args.holdout_events
    )
    if args.max_tiles:
        test, holdout = test[: args.max_tiles], holdout[: args.max_tiles]

    results: dict = {"checkpoint": str(args.checkpoint)}
    for name, tiles in (("test", test), ("holdout", holdout)):
        if not tiles:
            print(f"[{name}] no tiles found - skipped")
            continue
        pixel, building = evaluate(model, XBDTileDataset(tiles), device, args.amp, args.tile_size)
        summarise(f"{name}: {len(tiles)} tiles", building, pixel)
        results[name] = {"tiles": len(tiles), "pixel": pixel, "building": building}

    out = args.checkpoint.with_suffix(".eval.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    if "test" not in results:
        return 2
    ok = metrics.passes_gate(results["test"]["building"])
    print(
        f"gate (test building destroyed-F1 > {metrics.DESTROYED_F1_GATE}): "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
