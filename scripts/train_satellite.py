#!/usr/bin/env python3
"""Fine-tune SiameseChangeNet (S1) on xBD tile pairs.

    bash scripts/fetch_xbd.sh
    python scripts/train_satellite.py --xbd-root data/raw/xbd --epochs 30 --amp

Writes models/satellite_change.pt - the path the API's torch backend loads - and
a .metrics.json beside it. The best epoch is chosen by validation *building-level*
F1 on `destroyed`, the class CLAUDE.md gates on (> 0.80), not by loss.

Split discipline:
  * validation tiles come from xBD's train split, split by whole tile;
  * xBD's test split is never touched here - that is scripts/eval_xbd.py's job;
  * --holdout-events (default mexico-earthquake) are excluded from training so
    eval_xbd.py can report honest transfer to an unseen earthquake.

Sized for a 6 GB GPU: 512 px crops, batch 4, fp16 autocast. On CPU it runs, slowly;
use --max-tiles for a smoke run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings  # noqa: E402
from core.vision import metrics  # noqa: E402
from core.vision.datasets import (  # noqa: E402
    IGNORE_INDEX,
    XBDTileDataset,
    building_polygons,
    list_xbd_tiles,
    pixel_class_counts,
    split_tiles,
)
from core.vision.models import CHANGE_CLASSES, SiameseChangeNet  # noqa: E402
from core.vision.preprocess import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def normalise_batch(batch: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> ImageNet-normalised float, on whatever device it is on.
    Mirrors core.vision.preprocess.normalize, which inference uses."""
    mean = torch.as_tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch.float() / 255.0 - mean) / std


def class_weights(counts: np.ndarray, power: float = 0.5, cap: float = 50.0) -> torch.Tensor:
    """Inverse-frequency ** power, normalised so background weighs 1.

    Square-root damping: full inverse frequency on xBD puts ~1000x weight on
    `destroyed` pixels and the model learns to paint everything destroyed.
    """
    freq = counts.astype(np.float64) / max(counts.sum(), 1)
    freq = np.clip(freq, 1e-6, None)
    w = (freq[0] / freq) ** power
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, classes: list[int]) -> torch.Tensor:
    """Soft Dice over the damage classes only - it rewards overlap on rare
    classes that cross-entropy alone under-serves."""
    probs = logits.float().softmax(dim=1)
    valid = (target != IGNORE_INDEX).unsqueeze(1)
    safe_target = torch.where(target == IGNORE_INDEX, torch.zeros_like(target), target)
    onehot = F.one_hot(safe_target, probs.shape[1]).permute(0, 3, 1, 2).float()
    probs, onehot = probs * valid, onehot * valid
    losses = []
    for c in classes:
        inter = (probs[:, c] * onehot[:, c]).sum()
        denom = probs[:, c].sum() + onehot[:, c].sum()
        losses.append(1.0 - (2 * inter + 1.0) / (denom + 1.0))
    return torch.stack(losses).mean()


def evaluate(
    model, dataset: XBDTileDataset, device: str, amp: bool, tile_size: int
) -> tuple[dict, dict]:
    """Full-tile validation: pixel and building-level reports.

    Runs on whole tiles (sliding window) rather than crops so building votes
    see every footprint.
    """
    from core.vision.preprocess import tile_windows

    model.eval()
    n = len(CHANGE_CLASSES)
    pixel_cm = np.zeros((n, n), np.int64)
    building_cm = np.zeros((n, n), np.int64)

    with torch.inference_mode():
        for tile in dataset.tiles:
            pre, post, mask = dataset.load_full(tile)
            h, w = mask.shape
            logit_sum = torch.zeros((n, h, w), dtype=torch.float32)
            counts = torch.zeros((h, w), dtype=torch.float32)
            for x0, y0, x1, y1 in tile_windows(h, w, tile_size, tile_size // 8):
                p = torch.from_numpy(pre[y0:y1, x0:x1].transpose(2, 0, 1).copy())[None]
                q = torch.from_numpy(post[y0:y1, x0:x1].transpose(2, 0, 1).copy())[None]
                p, q = normalise_batch(p.to(device)), normalise_batch(q.to(device))
                with torch.autocast("cuda", enabled=amp and device == "cuda"):
                    out = model(p, q).float()[0].cpu()
                logit_sum[:, y0:y1, x0:x1] += out
                counts[y0:y1, x0:x1] += 1
            pred = (logit_sum / counts.clamp_min(1)).argmax(0).numpy().astype(np.uint8)

            pixel_cm += metrics.confusion(pred, mask, n, IGNORE_INDEX)
            bp, bt = metrics.building_votes(pred, building_polygons(tile.post_label))
            if bp:
                building_cm += metrics.confusion(np.array(bp), np.array(bt), n, IGNORE_INDEX)

    return metrics.report(pixel_cm, CHANGE_CLASSES), metrics.report(building_cm, CHANGE_CLASSES)


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--xbd-root", type=Path, default=Path("data/raw/xbd"))
    ap.add_argument(
        "--events",
        nargs="*",
        default=None,
        help="train only on these events (default: all present)",
    )
    ap.add_argument("--holdout-events", nargs="*", default=["mexico-earthquake"])
    ap.add_argument("--out", type=Path, default=settings.satellite_weights)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument(
        "--crops-per-tile",
        type=int,
        default=4,
        help="random crops drawn from each 1024 px tile per epoch",
    )
    ap.add_argument("--lr", type=float, default=3e-4, help="decoder LR")
    ap.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="encoder LR (default lr/10 - protects the ImageNet features)",
    )
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-tiles", type=int, default=None, help="cap tiles, for smoke runs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA")
    ap.add_argument(
        "--no-pretrained", action="store_true", help="random-init encoder (offline smoke runs only)"
    )
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2015)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or settings.resolved_device()

    tiles = list_xbd_tiles(args.xbd_root, "train", args.events, args.holdout_events)
    if not tiles:
        print(f"no xBD tiles under {args.xbd_root}/train - run scripts/fetch_xbd.sh")
        return 2
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]
    train_tiles, val_tiles = split_tiles(tiles, args.val_fraction, args.seed)
    events = sorted({t.event for t in tiles})
    print(f"device={device}  events={events}  held out={args.holdout_events}")
    print(f"tiles: {len(train_tiles)} train / {len(val_tiles)} val")

    counts = pixel_class_counts(train_tiles, seed=args.seed)
    weights = class_weights(counts)
    print("pixel counts:", dict(zip(CHANGE_CLASSES, counts.tolist(), strict=True)))
    print(
        "CE weights:  ",
        dict(zip(CHANGE_CLASSES, [round(float(w), 2) for w in weights], strict=True)),
    )

    train_set = XBDTileDataset(
        train_tiles, args.crop, augment=True, crops_per_tile=args.crops_per_tile, seed=args.seed
    )
    val_set = XBDTileDataset(val_tiles, args.crop, augment=False, seed=args.seed)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_set) > args.batch_size,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = SiameseChangeNet(pretrained=not args.no_pretrained).to(device)
    start_epoch, best = 1, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best = float(state.get("best_destroyed_f1", -1.0))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    encoder = [p for n, p in model.named_parameters() if n.startswith(("stem", "layer"))]
    decoder = [p for n, p in model.named_parameters() if not n.startswith(("stem", "layer"))]
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder, "lr": args.lr},
            {"params": encoder, "lr": args.encoder_lr or args.lr / 10},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce_weights = weights.to(device)
    damage_classes = [CHANGE_CLASSES.index(c) for c in ("minor", "major", "destroyed")]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for pre, post, mask in loader:
            pre = normalise_batch(pre.to(device, non_blocking=True))
            post = normalise_batch(post.to(device, non_blocking=True))
            mask = mask.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(pre, post)
            logits = logits.float()
            loss = F.cross_entropy(logits, mask, weight=ce_weights, ignore_index=IGNORE_INDEX)
            if args.dice_weight > 0:
                loss = loss + args.dice_weight * dice_loss(logits, mask, damage_classes)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss) * mask.shape[0]
            seen += mask.shape[0]
        scheduler.step()

        pixel, building = evaluate(model, val_set, device, use_amp, args.crop)
        destroyed = building["per_class"]["destroyed"]["f1"]
        score = -1.0 if destroyed is None else destroyed
        history.append(
            {
                "epoch": epoch,
                "loss": running / max(seen, 1),
                "building_destroyed_f1": destroyed,
                "building_macro_f1": building["macro_f1"],
                "pixel_macro_f1": pixel["macro_f1"],
            }
        )
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {running / max(seen, 1):.4f}  "
            f"bldg destroyed-F1 {destroyed}  bldg macro-F1 {building['macro_f1']}  "
            f"pixel macro-F1 {pixel['macro_f1']}  ({time.time() - t0:.0f}s)"
        )

        if score >= best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": CHANGE_CLASSES,
                    "epoch": epoch,
                    "best_destroyed_f1": best,
                    "events": events,
                    "crop": args.crop,
                },
                args.out,
            )
            args.out.with_suffix(".metrics.json").write_text(
                json.dumps(
                    {
                        "split": "validation (tiles from xBD train split)",
                        "epoch": epoch,
                        "pixel": pixel,
                        "building": building,
                        "gate_destroyed_f1": metrics.DESTROYED_F1_GATE,
                        "passes_gate": metrics.passes_gate(building),
                        "history": history,
                    },
                    indent=2,
                )
            )
            print(f"  saved {args.out}")

    print(f"\nbest validation building destroyed-F1: {best:.4f} (gate {metrics.DESTROYED_F1_GATE})")
    print(
        "validation is optimistic - run scripts/eval_xbd.py for the test split "
        "and the earthquake hold-out."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
