#!/usr/bin/env python3
"""Fine-tune GroundDamageNet (S2) on labelled field photos.

    python scripts/train_ground.py --data data/training/ground --epochs 20

Expected layout (one folder per GROUND_CLASSES entry; missing folders are fine
but a class with no photos cannot be learned):

    data/training/ground/
        none/       intact structures
        minor/      cracks, broken windows, partial roof loss
        major/      partial collapse, serious structural damage
        destroyed/  collapsed / rubble

Candidate public sources, to be checked for licence before use: the 2015 Nepal
earthquake housing reconstruction photo sets, Crisis Image Benchmark
(Alam et al.) damage-severity subset, and field photos collected via the
/api/upload/ground endpoint once labelled. Add Nepali masonry photos
specifically - that is the domain gap that matters.

Writes models/ground_damage.pt (the path the API loads) and .metrics.json.
Best epoch is chosen by F1 on the merged {major, destroyed} "damaged" call,
because that is the decision fusion acts on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings  # noqa: E402
from core.vision import metrics  # noqa: E402
from core.vision.datasets import (  # noqa: E402
    GroundPhotoDataset,
    list_ground_photos,
    stratified_split,
)
from core.vision.models import GROUND_CLASSES, GroundDamageNet  # noqa: E402


def damaged_f1(cm: np.ndarray) -> float:
    """F1 of the binary call {major, destroyed} vs {none, minor}."""
    damaged = [GROUND_CLASSES.index("major"), GROUND_CLASSES.index("destroyed")]
    truth_d = cm[damaged].sum()
    pred_d = cm[:, damaged].sum()
    tp = cm[np.ix_(damaged, damaged)].sum()
    return float(2 * tp / (truth_d + pred_d)) if truth_d + pred_d else 0.0


def run_eval(model, loader, device: str) -> np.ndarray:
    model.eval()
    n = len(GROUND_CLASSES)
    cm = np.zeros((n, n), np.int64)
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            cm += metrics.confusion(pred, y.numpy(), n)
    return cm


def main() -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=Path("data/training/ground"))
    ap.add_argument("--out", type=Path, default=settings.ground_weights)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR")
    ap.add_argument("--backbone-lr", type=float, default=None, help="default lr/10")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=2015)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or settings.resolved_device()

    items = list_ground_photos(args.data)
    if not items:
        print(
            f"no photos under {args.data}/<{'|'.join(GROUND_CLASSES)}>/ - see this script's docstring"
        )
        return 2
    per_class = np.bincount([c for _, c in items], minlength=len(GROUND_CLASSES))
    print(
        f"device={device}  photos per class:",
        dict(zip(GROUND_CLASSES, per_class.tolist(), strict=True)),
    )
    missing = [GROUND_CLASSES[i] for i, n in enumerate(per_class) if n == 0]
    if missing:
        print(f"WARNING: no photos for {missing}; the model can never predict them")

    train_items, val_items = stratified_split(items, args.val_fraction, args.seed)
    train_set = GroundPhotoDataset(train_items, augment=True)
    val_set = GroundPhotoDataset(val_items, augment=False)

    # Balance classes in every batch; field photo sets are dominated by rubble.
    labels = train_set.labels()
    counts = np.bincount(labels, minlength=len(GROUND_CLASSES)).clip(min=1)
    sampler = WeightedRandomSampler(
        [1.0 / counts[c] for c in labels], len(labels), replacement=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    model = GroundDamageNet(pretrained=not args.no_pretrained).to(device)
    head = list(model.net.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": head, "lr": args.lr},
            {"params": backbone, "lr": args.backbone_lr or args.lr / 10},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * y.shape[0]
            seen += y.shape[0]
        scheduler.step()

        cm = run_eval(model, val_loader, device) if len(val_set) else np.zeros((4, 4), np.int64)
        rep = metrics.report(cm, GROUND_CLASSES)
        score = damaged_f1(cm)
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {total / max(seen, 1):.4f}  "
            f"damaged-F1 {score:.3f}  macro-F1 {rep['macro_f1']}"
        )
        if score >= best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": GROUND_CLASSES,
                    "epoch": epoch,
                    "damaged_f1": score,
                },
                args.out,
            )
            args.out.with_suffix(".metrics.json").write_text(
                json.dumps({"epoch": epoch, "damaged_f1": score, "report": rep}, indent=2)
            )
            print(f"  saved {args.out}")

    print(f"\nbest validation damaged-F1: {best:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
