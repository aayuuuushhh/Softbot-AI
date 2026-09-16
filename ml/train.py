"""Train the siamese damage classifier on xBD chips.

    python -m ml.train --manifest data/processed/manifest.csv --epochs 8

Reports per-class F1 and a confusion matrix, never bare accuracy: on xBD a model that predicts
`no-damage` for everything scores ~80% and helps nobody.

See PROJECT_PLAN.md section 3.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .datasets.xbd import XBDChipDataset, class_counts
from .models.siamese import DAMAGE_CLASSES, SiameseDamageNet, class_weights_from_counts


def split_by_tile(manifest: Path, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split on the source *tile*, not the chip.

    Buildings from one tile look alike; splitting per chip leaks the validation set into
    training and gives a score that evaporates on real imagery.
    """
    import csv

    with open(manifest) as f:
        rows = list(csv.DictReader(f))

    tiles: dict[str, list[str]] = {}
    for row in rows:
        tile = row["chip_id"].rsplit("_", 1)[0]
        tiles.setdefault(tile, []).append(row["chip_id"])

    tile_names = sorted(tiles)
    random.Random(seed).shuffle(tile_names)
    cut = max(1, int(len(tile_names) * val_fraction))
    val_tiles, train_tiles = tile_names[:cut], tile_names[cut:]

    train_ids = [cid for t in train_tiles for cid in tiles[t]]
    val_ids = [cid for t in val_tiles for cid in tiles[t]]
    return train_ids, val_ids


def make_sampler(dataset: XBDChipDataset) -> WeightedRandomSampler:
    """Oversample the rare damage classes so a batch actually contains some damage."""
    labels = dataset.labels()
    counts = np.bincount(labels, minlength=len(DAMAGE_CLASSES)).clip(min=1)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, predictions, targets = 0.0, [], []

    for pre, post, label in tqdm(loader, leave=False, desc="train" if train else "val"):
        pre, post, label = pre.to(device), post.to(device), label.to(device)

        with torch.set_grad_enabled(train):
            logits = model(pre, post)
            loss = criterion(logits, label)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * label.size(0)
        predictions.extend(logits.argmax(1).cpu().tolist())
        targets.extend(label.cpu().tolist())

    return total_loss / max(1, len(targets)), predictions, targets


def macro_f1(targets, predictions) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(targets, predictions, average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("ml/checkpoints/siamese_resnet18.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2015)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only the head - much faster, use if the clock is short")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"No manifest at {args.manifest}. Run `python scripts/run_pipeline.py --prepare` "
            "first (see data/README.md)."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    counts = class_counts(args.manifest)
    print("class balance:", counts)

    train_ids, val_ids = split_by_tile(args.manifest, args.val_fraction, args.seed)
    train_set = XBDChipDataset(args.manifest, train_ids, augment=True)
    val_set = XBDChipDataset(args.manifest, val_ids, augment=False)
    print(f"train chips: {len(train_set)}  val chips: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=make_sampler(train_set),
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    model = SiameseDamageNet(freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(counts).to(device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, predictions, targets = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()

        f1 = macro_f1(targets, predictions)
        print(f"epoch {epoch}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  macro-F1 {f1:.4f}")

        if f1 >= best_f1:
            best_f1 = f1
            torch.save(
                {"model_state": model.state_dict(), "classes": DAMAGE_CLASSES,
                 "macro_f1": f1, "epoch": epoch, "class_counts": counts},
                args.out,
            )
            print(f"  saved {args.out} (macro-F1 {f1:.4f})")

    print("\n" + classification_report(
        targets, predictions, target_names=DAMAGE_CLASSES, zero_division=0, digits=3
    ))
    print("confusion matrix (rows = truth, cols = predicted)")
    print(DAMAGE_CLASSES)
    print(confusion_matrix(targets, predictions, labels=list(range(len(DAMAGE_CLASSES)))))

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "best_macro_f1": best_f1,
        "class_counts": counts,
        "report": classification_report(targets, predictions, target_names=DAMAGE_CLASSES,
                                        zero_division=0, output_dict=True),
    }, indent=2))
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
