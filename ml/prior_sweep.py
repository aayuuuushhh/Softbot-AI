"""Pick a prior-correction strength by looking at what each one costs.

    python -m ml.prior_sweep ml/checkpoints/siamese_resnet18.pt data/processed/manifest_test.csv

Training with a balanced sampler leaves the model expecting damage 25% of the time. Real
regions are ~95% intact, and `prior_logit_shift` corrects for that - but correcting *fully*
maximises accuracy by refusing to call anything damaged, which is the opposite of what a
triage tool is for.

So sweep it. Run the network once, then re-argmax the same logits under each strength: the
whole curve costs one forward pass. Read `dmg recall` and `severe missed` first and buy as
much precision as you can without giving those up - accuracy and macro-F1 will happily point
you somewhere useless.

Feed the chosen number to `python -m ml.evaluate --prior ... --prior-strength ...`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .datasets.xbd import XBDChipDataset
from .models.siamese import load_checkpoint, prior_logit_shift

STRENGTHS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def collect_logits(model, manifest: Path, device: str, batch_size: int, workers: int):
    """One forward pass. Every strength below is just a different argmax over these."""
    loader = DataLoader(
        XBDChipDataset(manifest, augment=False), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=(device == "cuda"),
    )
    logits, targets = [], []
    with torch.no_grad():
        for pre, post, label in loader:
            logits.append(model(pre.to(device), post.to(device)).float().cpu())
            targets.append(label)
    return torch.cat(logits), torch.cat(targets).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prior", default="0.95,0.03,0.015,0.005",
                        help="Deployment base rate, in DAMAGE_CLASSES order.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for path in (args.checkpoint, args.manifest):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"checkpoint: {args.checkpoint}\nmanifest:   {args.manifest}\ndevice:     {device}")

    model = load_checkpoint(str(args.checkpoint), device)
    logits, targets = collect_logits(model, args.manifest, device, args.batch_size, args.workers)
    shift = prior_logit_shift([float(v) for v in args.prior.split(",")])
    print(f"chips: {len(targets)}   prior: {args.prior}\n")

    print(f"{'strength':>9}{'acc':>7}{'macroF1':>9}{'dmg recall':>12}{'dmg prec':>10}"
          f"{'severe missed':>15}{'false alarms':>14}")
    print("-" * 76)
    for strength in STRENGTHS:
        predictions = (logits + strength * shift).argmax(1).numpy()
        matrix = np.zeros((4, 4), dtype=int)
        for truth, predicted in zip(targets, predictions):
            matrix[truth, predicted] += 1

        damaged, missed = matrix[1:].sum(), matrix[1:, 0].sum()
        flagged, hits = matrix[:, 1:].sum(), matrix[1:, 1:].sum()
        severe = matrix[2:].sum()
        print(f"{strength:>9.1f}{np.trace(matrix) / matrix.sum():>7.3f}"
              f"{f1_score(targets, predictions, average='macro', zero_division=0):>9.3f}"
              f"{(damaged - missed) / max(damaged, 1):>12.3f}"
              f"{hits / max(flagged, 1):>10.3f}"
              f"{matrix[2:, 0].sum() / max(severe, 1):>14.1%}{flagged - hits:>14d}")

    print("\nRead 'dmg recall' and 'severe missed' first: strength 1.0 usually wins on accuracy\n"
          "and macro-F1 by declining to report damage at all.")


# Required, not optional: DataLoader workers re-import this module under Python 3.14's
# default forkserver start method, and without the guard each one re-runs the sweep.
if __name__ == "__main__":
    main()
