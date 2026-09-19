"""Score a trained checkpoint against the held-out manifests.

    python -m ml.evaluate --checkpoint ml/checkpoints/siamese_resnet18.pt

Training reports validation numbers on tiles it split off itself. Those are optimistic: the
val tiles come from the same three events the model fitted on. This script is the honest read.

It scores two sets, and the gap between them is the finding:
  test     unseen tiles, *seen* events  -> does it generalise across tiles
  holdout  mexico-earthquake, unseen    -> does it transfer to seismic damage

The second number is the one to quote for Nepal. Expect it to be worse, and say so rather
than reporting the first and hoping nobody asks.

See PROJECT_PLAN.md section 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .datasets.xbd import XBDChipDataset, class_counts
from .models.siamese import DAMAGE_CLASSES, load_checkpoint, prior_logit_shift


def resolve_prior(spec: str | None, manifest: Path,
                  strength: float = 1.0) -> tuple[torch.Tensor | None, str]:
    """`--prior` -> a logit shift, plus a label saying where the numbers came from.

    `manifest` uses the evaluation set's own class balance. That is an *oracle*: it reads the
    labels being scored, so it is an upper bound on what the correction can buy, never a
    number to deploy on. A hand-set prior is the honest one - it is a claim about the region
    you are flying over, made before you see the answers.
    """
    if spec is None:
        return None, "none (model's uniform training prior)"
    if spec == "manifest":
        counts = class_counts(manifest)
        total = sum(counts.values()) or 1
        shares = ", ".join(f"{c}={counts[c] / total:.3f}" for c in DAMAGE_CLASSES)
        shift, label = prior_logit_shift(counts), f"manifest ORACLE - upper bound only ({shares})"
    else:
        values = [float(v) for v in spec.split(",")]
        shift, label = prior_logit_shift(values), f"fixed {values}"
    # Partial strength interpolates between the training prior (0.0) and the stated one (1.0).
    # Full strength optimises accuracy and is usually far too aggressive for triage, where
    # missing a destroyed building costs more than another false alarm - see ml/prior_sweep.py.
    return shift * strength, f"{label} @ strength {strength:g}"


def evaluate_manifest(model, manifest: Path, device: str, batch_size: int, workers: int,
                      logit_shift: torch.Tensor | None = None) -> dict:
    dataset = XBDChipDataset(manifest, augment=False)
    if len(dataset) == 0:
        return {"chips": 0}

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device == "cuda"),
    )

    predictions: list[int] = []
    targets: list[int] = []
    for pre, post, label in loader:
        predicted, _ = model.predict(pre.to(device), post.to(device), logit_shift)
        predictions.extend(predicted.cpu().tolist())
        targets.extend(label.tolist())

    present = sorted(set(targets))
    return {
        "chips": len(dataset),
        "class_counts": class_counts(manifest),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        # Macro-F1 over absent classes is misleading: a class with no support scores 0 and
        # drags the mean down for a reason that has nothing to do with the model.
        "macro_f1_present_classes": float(
            f1_score(targets, predictions, average="macro", labels=present, zero_division=0)
        ),
        "accuracy": float(np.mean(np.array(targets) == np.array(predictions))),
        "report": classification_report(
            targets, predictions, labels=list(range(len(DAMAGE_CLASSES))),
            target_names=DAMAGE_CLASSES, zero_division=0, output_dict=True,
        ),
        "confusion": confusion_matrix(
            targets, predictions, labels=list(range(len(DAMAGE_CLASSES)))
        ).tolist(),
        "_targets": targets,
        "_predictions": predictions,
    }


def print_result(label: str, manifest: Path, result: dict) -> None:
    print(f"\n{'=' * 70}\n{label}  ({manifest})\n{'=' * 70}")
    if not result["chips"]:
        print("  empty manifest - skipped")
        return

    print(f"chips: {result['chips']}   accuracy: {result['accuracy']:.3f}   "
          f"macro-F1: {result['macro_f1']:.3f}")
    print(classification_report(
        result["_targets"], result["_predictions"], labels=list(range(len(DAMAGE_CLASSES))),
        target_names=DAMAGE_CLASSES, zero_division=0, digits=3,
    ))
    print("confusion matrix (rows = truth, cols = predicted)")
    header = " " * 16 + "".join(f"{c[:11]:>13s}" for c in DAMAGE_CLASSES)
    print(header)
    for name, row in zip(DAMAGE_CLASSES, result["confusion"]):
        print(f"{name:>15s} " + "".join(f"{v:>13d}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("ml/checkpoints/siamese_resnet18.pt"))
    parser.add_argument("--test-manifest", type=Path,
                        default=Path("data/processed/manifest_test.csv"))
    parser.add_argument("--holdout-manifest", type=Path,
                        default=Path("data/processed/manifest_holdout.csv"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the JSON summary (default: next to the checkpoint)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prior", default=None,
                        help="Re-base the class prior before argmax. Either four comma-separated "
                             "weights in DAMAGE_CLASSES order (e.g. '0.95,0.03,0.01,0.01') or "
                             "'manifest' to use the eval set's own balance - the latter reads the "
                             "labels it is scoring, so treat it as an upper bound, not a result.")
    parser.add_argument("--prior-strength", type=float, default=1.0,
                        help="How far to move toward --prior (0 = off, 1 = all the way). "
                             "Pick it with ml/prior_sweep.py, not by eye.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}. Train one first, or copy it back from "
            "whichever machine did the training."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"checkpoint: {args.checkpoint}\ndevice: {device}")

    model = load_checkpoint(str(args.checkpoint), device)

    jobs = [
        ("TEST - unseen tiles, seen events", args.test_manifest),
        ("HOLDOUT - mexico-earthquake, never seen", args.holdout_manifest),
    ]

    summary = {}
    for label, manifest in jobs:
        if not manifest.exists():
            print(f"\n[skip] {label}: no manifest at {manifest} "
                  "(run `python scripts/run_pipeline.py --prepare`)")
            continue
        shift, prior_label = resolve_prior(args.prior, manifest, args.prior_strength)
        print(f"\n[{label}] prior: {prior_label}")
        result = evaluate_manifest(model, manifest, device, args.batch_size, args.workers, shift)
        result["prior"] = prior_label
        print_result(label, manifest, result)
        summary[manifest.stem] = {k: v for k, v in result.items() if not k.startswith("_")}

    test = summary.get(args.test_manifest.stem, {})
    holdout = summary.get(args.holdout_manifest.stem, {})
    if test.get("chips") and holdout.get("chips"):
        drop = test["macro_f1"] - holdout["macro_f1"]
        print(f"\n{'=' * 70}")
        print(f"macro-F1  test {test['macro_f1']:.3f}  ->  earthquake holdout "
              f"{holdout['macro_f1']:.3f}   (drop {drop:+.3f})")
        print("That drop is the domain-transfer cost. It belongs on the limitations slide.")

    out = args.out or args.checkpoint.with_suffix(".eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
