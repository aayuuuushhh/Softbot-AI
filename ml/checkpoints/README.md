# Place fine-tuned PyTorch checkpoints here (gitignored)

## Trust model (read before loading)

Lightning `.ckpt` files are **pickle-based**. `torch.load(..., weights_only=False)`
can execute arbitrary code. Only load checkpoints you trained or obtained from a
trusted teammate. Prefer verifying a SHA256 before copying into this directory:

```powershell
Get-FileHash ml\checkpoints\damage_best.ckpt -Algorithm SHA256
```

Record the hash next to the file name in your run notes. Never accept
user-uploaded checkpoints through the public API.

## Localization (from Kaggle tier3 run, ~epoch 7)

Holdout validation uses a disjoint ~10% pair split (`ml/finetune/data_layout.py`).
Reported F1 is only meaningful after that layout is in place (not when train/test
pointed at the same images).

- `loc_best.ckpt` — use this as `--ckpt_pre` / upload for damage stage
- `loc_last.ckpt`, `loc_step.ckpt` — backups

## Damage (for app inference)

After Kaggle damage training, copy:
`/results/dmg/checkpoints/best.ckpt` → `damage_best.ckpt`

Then set in `.env`:
```
INFERENCE_MODE=pytorch
PYTORCH_CHECKPOINT_PATH=ml/checkpoints/damage_best.ckpt
```

Note: replace any older `damage_best.ckpt` after a new damage stage finishes.
