# Model weights

Checkpoints are not committed (`models/*.pt` is gitignored).

| File | Produced by | Consumed by |
|---|---|---|
| `satellite_change.pt` | `scripts/train_satellite.py` (xBD) | `TorchChangeDetector` (S1) |
| `ground_damage.pt` | `scripts/train_ground.py` | `TorchGroundClassifier` (S2) |

When a file here is missing, `core/vision/inference.py` logs a warning and falls back to
the deterministic `stub` backend rather than failing. Set `UDDHAR_SAT_BACKEND=torch` only
after `scripts/eval_xbd.py` reports F1 > 0.80 on the `destroyed` class.
