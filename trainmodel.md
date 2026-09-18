# Training the damage classifier

Runbook for training `SiameseDamageNet` on another machine and bringing the result back.

Everything up to training is already done: imagery downloaded, chips extracted, manifests
built and verified. This document covers the training run itself.

---

## What you are training

A **siamese ResNet18**. One shared ImageNet-pretrained encoder sees the pre-disaster chip and
the post-disaster chip of the same building; the head reads `[pre, post, post - pre]` and
predicts one of four classes:

```
no-damage   minor-damage   major-damage   destroyed
```

The explicit difference term matters — damage is a *change*, so handing the head the
subtraction directly means it does not have to learn it from scratch.

Chips are 128×128 RGB crops of individual building footprints, padded 10 px, ImageNet-normalised.

| | |
|---|---|
| Model | `ml/models/siamese.py` |
| Training loop | `ml/train.py` |
| Evaluation | `ml/evaluate.py` |
| Checkpoint out | `ml/checkpoints/siamese_resnet18.pt` |

---

## What it trains on

| manifest | chips | no-damage | minor | major | destroyed | role |
|---|---:|---:|---:|---:|---:|---|
| `manifest.csv` | 66,344 | 74.4% | 8.0% | 3.8% | 13.9% | **training** |
| `manifest_test.csv` | 20,473 | 80.3% | 5.3% | 2.9% | 11.4% | held out |
| `manifest_holdout.csv` | 43,596 | 99.4% | 0.4% | 0.1% | 0.0% | held out |

Training uses **three events** — `hurricane-michael`, `palu-tsunami`, `santa-rosa-wildfire` —
picked for class coverage, not disaster type. `mexico-earthquake` is deliberately excluded:
it has **3 destroyed buildings in the entire event**, so it cannot teach the rare classes,
but it is the only earthquake in the pool and therefore the honest test of seismic transfer.

The three manifests share no chip IDs. Verified, not assumed.

---

## 1. Hardware

| | |
|---|---|
| Minimum | Any CUDA GPU with 4 GB VRAM |
| Comfortable | 6 GB+ (batch size 64 at 128 px fits easily) |
| CPU-only | Works, but expect hours rather than minutes — use `--freeze-backbone` |

Disk: ~5 GB for the chips, ~50 MB for the checkpoint.

---

## 2. Get the data onto the machine

### Option A — the bundle (recommended)

On this machine:

```bash
bash scripts/make_training_bundle.sh          # -> data/xbd_training_bundle.tar  (4.8 GB)
```

It packs the chips, the three manifests, `ml/` and `config/paths.yaml`. It does **not** pack
the 3.3 GB of raw 1024×1024 tiles — chip extraction already happened and the training box
never reads them.

Copy `xbd_training_bundle.tar` and `xbd_training_bundle.tar.sha256` across, then:

```bash
git clone <this repo> softbot && cd softbot
sha256sum -c xbd_training_bundle.tar.sha256    # verify before unpacking
tar -xf xbd_training_bundle.tar
```

Unpack it **at the root of a clone**. The manifests store chip paths relative to the repo
root (`data/processed/chips/train/pre/...`), so they resolve on any machine as long as the
layout is right. If you unpack somewhere else, every path breaks.

### Option B — rebuild from source

Slower (~20 min download, ~25 min extraction) but needs no file transfer:

```bash
git clone <this repo> softbot && cd softbot
bash scripts/fetch_xbd.sh                      # 3.3 GB, resumable, self-verifying
python scripts/run_pipeline.py --prepare       # extract chips, print class balance
```

`fetch_xbd.sh` caps itself at 6 parallel connections because HuggingFace returns HTTP 429
above roughly that. Raising it makes the download fail, not finish faster.

---

## 3. Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r ml/requirements.txt
```

For a CPU-only machine, install torch from the CPU index instead (saves a ~3 GB CUDA download):

```bash
uv pip install --python .venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r ml/requirements.txt
```

`ml/requirements.txt` is all you need — the training path does not import the backend.
Confirm the GPU is visible:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Smoke test first

Three epochs, head only, backbone frozen. A few minutes at most.

```bash
.venv/bin/python -m ml.train --freeze-backbone --epochs 3
```

**What you are checking:** that macro-F1 moves off zero and loss decreases. You are not
checking that the numbers are good — they will not be. This exists to catch a broken data
path before you commit to the full run.

If macro-F1 is still ~0.00 after 3 epochs, stop and read the troubleshooting table below.

---

## 5. The real run

```bash
.venv/bin/python -m ml.train --epochs 8
```

Rough expectation on a modern laptop GPU: a few minutes per epoch, so 20–45 minutes total.
The first epoch's `tqdm` bar gives you the real number — extrapolate from that rather than
trusting this estimate. PNG decode of 260k small files is usually the bottleneck, not the
GPU, so `--workers` matters more than it looks.

### Useful flags

| flag | default | when to change it |
|---|---|---|
| `--epochs` | 8 | Fewer if the clock is short; a finished 3-epoch run beats an unfinished 8 |
| `--batch-size` | 64 | Drop to 32 or 16 if you hit CUDA OOM |
| `--workers` | 4 | Raise toward your core count if the GPU is idling between batches |
| `--lr` | 3e-4 | Leave it unless loss diverges |
| `--freeze-backbone` | off | Trains only the head. Much faster, meaningfully worse |
| `--val-fraction` | 0.2 | Leave it |
| `--seed` | 2015 | Leave it — reproducibility |
| `--out` | `ml/checkpoints/siamese_resnet18.pt` | Leave it; `evaluate` expects this path |

### What the script already handles

Do not re-implement these:

- **Class imbalance, twice over** — inverse-frequency weights on the loss *and* a
  `WeightedRandomSampler` that oversamples rare classes. Batches come out near-balanced
  across all four classes despite the 74/8/4/14 pool.
- **Tile-level splitting** — validation splits on the source *tile*, not the chip. Buildings
  from one tile look alike; splitting per chip leaks validation into training and gives a
  score that evaporates on real imagery.
- **Paired augmentation** — flips and rotations apply identically to pre and post. Flip one
  and not the other and you have invented damage that is not there.
- **Checkpointing on best macro-F1**, not on the last epoch.

---

## 6. Read the output honestly

The script prints per-class F1 and a confusion matrix, never bare accuracy. That is
deliberate: a model that answers `no-damage` for everything scores ~74% accuracy on this
training pool and ~99% on the earthquake hold-out, while being useless for the one job it has.

**Watch macro-F1 and the per-class recall for `destroyed` and `major-damage`.** Those are the
classes that decide where responders go.

Expect the confusion matrix to show adjacent-class confusion — `minor` mistaken for `no-damage`,
`major` for `destroyed`. That is normal and much less harmful than a `destroyed` building
predicted as `no-damage`. Look for that specific cell.

### Known data limitation

`minor-damage` comes almost entirely from `hurricane-michael` (6,271 of 6,562 across all
events). `palu-tsunami` has exactly **1**. If `minor-damage` F1 comes out weak, that is a
property of the data, not a bug in the training loop — do not burn hours debugging the
optimizer over it. Report it in the confusion matrix and move on.

---

## 7. Outputs

| file | what it is |
|---|---|
| `ml/checkpoints/siamese_resnet18.pt` | weights, class names, best macro-F1, epoch, class counts |
| `ml/checkpoints/siamese_resnet18.metrics.json` | best macro-F1, class counts, full per-class report |

**Bring both back.** The `.json` is what goes on the slide; the `.pt` is what the dashboard loads.

---

## 8. Evaluate (back on the main machine)

Drop the checkpoint into `ml/checkpoints/` and run:

```bash
.venv/bin/python -m ml.evaluate
```

This scores two sets that training never saw:

- **test** — unseen tiles, *seen* events. Does it generalise across tiles?
- **holdout** — `mexico-earthquake`, never seen at all. Does it transfer to seismic damage?

It prints both, then the gap:

```
macro-F1  test 0.XXX  ->  earthquake holdout 0.XXX   (drop -0.XXX)
```

**That drop is the finding.** It is the measured cost of training on hurricanes, tsunamis and
wildfires and applying the result to earthquakes. Quote the hold-out number for anything
Nepal-related, not the test number, and put the drop on the limitations slide. Reporting the
higher number and hoping nobody asks is the one thing that turns an honest project into a
dishonest one.

Results are written to `ml/checkpoints/siamese_resnet18.eval.json`.

Then confirm the whole pipeline sees it:

```bash
.venv/bin/python scripts/run_pipeline.py --check     # Model checkpoint -> [ok]
```

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `No manifest at data/processed/manifest.csv` | Bundle not unpacked, or unpacked in the wrong directory | Unpack at the repo root |
| `FileNotFoundError` on a chip PNG | Manifest unpacked without `data/processed/chips/`, or unpacked outside a clone | Re-extract the full tar at the repo root |
| `CUDA out of memory` | Batch too large for the card | `--batch-size 32`, then `16` |
| macro-F1 stuck at 0.00 | Data path broken, or every prediction is one class | Check the printed class balance is 74/8/4/14, not all-zeros |
| Epochs crawling, GPU near idle | PNG decode bottleneck | Raise `--workers` toward core count |
| `torch.cuda.is_available()` is `False` | CPU wheels installed | Reinstall torch from the CUDA index |
| Validation F1 far above test F1 later | Normal — val tiles share events with training | This is exactly why `evaluate.py` exists |

---

## Licence

xBD is **CC BY-NC-SA 4.0 — non-commercial**. Fine for the hackathon and for research, but it
conflicts with the commercial tiers in the business model. Credit it on the demo slide and
have an answer ready if a judge asks.
