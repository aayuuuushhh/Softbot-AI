# Local data layout — UDDHAR

Git tracks **demo pairs only** (`data/demo/`). Full xBD archives stay on disk outside git.

## Archives (local download)

| Dataset | MD5 | Typical path |git@github.com:lamafuri/Uddhar.git
|---------|-----|--------------|
| Test | `1b39c47e05d1319c17cc8763cee6fe0c` | `D:\test_images_labels_targets.tar` |
| Train (official) | `a20ebbfb7eb3452785b63ad02ffd1e16` | `D:\train_images_labels_targets.tar.gz` |
| Train (Kaggle tier1) | — | `D:\archive.zip` |
| Tier3 (Kaggle) | — | `D:\archive (1).zip` |

Verify official archives:

```powershell
.\scripts\verify-downloads.ps1
```

Verify Kaggle archives:

```powershell
.\scripts\verify-kaggle-archive.ps1 -ArchivePath D:\archive.zip
.\scripts\verify-kaggle-archive.ps1 -ArchivePath "D:\archive (1).zip"
```

## Extracted directories (gitignored)

| Path | Contents |
|------|----------|
| `data/demo/` | 10 curated pairs (committed) |
| `data/test/` | Full test set (~2.6 GB archive extract) |
| `data/train/` | Full train set (flat layout: images, labels, targets) |
| `data/train_subset/` | Filtered earthquake/flood/wildfire for PyTorch fine-tune |
| `data/tier3/` | xBD tier3 disasters (flat layout: images, labels, targets) |
| `data/tier3_subset/` | Filtered tier3 wildfire/flood/tornado for extra fine-tune/eval |

### Extract test

```powershell
New-Item -ItemType Directory -Force -Path data\test | Out-Null
tar -xf D:\test_images_labels_targets.tar -C data
```

### Extract train — official archive

```powershell
New-Item -ItemType Directory -Force -Path data\train | Out-Null
tar -xzf D:\train_images_labels_targets.tar.gz -C data
```

### Extract train — Kaggle `archive.zip` (tier1)

Kaggle nests files as `train/train/images`, `train/train/labels`. Flatten into `data/train/`:

```powershell
.\scripts\extract-kaggle-archive.ps1 -ArchivePath D:\archive.zip
```

### Extract tier3 — Kaggle `archive (1).zip`

Flat layout inside zip as `tier3/images`, `tier3/labels` → `data/tier3/`:

```powershell
.\scripts\extract-kaggle-archive.ps1 -ArchivePath "D:\archive (1).zip"
```

### Generate target masks (PyTorch training)

After labels are in `data/train/labels/`, rasterize JSON polygons to PNG masks:

```powershell
.\backend\.venv\Scripts\python.exe scripts\generate_train_targets.py
```

Requires `ml/pytorch-xview2` (cloned locally, gitignored).

### Build train subset

Filter to hackathon disaster prefixes (earthquake, flood, wildfire):

```powershell
python scripts\prepare_train_subset.py --train-dir D:\AMD\data\train --output-dir D:\AMD\data\train_subset
```

### Build tier3 subset

Filter tier3 wildfire, flood, and tornado events:

```powershell
python scripts\prepare_tier3_subset.py --tier3-dir D:\AMD\data\tier3 --output-dir D:\AMD\data\tier3_subset
```

## Status (last verified)

| Item | Status |
|------|--------|
| Test archive MD5 | **OK** at `D:\test_images_labels_targets.tar` |
| Test data | Extracted to `data/test/` (1866 images) |
| Kaggle `D:\archive.zip` | **Present** — verified and extracted |
| Kaggle `D:\archive (1).zip` | **Present** — tier3 verified and extracted |
| `data/train/images` | 5598 files (2799 pre/post pairs) |
| `data/train/labels` | 5598 JSON labels |
| `data/train/targets` | 5598 PNG masks (generated via `convert2png.py`) |
| `data/train_subset/` | 2898 files per folder (~1449 pairs) |
| `data/tier3/images` | 12738 files (6369 pre/post pairs) |
| `data/tier3/labels` | 12738 JSON labels |
| `data/tier3/targets` | 12738 PNG masks (generated via `convert2png.py`) |
| `data/tier3_subset/` | 11860 images (~5930 pairs) — wildfire, flood, tornado |
| Official train archive | Optional — Kaggle archives used instead |

## AMD GPU sync

Copy subset + project to the MI300 instance:

```bash
bash scripts/rsync_to_amd.sh root@YOUR_DROPLET_IP
```

See [AMD_FINETUNE_PLAN.md](AMD_FINETUNE_PLAN.md) for training commands.
