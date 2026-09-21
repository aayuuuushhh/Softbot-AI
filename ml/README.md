# xView2 ML — UDDHAR (Team DarkNem)

Building damage assessment from pre/post satellite imagery.

## Framework decision

| Path | Framework | Use |
|------|-----------|-----|
| **Demo inference (baseline)** | TF 1.15 Docker (`darknem-xview2-inference`) | Ship now — ~2 min/pair on CPU |
| **Fine-tuning** | **PyTorch + ROCm** (`ml/pytorch-xview2`) | AMD MI300 when GPU approved |
| TF 1.15 fine-tune on ROCm | **Rejected** | No practical ROCm path for TF 1.15 |

CPU data prep is complete: `data/train/` (5598 masks) and `data/train_subset/` (~1449 pairs). See [docs/DATA.md](../docs/DATA.md).

## Inference modes

| Mode | Behavior | When to use |
|------|----------|-------------|
| `stub` | GT copy for demo pairs, else pixel-diff heuristic | Fast rehearsed demo |

**`stub-heuristic` limitation:** The pixel-diff fallback only labels *changed* regions as minor, major, or destroyed (classes 2–4). It cannot produce undamaged building pixels (class 1) — that requires ground-truth targets (`stub-groundtruth` on demo pairs) or a real model (`docker` / `pytorch`).
| `docker` | Official xView2 TF 1.15 baseline | Credibility / IoU baseline |
| `pytorch` | Fine-tuned checkpoint via `ml/pytorch-inference/` | After AMD GPU training |

Set in `.env`:

```
INFERENCE_MODE=docker   # or stub | pytorch
```

## TF baseline inference image

Pretrained weights download at image build time:

- Localization: https://github.com/DIUx-xView/xView2/releases/download/v1.0/localization.h5
- Classification: https://github.com/DIUx-xView/xView2/releases/download/v1.0/classification.hdf5

### Build

```powershell
docker compose --profile build-ml build ml
# or
docker build -t darknem-xview2-inference -f ml/inference/Dockerfile ml/inference
```

Requires ~8 GB RAM, stable network, ~15–30 min first build.

### Run backend with real ML

**Use local venv, not `docker compose up` backend**, when `INFERENCE_MODE=docker`:

```powershell
# In .env
INFERENCE_MODE=docker

.\scripts\start-backend.ps1
```

The containerized backend image does not include the Docker CLI. For hackathon demo, run the API from `start-backend.ps1` and invoke `docker run darknem-xview2-inference` per request.

### Manual container test

```powershell
docker run --rm `
  -v D:\AMD\data\demo\images:/submission `
  -v D:\AMD\backend\app\outputs\smoke:/output `
  darknem-xview2-inference `
  /submission/mexico-earthquake_00000005_pre_disaster.png `
  /submission/mexico-earthquake_00000005_post_disaster.png `
  /output/localization.png `
  /output/classification.png
```

### Validate all demo pairs (CPU, ~30 min)

```powershell
cd backend
$env:PYTHONUNBUFFERED = "1"
..\.venv\Scripts\python.exe ..\scripts\validate_docker_pairs.py
```

Last run (building IoU vs ground truth): earthquake pairs ~0.63–0.71, flood pairs ~0.37–0.76.

## PyTorch fine-tuning (AMD GPU)

Vendored repo: `ml/pytorch-xview2/` (gitignored clone of `michal2409/xView2`).

| Artifact | Path |
|----------|------|
| ROCm training image | `ml/pytorch-xview2/Dockerfile.rocm` |
| Training scripts | `ml/finetune/train_localization.sh`, `train_damage.sh` |
| Full pipeline | `ml/finetune/run_amd_pipeline.sh` |
| Config | `ml/finetune/config_subset.yaml` |
| Inference script | `ml/pytorch-inference/infer_pair.py` |
| Checkpoint (after train) | `ml/checkpoints/damage_best.ckpt` |

Full GPU runbook: [docs/AMD_FINETUNE_PLAN.md](../docs/AMD_FINETUNE_PLAN.md)

### Compare models (IoU on demo pairs)

```powershell
# TF baseline only (works now)
.\backend\.venv\Scripts\python.exe scripts\compare_models.py --modes docker

# After checkpoint copied from GPU
.\backend\.venv\Scripts\python.exe scripts\compare_models.py --modes docker pytorch
```

Report: `backend/app/outputs/model_compare/report.json`

## Fine-tuning details

See [ml/finetune/README.md](finetune/README.md).
