# AMD GPU execution plan — UDDHAR

**Team:** DarkNem  
**When:** After AMD Developer Cloud approval

## Decision summary

| Work | Framework | Where |
|------|-----------|-------|
| Demo inference (baseline) | TF 1.15 Docker | Local CPU + Docker |
| Fine-tuning | **PyTorch + ROCm** | AMD MI300 instance |
| TF 1.15 training on ROCm | **Not attempted** | — |

CPU prep is **complete** on the laptop. Training data lives in `data/train_subset/` (~1449 pairs).

## 1. SSH to droplet

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519_amd root@YOUR_DROPLET_IP
```

See [AMD_CLOUD_SSH.md](AMD_CLOUD_SSH.md) for key setup.

## 2. Verify GPU (PyTorch)

```bash
rocm-smi
python3 -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Expected: `cuda: True` and device name containing MI300.

If import fails, install ROCm PyTorch per https://rocm.docs.amd.com/ before training.

## 3. Sync project + data

From your laptop (WSL or Git Bash):

```bash
bash scripts/rsync_to_amd.sh root@YOUR_DROPLET_IP
```

Or manually:

```bash
git clone https://github.com/AhmadRaza4076/DisasterIQ.git
cd DisasterIQ
# rsync data/train_subset/ from laptop
```

## 4. PyTorch fine-tune pipeline

All scripts are in `ml/finetune/`. Training uses `michal2409/xView2` vendored at `ml/pytorch-xview2/`.

### CPU prep before GPU (run on laptop or AMD box)

`run_amd_pipeline.sh` does this automatically, but you can run manually:

```bash
# 1. Patch hardcoded /workspace/xview2/utils/index.csv paths
python3 ml/finetune/patch_pytorch_xview2.py

# 2. Regenerate index.csv for YOUR subset (not upstream 8k train index)
python3 scripts/generate_subset_index.py \
  --data-dir data/train_subset \
  --out ml/pytorch-xview2/utils/index.csv

# 3. Smoke test dataset (no GPU) — needs xView2 Python deps
python3 scripts/test_pytorch_dataset.py --data-dir data/train_subset
```

Training hyperparameters are loaded from `ml/finetune/config_subset.yaml` via `load_config.py` (requires `pip install pyyaml`).

### Option A — native on GPU instance

```bash
cd DisasterIQ
bash ml/finetune/run_amd_pipeline.sh
```

This runs:
0. **Patch** upstream loader + **regenerate** `index.csv` for `train_subset`
1. **Localization** (`train_localization.sh`) — defaults from `config_subset.yaml`
2. **Damage** (`train_damage.sh`) — siamese U-Net, `focal+dice`
3. **Eval** on `data/test/`

### Option B — ROCm Docker

```bash
cd ml/pytorch-xview2
docker build -f Dockerfile.rocm -t darknem-xview2-train .
docker run --rm --device=/dev/kfd --device=/dev/dri \
  -v /data/train_subset:/data/train_subset \
  -v /data/test:/data/test \
  -v /results:/results \
  -v $PWD/../finetune:/workspace/finetune \
  darknem-xview2-train \
  bash /workspace/finetune/run_amd_pipeline.sh
```

Config reference: `ml/finetune/config_subset.yaml` (parsed by `ml/finetune/load_config.py`)

**Time box:** 4–6 hours max. If not converging, stop and ship TF baseline for submission.

## 5. Bring checkpoint back to laptop

```bash
# On GPU instance
scp /results/dmg/checkpoints/best.ckpt user@laptop:DisasterIQ/ml/checkpoints/damage_best.ckpt
```

On laptop `.env`:

```
INFERENCE_MODE=pytorch
PYTORCH_CHECKPOINT_PATH=ml/checkpoints/damage_best.ckpt
```

Restart backend and compare:

```powershell
.\backend\.venv\Scripts\python.exe scripts\compare_models.py --modes docker pytorch
```

With `INFERENCE_MODE=pytorch`, ranked zones include per-zone `confidence` (mean predicted-class probability over building pixels). Docker baseline and stub modes leave `confidence` null — only the fine-tuned PyTorch path exposes softmax probabilities.

## 6. What success looks like for judges

Minimum (already achievable):
- Deterministic zone scoring from satellite masks
- Live Fireworks situation brief narrating ranked JSON
- GitHub repo + demo video on lablab.ai

Stretch:
- "We fine-tuned a PyTorch damage model on AMD MI300 using ROCm"
- Before/after IoU on held-out demo pairs (`scripts/compare_models.py`)

## 7. Teardown

```bash
# On AMD dashboard: destroy droplet when done
```

Never commit GPU private keys or `.env` secrets.
