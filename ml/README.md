# ML

The backend supports three inference modes:

| Mode | Model |
|---|---|
| `stub` | Pixel-difference heuristic; not machine learning |
| `docker` | Pretrained xView2 TensorFlow baseline |
| `pytorch` | Fine-tuned xView2 model; requires a checkpoint |

## Run pretrained model

Build the inference image, set `INFERENCE_MODE=docker` in `.env`, then start the
backend:

```powershell
docker compose --profile build-ml build ml
.\scripts\start-backend.ps1
```

Docker inference runs locally; the backend cannot invoke the host Docker daemon
from inside its own container.

## Fine-tune with xBD

Register at [xView2](https://xview2.org) and download the labeled training
archive. xBD is licensed for non-commercial use; follow its attribution and
share-alike terms.

On Windows, clone the training code and prepare the data:

```powershell
git clone --depth 1 https://github.com/michal2409/xView2.git ml/pytorch-xview2
.\scripts\verify-kaggle-archive.ps1 -ArchivePath "D:\xbd\train.zip"
.\scripts\extract-kaggle-archive.ps1 -ArchivePath "D:\xbd\train.zip"
.\backend\.venv\Scripts\python.exe scripts\generate_train_targets.py --data-dir data\train
.\backend\.venv\Scripts\python.exe scripts\prepare_train_subset.py --train-dir data\train --output-dir data\train_subset
.\scripts\zip_train_subset.ps1 -SubsetDir data\train_subset -OutputPath "D:\xbd\train-subset.zip"
```

Upload the subset ZIP as a private Kaggle dataset. In a Kaggle notebook, enable a
CUDA GPU, clone or upload this project, attach the dataset, and extract it to
`/kaggle/working/data/train_subset` so that `images/`, `labels/`, and `targets/`
are directly inside that folder. Then run:

```bash
mkdir -p /kaggle/working/data/train_subset
unzip -q /kaggle/input/YOUR_DATASET/YOUR_ARCHIVE.zip -d /kaggle/working/data/train_subset
git clone --depth 1 https://github.com/michal2409/xView2.git ml/pytorch-xview2
python ml/finetune/kaggle_train.py --stage all
```

Download `/kaggle/working/damage_best.ckpt` to `ml/checkpoints/damage_best.ckpt`,
set `INFERENCE_MODE=pytorch` in `.env`, and restart the backend. The training
configuration is in `ml/finetune/config_subset_kaggle.yaml`.
