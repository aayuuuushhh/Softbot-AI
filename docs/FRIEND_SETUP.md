# Friend laptop setup — Satellite Disaster-Damage Triage (Team DarkNem)

Step-by-step guide to replicate the dev environment and run the app locally.

## 1. Get the code

```powershell
git clone https://github.com/<ORG>/satellite-disaster-triage.git
cd satellite-disaster-triage
```

If the repo is not public yet, ask Ahmad for a zip (exclude `data/test/`, `backend/.venv/`, `frontend/node_modules/`).

## 2. Install prerequisites

| Tool | Install | Verify |
|------|---------|--------|
| Node.js 18+ | `winget install OpenJS.NodeJS.LTS` | `node --version` |
| Python 3.12 | `winget install Python.Python.3.12` | `%LOCALAPPDATA%\Programs\Python\Python312\python.exe --version` |
| WSL 2 | **Admin:** `.\scripts\install-wsl-admin.ps1` then **reboot** | `wsl -l -v` |
| Docker Desktop | https://www.docker.com/products/docker-desktop/ | `docker ps` |

Run the checker:

```powershell
.\scripts\verify-prerequisites.ps1
```

Turn off Windows Store Python aliases if `python` fails: Settings → Apps → Advanced app settings → App execution aliases.

## 3. Download xBD test data (do NOT copy Ahmad's 7 GB folder)

1. Register at [xview2.org](https://xview2.org) — **Open Source** track
2. Download `test_images_labels_targets.tar` (~2.6 GB)
3. Verify MD5: `1b39c47e05d1319c17cc8763cee6fe0c`
4. Extract to project root (creates `data/test/`):

```powershell
New-Item -ItemType Directory -Force -Path data\test | Out-Null
tar -xf D:\path\to\test_images_labels_targets.tar -C data --strip-components=1
# Or if tar extracts a test/ subfolder:
# tar -xf ... -C data
```

## 4. Demo pairs (choose one)

**Option A — Already in repo:** `data/demo/` is committed (10 pairs). Skip to step 5.

**Option B — Re-curate from your test extract:**

```powershell
.\scripts\curate_demo_subset.ps1
```

**Option C — From tar only (no full test extract):**

```powershell
.\scripts\curate_demo_subset.ps1 -TarPath D:\path\to\test_images_labels_targets.tar
```

Pair list is in `data/demo/manifest.json` (5 earthquake + 5 flood).

## 5. Environment

```powershell
Copy-Item .env.example .env
# Optional when hackathon credits unlock:
# FIREWORKS_API_KEY=...
```

## 6. Run the app

Terminal 1:

```powershell
.\scripts\start-backend.ps1
```

Terminal 2:

```powershell
.\scripts\start-frontend.ps1
```

Open http://localhost:3000 → pick a demo pair → **Analyze**.

## 7. Optional: ML Docker image

Stub inference works without GPU. To build baseline inference (~30 min, stable network):

```powershell
docker compose --profile build-ml build ml
```

Set `INFERENCE_MODE=docker` in `.env` only after the image builds.

## 8. Train data (for fine-tuning later)

| Archive | Size | MD5 |
|---------|------|-----|
| `train_images_labels_targets.tar.gz` | ~7.8 GB | `a20ebbfb7eb3452785b63ad02ffd1e16` |

Download separately; do not commit to git.

## Troubleshooting

- **No demo pairs in UI:** Run `curate_demo_subset.ps1` or check `GET http://localhost:8000/health` for `demo_pairs: 10`
- **Backend import errors:** Delete `backend\.venv` and re-run `start-backend.ps1`
- **Docker won't start:** Reboot after WSL install; run `.\scripts\start-docker-admin.ps1` as Admin

## Key docs

- [README.md](../README.md) — architecture and API
- [TEAM_ROLES.md](TEAM_ROLES.md) — who owns what
- [scripts/PREREQUISITES-STATUS.md](../scripts/PREREQUISITES-STATUS.md) — Ahmad's machine snapshot
