# Prerequisites status

## All ready

| Item | Status |
|------|--------|
| Node.js v24.16.0 | OK |
| Python 3.12.10 | OK — `%LOCALAPPDATA%\Programs\Python\Python312\` |
| Backend venv + numpy | OK — `backend\.venv\Scripts\python.exe` |
| WSL 2 + Ubuntu | OK — version 2.7.10 |
| Docker Desktop | OK — `docker ps` and `hello-world` pass |
| Docker Compose | OK — v5.3.0 |

Verify anytime: `.\scripts\verify-prerequisites.ps1`

## ML inference image

Build when needed (first run ~15–30 min):

```powershell
cd E:\DisasterIQ
docker compose --profile build-ml build ml
```

Then set `INFERENCE_MODE=docker` in `.env` for real baseline inference.

Note: the backend shells out to `docker run` for ML inference using host
filesystem paths, so run the backend on the host (`.\scripts\start-backend.ps1`)
rather than inside docker-compose when `INFERENCE_MODE=docker`.
