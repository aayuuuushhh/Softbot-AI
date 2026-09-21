# Push to GitHub (one-time)

The repo is initialized locally. GitHub CLI must be authenticated before creating the remote.

## 1. Authenticate

```powershell
gh auth login
```

Choose: GitHub.com → HTTPS → Login with browser.

## 2. Create public repo and push

From `D:\AMD`:

```powershell
.\scripts\push-to-github.ps1
```

Or manually:

```powershell
gh repo create satellite-disaster-triage --public --source=. --remote=origin --description "Satellite Disaster-Damage Triage — AMD ACT II Hackathon (DarkNem)" --push
```

Replace `satellite-disaster-triage` with your preferred repo name.

## 3. Share with friend

Send the clone URL and point them to [docs/FRIEND_SETUP.md](FRIEND_SETUP.md).

## What is excluded from git

See `.gitignore`: `data/test/`, `backend/.venv/`, `frontend/node_modules/`, `ml/xview2-baseline/`, archives.

`data/demo/` (10 pairs) **is** included so stub mode works out of the box.
