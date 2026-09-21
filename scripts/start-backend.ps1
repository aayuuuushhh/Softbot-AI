# Start backend (Windows Python venv)
$repoRoot = Split-Path $PSScriptRoot -Parent
$backendDir = Join-Path $repoRoot "backend"
$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
$systemPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating venv..."
    if (-not (Test-Path $systemPython)) {
        Write-Host "Python 3.12 not found. Install with: winget install Python.Python.3.12" -ForegroundColor Red
        exit 1
    }
    & $systemPython -m venv (Join-Path $backendDir ".venv")
    & $venvPython -m pip install -r (Join-Path $backendDir "requirements.txt")
}

# Launch from the repo root, not backend/, so config.py finds the root .env
# (it loads ".env" relative to the working directory). --app-dir puts the
# `app` package on the path without changing the working directory. Running
# from backend/ would silently fall back to defaults: INFERENCE_MODE=stub and
# an empty FIREWORKS_API_KEY (stub brief instead of live inference/narration).
Set-Location $repoRoot
& $venvPython -m uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
