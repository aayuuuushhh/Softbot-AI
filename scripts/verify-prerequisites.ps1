# Check dev prerequisites for Disaster Triage project
Write-Host "=== Prerequisites Check ===" -ForegroundColor Cyan

$ok = $true

# Node.js
try {
    $nodeVer = node --version 2>$null
    Write-Host "[OK] Node.js $nodeVer" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Node.js not found" -ForegroundColor Red
    $ok = $false
}

# Python
$py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (Test-Path $py) {
    $pyVer = & $py --version 2>&1
    Write-Host "[OK] Python $pyVer at $py" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Python 3.12 not found. Run: winget install Python.Python.3.12" -ForegroundColor Red
    $ok = $false
}

# Backend venv
$repoRoot = Split-Path $PSScriptRoot -Parent
$venvPy = Join-Path $repoRoot "backend\.venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $numpyOk = & $venvPy -c "import numpy" 2>$null; if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Backend venv with numpy" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Backend venv missing deps. Run: cd $repoRoot\backend; .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] Backend venv not created. Run: .\scripts\start-backend.ps1" -ForegroundColor Yellow
}

# WSL
$wslList = wsl -l -v 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $wslList -match "not installed|no installed") {
    Write-Host "[FAIL] WSL not installed. Run as Admin: .\scripts\install-wsl-admin.ps1 then reboot" -ForegroundColor Red
    $ok = $false
} else {
    Write-Host "[OK] WSL installed" -ForegroundColor Green
}

# Docker
$dockerPs = docker ps 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $dockerPs -match "unable to start|error response from daemon") {
    Write-Host "[FAIL] Docker not running. Run: .\scripts\start-docker-admin.ps1 (as Admin) or open Docker Desktop" -ForegroundColor Red
    $ok = $false
} else {
    Write-Host "[OK] Docker engine running" -ForegroundColor Green
}

Write-Host ""
if ($ok) { Write-Host "All prerequisites ready." -ForegroundColor Green }
else { Write-Host "Some prerequisites need attention (see above)." -ForegroundColor Yellow }
