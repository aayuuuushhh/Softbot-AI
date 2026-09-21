# cleanup-c-drive.ps1 — reclaim C: space from Docker + temp + stray dev servers
param(
    [switch]$Force
)

$ErrorActionPreference = "Continue"

function Show-Disk {
    Write-Host "`n=== Disk space ===" -ForegroundColor Cyan
    Get-PSDrive C, D | ForEach-Object {
        $freeGB = [math]::Round($_.Free / 1GB, 1)
        $usedGB = [math]::Round($_.Used / 1GB, 1)
        Write-Host ("{0}:  {1} GB free / {2} GB used" -f $_.Name, $freeGB, $usedGB)
    }
}

Show-Disk

Write-Host "`n=== Stopping stray dev processes ===" -ForegroundColor Cyan

$port8001 = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $port8001) {
    Write-Host "Stopping PID $procId (uvicorn on :8001)"
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        $_.CommandLine -match 'validate_docker_pairs|compare_models'
    } |
    ForEach-Object {
        Write-Host "Stopping python PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host "`n=== Docker disk usage (before) ===" -ForegroundColor Cyan
docker system df

$doPrune = $Force
if (-not $doPrune) {
    $confirm = Read-Host "`nPrune ALL unused Docker images + build cache? (y/N)"
    $doPrune = ($confirm -eq 'y')
}

if ($doPrune) {
    Write-Host "Pruning build cache..."
    docker builder prune -af
    Write-Host "Pruning unused images..."
    docker image prune -af
    Write-Host "`n=== Docker disk usage (after) ===" -ForegroundColor Cyan
    docker system df
} else {
    Write-Host "Skipped Docker prune."
}

Write-Host "`n=== Cleaning %TEMP% ===" -ForegroundColor Cyan
$tempBefore = (Get-ChildItem $env:TEMP -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum).Sum
Write-Host ("Temp size before: {0:N1} GB" -f ($tempBefore / 1GB))

Get-ChildItem $env:TEMP -Force -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$tempAfter = (Get-ChildItem $env:TEMP -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum).Sum
Write-Host ("Temp size after:  {0:N1} GB" -f ($tempAfter / 1GB))

Write-Host "`n=== Pip cache ===" -ForegroundColor Cyan
$pip = "D:\AMD\backend\.venv\Scripts\pip.exe"
if (Test-Path $pip) {
    & $pip cache purge 2>$null
} else {
    pip cache purge 2>$null
}

Show-Disk
Write-Host "`nDone." -ForegroundColor Green
