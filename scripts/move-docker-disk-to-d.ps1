# Move Docker Desktop disk image to D: (frees C: for builds)
param(
    [string]$TargetDir = "D:\Docker"
)

Write-Host "=== Disk space ===" -ForegroundColor Cyan
Get-PSDrive C, D | Format-Table Name, @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}, @{N='TotalGB';E={[math]::Round(($_.Used+$_.Free)/1GB,2)}}

Write-Host ""
Write-Host "C: is low on space. ML Docker builds need about 25-35 GB where Docker stores images." -ForegroundColor Yellow
Write-Host "Project and data should stay on D:\AMD (already there)."
Write-Host ""

if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Write-Host "Created $TargetDir"
}

Write-Host "=== Move Docker disk to D: (one-time, manual) ===" -ForegroundColor Cyan
Write-Host "1. Quit Docker Desktop completely (tray icon -> Quit)"
Write-Host "2. Open Docker Desktop -> Settings -> Resources -> Advanced"
Write-Host "3. Disk image location -> Browse -> $TargetDir"
Write-Host "4. Apply and Restart (migration may take several minutes)"
Write-Host ""
Write-Host "Verify after restart: docker run --rm hello-world"
Write-Host "ML build: cd D:\AMD; docker compose --profile build-ml build ml"
Write-Host ""
Write-Host "=== Docker WSL disks on C: ===" -ForegroundColor Cyan
Get-ChildItem "$env:LOCALAPPDATA\Docker" -Recurse -Filter "*.vhdx" -ErrorAction SilentlyContinue |
    Select-Object FullName, @{N='GB';E={[math]::Round($_.Length/1GB,2)}}
