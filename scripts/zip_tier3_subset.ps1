# Zip pre-built tier3_subset for Kaggle dataset upload
# Usage:
#   .\scripts\zip_tier3_subset.ps1
#   .\scripts\zip_tier3_subset.ps1 -OutputPath D:\AMD\tier3_subset.zip

param(
    [string]$SubsetDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "data\tier3_subset"),
    [string]$OutputPath = (Join-Path (Split-Path $PSScriptRoot -Parent) "tier3_subset.zip")
)

$ErrorActionPreference = "Stop"

foreach ($sub in @("images", "labels", "targets")) {
    $path = Join-Path $SubsetDir $sub
    if (-not (Test-Path $path)) {
        Write-Error "Missing $path — run prepare_tier3_subset.py first."
    }
}

$imgCount = (Get-ChildItem (Join-Path $SubsetDir "images") -File).Count
Write-Host "tier3_subset: $imgCount files in images/"

if (Test-Path $OutputPath) {
    Remove-Item $OutputPath -Force
}

# Archive as tier3_subset/... so Kaggle input matches stage_kaggle_data.py
$staging = Join-Path $env:TEMP "disasteriq-tier3-zip"
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $staging "tier3_subset") | Out-Null
foreach ($sub in @("images", "labels", "targets")) {
    Copy-Item (Join-Path $SubsetDir $sub) (Join-Path $staging "tier3_subset" $sub) -Recurse
}
Compress-Archive -Path (Join-Path $staging "tier3_subset") -DestinationPath $OutputPath -CompressionLevel Optimal
Remove-Item $staging -Recurse -Force

$sizeMb = [math]::Round((Get-Item $OutputPath).Length / 1MB, 1)
Write-Host "Wrote $OutputPath ($sizeMb MB)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. https://www.kaggle.com/datasets -> New Dataset"
Write-Host "  2. Upload $OutputPath (private, slug e.g. disasteriq-tier3-subset)"
Write-Host "  3. On Kaggle notebook: add BOTH disasteriq-train-subset AND tier3 datasets as Input"
