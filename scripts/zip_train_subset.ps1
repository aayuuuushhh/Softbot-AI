# Zip pre-built train_subset for Kaggle dataset upload
# Usage:
#   .\scripts\zip_train_subset.ps1
#   .\scripts\zip_train_subset.ps1 -OutputPath D:\disasteriq-train-subset.zip

param(
    [string]$SubsetDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "data\train_subset"),
    [string]$OutputPath = (Join-Path (Split-Path $PSScriptRoot -Parent) "disasteriq-train-subset.zip")
)

$ErrorActionPreference = "Stop"

foreach ($sub in @("images", "labels", "targets")) {
    $path = Join-Path $SubsetDir $sub
    if (-not (Test-Path $path)) {
        Write-Error "Missing $path — run prepare_train_subset.py first."
    }
}

$imgCount = (Get-ChildItem (Join-Path $SubsetDir "images") -File).Count
Write-Host "train_subset: $imgCount files in images/"

if (Test-Path $OutputPath) {
    Remove-Item $OutputPath -Force
}

# Zip contents of train_subset (images/, labels/, targets/ at archive root)
$items = @(
    (Join-Path $SubsetDir "images"),
    (Join-Path $SubsetDir "labels"),
    (Join-Path $SubsetDir "targets")
)
Compress-Archive -Path $items -DestinationPath $OutputPath -CompressionLevel Optimal

$sizeMb = [math]::Round((Get-Item $OutputPath).Length / 1MB, 1)
Write-Host "Wrote $OutputPath ($sizeMb MB)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. https://www.kaggle.com/datasets -> New Dataset"
Write-Host "  2. Upload $OutputPath"
Write-Host "  3. Title: disasteriq-train-subset (private)"
Write-Host "  4. Open notebooks/kaggle_finetune.ipynb on Kaggle — see docs/KAGGLE_FINETUNE.md"
