# Extract Kaggle xBD zip (train-nested or tier3-flat) into data/{train|tier3}
param(
    [string]$ArchivePath = "D:\archive.zip",
    [string]$RepoRoot = "",
    [string]$OutputDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

if (-not $RepoRoot) {
    $RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
}

if (-not (Test-Path $ArchivePath)) {
    Write-Host "Archive not found: $ArchivePath" -ForegroundColor Red
    exit 1
}

# Detect layout from zip without full extract
$z = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
try {
    $entries = $z.Entries
    $tier3Count = ($entries | Where-Object { $_.FullName -match '^tier3/images/.*_post_disaster\.png$' }).Count
    $trainCount = ($entries | Where-Object { $_.FullName -match '^train/train/images/.*_post_disaster\.png$' }).Count
} finally {
    $z.Dispose()
}

if ($tier3Count -gt 0) {
    $layout = "tier3"
    $srcImages = "tier3\images"
    $srcLabels = "tier3\labels"
    if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot "data\tier3" }
} elseif ($trainCount -gt 0) {
    $layout = "train-nested"
    $srcImages = "train\train\images"
    $srcLabels = "train\train\labels"
    if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot "data\train" }
} else {
    Write-Host "Unrecognized archive layout in $ArchivePath" -ForegroundColor Red
    exit 1
}

$imagesDir = Join-Path $OutputDir "images"
$labelsDir = Join-Path $OutputDir "labels"
$staging = Join-Path $RepoRoot "data\_kaggle_staging_$layout"

Write-Host "Layout: $layout -> $OutputDir" -ForegroundColor Cyan

if ((Test-Path $imagesDir) -and -not $Force) {
    $count = (Get-ChildItem $imagesDir -File -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($count -gt 500) {
        Write-Host "Already extracted: $count images in $imagesDir. Use -Force to re-extract." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "Extracting $ArchivePath ..." -ForegroundColor Cyan
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force -Path $staging | Out-Null
Expand-Archive -Path $ArchivePath -DestinationPath $staging -Force

$srcImagesPath = Join-Path $staging $srcImages
$srcLabelsPath = Join-Path $staging $srcLabels
if (-not (Test-Path $srcImagesPath)) {
    Write-Host "Expected $srcImages inside archive" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $imagesDir, $labelsDir | Out-Null
Write-Host "Copying images..."
Copy-Item -Path (Join-Path $srcImagesPath "*") -Destination $imagesDir -Force
if (Test-Path $srcLabelsPath) {
    Write-Host "Copying labels..."
    Copy-Item -Path (Join-Path $srcLabelsPath "*") -Destination $labelsDir -Force
}

$imgCount = (Get-ChildItem $imagesDir -File | Measure-Object).Count
$lblCount = (Get-ChildItem $labelsDir -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "Done: $imgCount images, $lblCount labels -> $OutputDir" -ForegroundColor Green

Write-Host "Removing staging folder..."
Remove-Item $staging -Recurse -Force
Write-Host "Next: python scripts/generate_train_targets.py --data-dir $OutputDir"
