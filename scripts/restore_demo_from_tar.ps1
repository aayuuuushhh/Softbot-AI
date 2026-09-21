# Restore data/demo from tar — PowerShell 5.1 compatible
param(
    [string]$TarPath = "D:\test_images_labels_targets.tar",
    [string]$DemoDir = "",
    [string]$Manifest = ""
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path $PSScriptRoot -Parent
if (-not $DemoDir) { $DemoDir = Join-Path $RootDir "data\demo" }
if (-not $Manifest) { $Manifest = Join-Path $RootDir "data\demo\manifest.json" }
$DemoDir = (Resolve-Path $DemoDir).Path
$Manifest = (Resolve-Path $Manifest).Path
$manifestJson = Get-Content $Manifest -Raw | ConvertFrom-Json
$demoPairs = @($manifestJson.pairs)
$staging = Join-Path $RootDir "data\_demo_staging"

foreach ($sub in @("images", "labels", "targets")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DemoDir $sub) | Out-Null
}

Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $staging | Out-Null

Write-Host "Extracting $($demoPairs.Count) pairs from $TarPath ..."

foreach ($base in $demoPairs) {
    $members = @(
        "test/images/${base}_pre_disaster.png",
        "test/images/${base}_post_disaster.png",
        "test/labels/${base}_pre_disaster.json",
        "test/labels/${base}_post_disaster.json",
        "test/targets/${base}_pre_disaster_target.png",
        "test/targets/${base}_post_disaster_target.png"
    )
    foreach ($member in $members) {
        tar -xf $TarPath -C $staging $member
        if ($LASTEXITCODE -ne 0) { throw "tar failed: $member" }
    }
    $root = Join-Path $staging "test"
    foreach ($sub in @("images", "labels", "targets")) {
        $srcDir = Join-Path $root $sub
        $destDir = Join-Path $DemoDir $sub
        Get-ChildItem $srcDir -Filter "${base}_*" | Copy-Item -Destination $destDir -Force
    }
    Write-Host "  OK: $base"
}

Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
$n = (Get-ChildItem (Join-Path $DemoDir "images") -Filter "*_pre_disaster.png").Count
Write-Host "Done. Demo pairs: $n"
