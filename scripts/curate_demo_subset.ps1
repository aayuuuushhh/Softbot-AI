# Copy curated demo pairs from extracted test set into data/demo/
# Option A: test already extracted to data/test/
# Option B: pass -TarPath to pull only manifest pairs from the archive (no full extract)

param(
    [string]$TestDir = "",
    [string]$DemoDir = "",
    [string]$Manifest = "",
    [string]$TarPath = ""
)

$ErrorActionPreference = "Stop"
if (-not $TestDir) { $TestDir = Join-Path $PSScriptRoot "..\data\test" }
if (-not $DemoDir) { $DemoDir = Join-Path $PSScriptRoot "..\data\demo" }
if (-not $Manifest) { $Manifest = Join-Path $PSScriptRoot "..\data\demo\manifest.json" }
$DemoDir = (Resolve-Path $DemoDir).Path
$Manifest = (Resolve-Path $Manifest).Path

$manifest = Get-Content $Manifest -Raw | ConvertFrom-Json
foreach ($sub in @("images", "labels", "targets")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DemoDir $sub) | Out-Null
}

function Copy-PairFiles {
    param(
        [string]$Root,
        [string]$Base
    )
    $files = @(
        @{ Sub = "images"; Name = "${Base}_pre_disaster.png" },
        @{ Sub = "images"; Name = "${Base}_post_disaster.png" },
        @{ Sub = "labels"; Name = "${Base}_pre_disaster.json" },
        @{ Sub = "labels"; Name = "${Base}_post_disaster.json" },
        @{ Sub = "targets"; Name = "${Base}_pre_disaster_target.png" },
        @{ Sub = "targets"; Name = "${Base}_post_disaster_target.png" }
    )
    foreach ($f in $files) {
        $src = Join-Path (Join-Path $Root $f.Sub) $f.Name
        $dest = Join-Path (Join-Path $DemoDir $f.Sub) $f.Name
        if (-not (Test-Path $src)) {
            Write-Error "Missing $src"
            exit 1
        }
        Copy-Item $src $dest -Force
    }
    Write-Host "Copied pair: $Base"
}

function Get-MemberPaths([string]$Base) {
    return @(
        "test/images/${Base}_pre_disaster.png",
        "test/images/${Base}_post_disaster.png",
        "test/labels/${Base}_pre_disaster.json",
        "test/labels/${Base}_post_disaster.json",
        "test/targets/${Base}_pre_disaster_target.png",
        "test/targets/${Base}_post_disaster_target.png"
    )
}

if ($TarPath -and (Test-Path $TarPath)) {
    $staging = Join-Path $PSScriptRoot "..\data\_demo_staging"
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    try {
        foreach ($base in $manifest.pairs) {
            foreach ($member in (Get-MemberPaths $base)) {
                $tarOut = & tar -xf $TarPath -C $staging $member 2>&1
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "Failed to extract ${member}: $tarOut"
                    exit 1
                }
            }
        }
        $extractRoot = Join-Path $staging "test"
        foreach ($base in $manifest.pairs) {
            Copy-PairFiles -Root $extractRoot -Base $base
        }
    } finally {
        Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    if (-not (Test-Path (Join-Path $TestDir "images"))) {
        Write-Error "data/test not found. Extract test archive or pass -TarPath D:\test_images_labels_targets.tar"
        exit 1
    }
    foreach ($base in $manifest.pairs) {
        Copy-PairFiles -Root $TestDir -Base $base
    }
}

$pairCount = (Get-ChildItem (Join-Path $DemoDir "images") -Filter "*_pre_disaster.png").Count
Write-Host "Done. Demo pairs: $pairCount (expect 10)"
