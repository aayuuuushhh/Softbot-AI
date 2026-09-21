# Verify Kaggle xBD archive (train or tier3 layout)
param(
    [string]$ArchivePath = "D:\archive.zip"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

if (-not (Test-Path $ArchivePath)) {
    Write-Host "[FAIL] Archive not found: $ArchivePath" -ForegroundColor Red
    exit 1
}

$sizeGb = [math]::Round((Get-Item $ArchivePath).Length / 1GB, 2)
$drive = (Split-Path $ArchivePath -Qualifier)
$freeGb = [math]::Round((Get-PSDrive ($drive.TrimEnd(':'))).Free / 1GB, 1)

Write-Host "Archive: $ArchivePath ($sizeGb GB)" -ForegroundColor Cyan
Write-Host "Free space on ${drive}: $freeGb GB"

if ($freeGb -lt 15) {
    Write-Host "[WARN] Recommend at least 15 GB free for extract + targets" -ForegroundColor Yellow
}

$z = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
try {
    $entries = $z.Entries
    $layout = "unknown"

    $trainNested = ($entries | Where-Object { $_.FullName -match '^train/train/images/.*_post_disaster\.png$' }).Count
    $tier3Flat = ($entries | Where-Object { $_.FullName -match '^tier3/images/.*_post_disaster\.png$' }).Count
    $tier3Labels = ($entries | Where-Object { $_.FullName -match '^tier3/labels/.*_post_disaster\.json$' }).Count
    $trainLabels = ($entries | Where-Object { $_.FullName -match '^train/train/labels/.*_post_disaster\.json$' }).Count

    if ($tier3Flat -gt 0) {
        $layout = "tier3"
        $imagePrefix = "tier3/images/"
        $labelPrefix = "tier3/labels/"
        $trainPost = $tier3Flat
        $trainPre = ($entries | Where-Object { $_.FullName -match '^tier3/images/.*_pre_disaster\.png$' }).Count
        $trainLabels = $tier3Labels
    } elseif ($trainNested -gt 0) {
        $layout = "train-nested"
        $imagePrefix = "train/train/images/"
        $labelPrefix = "train/train/labels/"
        $trainPost = $trainNested
        $trainPre = ($entries | Where-Object { $_.FullName -match '^train/train/images/.*_pre_disaster\.png$' }).Count
    } else {
        Write-Host "[FAIL] Unrecognized archive layout" -ForegroundColor Red
        exit 1
    }

    Write-Host "Detected layout: $layout"
    Write-Host ""
    Write-Host "Pre images:  $trainPre"
    Write-Host "Post images: $trainPost"
    Write-Host "Post labels: $trainLabels"

  $trainTargets = ($entries | Where-Object { $_.FullName -match 'targets/.*\.png$' }).Count
    Write-Host "Targets in zip: $trainTargets $(if ($trainTargets -eq 0) { '(will generate via convert2png)' } else { '' })"

    if ($layout -eq "tier3") {
        Write-Host ""
        Write-Host "Tier3 disaster types (post images):"
        $disasters = $entries |
            Where-Object { $_.FullName -like "${imagePrefix}*_post_disaster.png" } |
            ForEach-Object {
                $name = [System.IO.Path]::GetFileName($_.FullName)
                if ($name -match '^(.+?)_\d+_post_disaster\.png$') { $Matches[1] } else { "unknown" }
            } |
            Group-Object |
            Sort-Object Count -Descending
        foreach ($d in $disasters) {
            Write-Host ("  {0} : {1} post images ({2} pairs)" -f $d.Name, $d.Count, [math]::Floor($d.Count))
        }
    } else {
        $prefixes = @(
            'mexico-earthquake', 'midwest-flooding', 'nepal-flooding',
            'socal-fire', 'santa-rosa-wildfire'
        )
        Write-Host ""
        Write-Host "Hackathon subset (post images):"
        foreach ($p in $prefixes) {
            $c = ($entries | Where-Object { $_.FullName -like "${imagePrefix}${p}*_post_disaster.png" }).Count
            Write-Host "  $p : $c"
        }
    }

    $sample = $entries | Where-Object { $_.FullName -like "${labelPrefix}*_post_disaster.json" } | Select-Object -First 1
    if ($sample) {
        $sr = New-Object System.IO.StreamReader($sample.Open())
        $txt = $sr.ReadToEnd()
        $sr.Close()
        if ($txt -match '"feature_type"\s*:\s*"building"') {
            Write-Host ""
            Write-Host "[OK] Label JSON schema looks like xBD" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Unexpected label JSON format" -ForegroundColor Yellow
        }
    }

    if ($trainPost -ge 100 -and $trainLabels -ge 100 -and $trainPre -eq $trainPost) {
        Write-Host ""
        Write-Host "[OK] Archive ready for extraction" -ForegroundColor Green
        exit 0
    }
    Write-Host "[FAIL] Unexpected file counts" -ForegroundColor Red
    exit 1
} finally {
    $z.Dispose()
}
