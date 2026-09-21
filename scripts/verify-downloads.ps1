# Verify xBD archive MD5 checksums (Windows)
param(
    [string]$TestTar = "",
    [string]$TrainTar = "",
    [string]$Tier3Tar = ""
)

$Expected = @{
    test  = "1b39c47e05d1319c17cc8763cee6fe0c"
    train = "a20ebbfb7eb3452785b63ad02ffd1e16"
}

function Find-Archive([string]$Name, [string[]]$Candidates) {
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    $found = Get-ChildItem -Path D:\ -Filter $Name -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    return $found?.FullName
}

if (-not $TestTar) {
    $TestTar = Find-Archive "test_images_labels_targets.tar" @(
        "D:\test_images_labels_targets.tar",
        "D:\AMD\data\test_images_labels_targets.tar"
    )
}
if (-not $TrainTar) {
    $TrainTar = Find-Archive "train_images_labels_targets.tar.gz" @(
        "D:\train_images_labels_targets.tar.gz",
        "D:\AMD\data\train_images_labels_targets.tar.gz"
    )
}

$allOk = $true
foreach ($item in @(
    @{ Label = "test"; Path = $TestTar; Md5 = $Expected.test },
    @{ Label = "train"; Path = $TrainTar; Md5 = $Expected.train }
)) {
    if (-not $item.Path) {
        Write-Host "[SKIP] $($item.Label) archive not found" -ForegroundColor Yellow
        continue
    }
    $hash = (Get-FileHash -Algorithm MD5 -Path $item.Path).Hash.ToLower()
    if ($hash -eq $item.Md5) {
        Write-Host "[OK] $($item.Label): $hash ($($item.Path))" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $($item.Label): expected $($item.Md5), got $hash" -ForegroundColor Red
        $allOk = $false
    }
}

if ($Tier3Tar -and (Test-Path $Tier3Tar)) {
    $hash = (Get-FileHash -Algorithm MD5 -Path $Tier3Tar).Hash.ToLower()
    Write-Host "[INFO] tier3: $hash ($Tier3Tar)"
}

if (-not $allOk) { exit 1 }
