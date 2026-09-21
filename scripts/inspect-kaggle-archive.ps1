# Inspect any Kaggle xBD zip without full extract
param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

if (-not (Test-Path $ArchivePath)) {
    Write-Host "[FAIL] Not found: $ArchivePath" -ForegroundColor Red
    exit 1
}

$sizeGb = [math]::Round((Get-Item $ArchivePath).Length / 1GB, 2)
Write-Host "Archive: $ArchivePath ($sizeGb GB)" -ForegroundColor Cyan

$z = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
try {
    $entries = $z.Entries
    Write-Host "Total zip entries: $($entries.Count)"

    Write-Host "`nTop-level path segments:"
    $entries |
        ForEach-Object {
            $parts = $_.FullName -split '[\\/]'
            if ($parts.Count -ge 1 -and $parts[0]) { $parts[0] }
        } |
        Group-Object |
        Sort-Object Count -Descending |
        Select-Object -First 10 Name, Count |
        Format-Table -AutoSize

    Write-Host "Sample paths:"
    $entries | Select-Object -First 15 -ExpandProperty FullName

    foreach ($pattern in @(
        @{ Name = "pre_disaster png";  Re = '_pre_disaster\.png$' },
        @{ Name = "post_disaster png"; Re = '_post_disaster\.png$' },
        @{ Name = "post labels json";  Re = '_post_disaster\.json$' },
        @{ Name = "target png";        Re = '_post_disaster_target\.png$' },
        @{ Name = "target png alt";    Re = 'targets/.*\.png$' }
    )) {
        $c = ($entries | Where-Object { $_.FullName -match $pattern.Re }).Count
        Write-Host ("{0,-20} {1}" -f $pattern.Name, $c)
    }
} finally {
    $z.Dispose()
}
