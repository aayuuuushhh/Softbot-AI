# Run this script as Administrator (right-click PowerShell -> Run as administrator)
# Installs WSL 2 required for Docker Desktop. Reboot required after completion.

Write-Host "Installing Windows Subsystem for Linux (WSL 2)..." -ForegroundColor Cyan

$features = @(
    "Microsoft-Windows-Subsystem-Linux",
    "VirtualMachinePlatform"
)

foreach ($feature in $features) {
    $state = (Get-WindowsOptionalFeature -Online -FeatureName $feature -ErrorAction SilentlyContinue).State
    if ($state -ne "Enabled") {
        Write-Host "Enabling $feature..."
        Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart | Out-Null
    } else {
        Write-Host "$feature already enabled."
    }
}

Write-Host "Running wsl --install..."
wsl --install --no-launch

Write-Host ""
Write-Host "WSL install initiated. REBOOT your PC, then:" -ForegroundColor Yellow
Write-Host "  1. Open Docker Desktop and wait until Running"
Write-Host "  2. Run: docker ps"
Write-Host "  3. Tell Cursor: prerequisites done (for Docker verify)"
