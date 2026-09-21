# Start Docker Desktop service (requires Administrator)
Write-Host "Starting com.docker.service..." -ForegroundColor Cyan
try {
    Start-Service com.docker.service -ErrorAction Stop
    Write-Host "Service started." -ForegroundColor Green
} catch {
    Write-Host "Failed: $_" -ForegroundColor Red
    exit 1
}
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Write-Host "Docker Desktop launched. Wait 1-2 minutes then run: docker ps"
