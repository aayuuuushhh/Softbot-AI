# Verify Fireworks brief wiring (uses repo .env, never prints the key)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$envFile = Join-Path $root ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "Missing .env" -ForegroundColor Red
    exit 1
}

Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        Set-Item -Path "Env:$($matches[1].Trim())" -Value $matches[2].Trim()
    }
}

if (-not $env:FIREWORKS_API_KEY) {
    Write-Host "FIREWORKS_API_KEY not set in .env" -ForegroundColor Red
    exit 1
}

Push-Location (Join-Path $root "backend")
try {
    $diag = & .\.venv\Scripts\python.exe -c @"
import httpx
from app.config import settings
r = httpx.get('https://api.fireworks.ai/inference/v1/models',
    headers={'Authorization': f'Bearer {settings.fireworks_api_key}'}, timeout=30)
print(r.status_code)
print(r.text[:300])
"@
    if ($diag -match 'suspended') {
        Write-Host "Fireworks account is suspended (billing/credits). Fix at:" -ForegroundColor Red
        Write-Host "  https://fireworks.ai/account/billing" -ForegroundColor Yellow
        Write-Host "Demo still works via fireworks-fallback stub brief." -ForegroundColor Gray
        exit 3
    }

    $out = & .\.venv\Scripts\python.exe -c @"
import asyncio
from app.services.narrator import generate_brief
async def main():
    r = await generate_brief(
        {'summary': {'total_building_pixels': 100, 'destroyed_pct': 1, 'major_pct': 2, 'minor_pct': 3}, 'zones': []},
        'verification',
    )
    print(r.source)
asyncio.run(main())
"@
    Write-Host "Brief source: $out"
    if ($out -eq "fireworks") {
        Write-Host "Live Fireworks brief OK" -ForegroundColor Green
        exit 0
    }
    if ($out -eq "fireworks-fallback") {
        Write-Host "Key loaded but Fireworks API error - get a full API key from app.fireworks.ai" -ForegroundColor Yellow
        exit 2
    }
    Write-Host "Still using stub - restart backend after editing .env" -ForegroundColor Yellow
    exit 1
} finally {
    Pop-Location
}
