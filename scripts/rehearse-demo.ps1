# Rehearse demo flow before recording video
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

Write-Host "=== DisasterIQ demo rehearsal ===" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $root ".env"))) {
    Write-Host "Missing .env - copy .env.example and set FIREWORKS_API_KEY" -ForegroundColor Red
    exit 1
}

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health"
} catch {
    Write-Host "Backend not running. Start: .\scripts\start-backend.ps1" -ForegroundColor Red
    exit 1
}
Write-Host "Backend OK - inference_mode=$($health.inference_mode), pairs=$($health.demo_pairs)"

$pairs = Invoke-RestMethod -Uri "http://localhost:8000/demo/pairs"
$preferred = "river-valley-flood_01"
$pairId = if ($pairs.id -contains $preferred) { $preferred } else { $pairs[0].id }
Write-Host "Analyzing demo pair: $pairId"

$analysisJson = curl.exe -s -X POST "http://localhost:8000/analyze" -F "demo_pair_id=$pairId"
$analysis = $analysisJson | ConvertFrom-Json
Write-Host "Zones ranked: $($analysis.zones.Count), destroyed_pct=$($analysis.summary.destroyed_pct)%"

$briefBody = @{
    analysis = $analysis
    context  = "Pakistan disaster response rehearsal"
} | ConvertTo-Json -Depth 10

$brief = Invoke-RestMethod -Uri "http://localhost:8000/brief" -Method Post -Body $briefBody -ContentType "application/json"
$color = if ($brief.source -eq "fireworks") { "Green" } else { "Yellow" }
Write-Host "Brief source: $($brief.source)" -ForegroundColor $color
$previewLen = [Math]::Min(120, $brief.brief.Length)
Write-Host "Brief preview: $($brief.brief.Substring(0, $previewLen))..."

Write-Host ""
Write-Host "Ready to record:" -ForegroundColor Green
Write-Host "  1. INFERENCE_MODE=stub in .env (fast masks for video)"
Write-Host "  2. Frontend: .\scripts\start-frontend.ps1 -> http://localhost:3000"
Write-Host "  3. Follow docs/SUBMISSION.md script"
Write-Host "  4. Submit lablab.ai with GitHub URL + video"
