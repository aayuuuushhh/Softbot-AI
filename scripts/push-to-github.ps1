# Creates public GitHub repo and pushes (requires: gh auth login)
param(
    [string]$RepoName = "satellite-disaster-triage",
    [string]$Description = "Satellite Disaster-Damage Triage — AMD ACT II Hackathon (Team DarkNem)"
)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "GitHub CLI (gh) not found. Install: winget install GitHub.cli"
    exit 1
}

$auth = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Run: gh auth login" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path ".git")) {
    Write-Error "Not a git repo. Run git init first."
    exit 1
}

$existing = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote origin already set: $existing"
    git push -u origin HEAD
    exit $LASTEXITCODE
}

gh repo create $RepoName --public --source=. --remote=origin --description $Description --push
if ($LASTEXITCODE -eq 0) {
    Write-Host "Repo created and pushed. Share: $(gh repo view --json url -q .url)"
}
