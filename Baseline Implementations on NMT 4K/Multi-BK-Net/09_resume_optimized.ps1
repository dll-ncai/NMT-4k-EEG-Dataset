$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run .\00_create_environment.ps1 first." }

$LastCheckpoint = Join-Path "G:\Dataset\NMT-4K-EEG\multibknet_nmt_outputs" "runs\seed_2026\development\last.pt"
if (-not (Test-Path $LastCheckpoint)) {
    throw "No completed-epoch checkpoint exists. Use .\04_train.ps1 instead."
}

# Runtime-only throughput settings. These do not alter config.yaml or its hash.
# The configured effective batch remains exactly 64 windows.
$env:MBK_RUNTIME_MICRO_BATCH_SIZE = "64"
$env:MBK_RUNTIME_ACCUMULATION_STEPS = "1"
$env:MBK_RUNTIME_VALIDATION_BATCH_SIZE = "64"
$env:MBK_RUNTIME_NUM_WORKERS = "2"
$env:MBK_RUNTIME_PREFETCH_FACTOR = "2"
$env:MBK_RUNTIME_MEMMAP_CACHE_SIZE = "128"

Write-Host "Resume optimization" -ForegroundColor Cyan
Write-Host "  micro-batch       : 64"
Write-Host "  accumulation      : 1"
Write-Host "  effective batch   : 64 (unchanged)"
Write-Host "  loader workers    : 2"
Write-Host "  prefetch per worker: 2"
Write-Host "  memory-map cache  : 128 recordings per worker"

& $Python -m multibknet_nmt.train --config config.yaml --resume
if ($LASTEXITCODE -ne 0) { throw "Optimized resume failed." }
