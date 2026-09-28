param(
    [string]$DataRoot = "F:\NMT_processed",
    [int]$BatchSize = 16,
    [int]$UpdateFreq = 4,
    [int]$EpochSize = 50000,
    [int]$Epochs = 30,
    [switch]$SkipEnvironmentCheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not $SkipEnvironmentCheck) {
    python check_labram_environment.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python prepare_nmt_manifest.py `
    --inspection-csv "data\windows_inspection_report.csv" `
    --data-root $DataRoot `
    --output "data\nmt_finetune_manifest.csv" `
    --val-fraction 0.15 `
    --seed 2026 `
    --verify-files sample
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python run_nmt_finetuning.py `
    --manifest "data\nmt_finetune_manifest.csv" `
    --checkpoint "LaBraM\checkpoints\labram-base.pth" `
    --output-dir "outputs\nmt_labram_base" `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --update-freq $UpdateFreq `
    --epoch-size $EpochSize `
    --num-workers 3 `
    --amp-dtype auto `
    --allow-unsafe-checkpoint
exit $LASTEXITCODE
