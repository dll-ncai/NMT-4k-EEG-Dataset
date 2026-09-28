param(
    [string]$Checkpoint = "outputs\nmt_labram_base\best.pt",
    [int]$BatchSize = 16
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python run_nmt_finetuning.py `
    --manifest "data\nmt_finetune_manifest.csv" `
    --checkpoint "LaBraM\checkpoints\labram-base.pth" `
    --resume $Checkpoint `
    --output-dir "outputs\nmt_labram_base" `
    --eval-only `
    --batch-size $BatchSize `
    --num-workers 3 `
    --amp-dtype auto `
    --allow-unsafe-checkpoint
exit $LASTEXITCODE
