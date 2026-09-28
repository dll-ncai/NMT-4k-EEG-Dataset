param(
    [string]$Config = "config.yaml"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Checking setup..."
python 00_check_setup.py --config $Config

Write-Host "[2/5] Creating train/validation/evaluation manifest..."
python 01_make_splits.py --config $Config

Write-Host "[3/5] Preprocessing/caching EDFs (resumable)..."
python 02_preprocess.py --config $Config

Write-Host "[4/5] Training (auto-resumes from last.pt if present)..."
python 03_train.py --config $Config --resume auto

Write-Host "[5/5] Evaluating held-out evaluation split..."
python 04_evaluate.py --config $Config --split evaluation
