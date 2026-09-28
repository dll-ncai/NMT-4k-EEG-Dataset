#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-/path/to/NMT_processed}"
cd "$PROJECT_ROOT"

python check_labram_environment.py
python prepare_nmt_manifest.py \
  --inspection-csv data/windows_inspection_report.csv \
  --data-root "$DATA_ROOT" \
  --output data/nmt_finetune_manifest.csv \
  --val-fraction 0.15 \
  --seed 2026 \
  --verify-files sample

python run_nmt_finetuning.py \
  --manifest data/nmt_finetune_manifest.csv \
  --checkpoint LaBraM/checkpoints/labram-base.pth \
  --output-dir outputs/nmt_labram_base \
  --epochs 30 \
  --batch-size 16 \
  --update-freq 4 \
  --epoch-size 50000 \
  --num-workers 3 \
  --amp-dtype auto \
  --allow-unsafe-checkpoint
