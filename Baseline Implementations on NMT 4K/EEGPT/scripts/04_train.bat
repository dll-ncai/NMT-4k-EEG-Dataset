@echo off
cd /d "%~dp0\.."
python -m eegpt_nmt.train --config configs/default.yaml --allow-incomplete-evaluation-for-training
pause
