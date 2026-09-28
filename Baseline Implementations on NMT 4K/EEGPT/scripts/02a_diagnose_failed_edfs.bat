@echo off
cd /d "%~dp0\.."
python -m eegpt_nmt.diagnose_edfs --config configs/default.yaml
pause
