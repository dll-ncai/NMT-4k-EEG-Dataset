@echo off
cd /d "%~dp0\.."
python -m eegpt_nmt.preprocess --config configs/default.yaml
pause

