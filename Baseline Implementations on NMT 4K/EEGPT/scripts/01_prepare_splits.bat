@echo off
cd /d "%~dp0\.."
python -m eegpt_nmt.prepare_splits --config configs/default.yaml
pause

