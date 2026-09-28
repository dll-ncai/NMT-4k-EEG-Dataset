@echo off
cd /d "%~dp0\.."
python -m eegpt_nmt.verify_setup --config configs/default.yaml
pause

