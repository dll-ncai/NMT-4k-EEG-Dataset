@echo off
cd /d "%~dp0\.."
python -m eegpt_nmt.audit_protocol --config configs/default.yaml --allow-incomplete-evaluation-for-training
pause
