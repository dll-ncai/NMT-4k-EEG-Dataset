@echo off
cd /d "%~dp0\.."
echo Run this only after all model and threshold choices are fixed on validation.
choice /M "Evaluate the untouched official NMT partition now"
if errorlevel 2 exit /b 0
python -m eegpt_nmt.evaluate --config configs/default.yaml --confirm-final-test
pause

