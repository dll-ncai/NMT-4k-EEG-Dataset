@echo off
cd /d "%~dp0\.."
echo ================================================================
echo PROVISIONAL INCOMPLETE-COHORT EVALUATION: 997 OF 1000 RECORDINGS
echo ================================================================
echo This does not produce the official NMT-4K 1000-record benchmark.
echo The selected checkpoint and validation threshold must stay frozen.
echo Outputs will be saved in a separate provisional directory.
choice /M "Evaluate the available 997-record partition now"
if errorlevel 2 exit /b 0
python -m eegpt_nmt.evaluate --config configs/default.yaml --confirm-provisional-incomplete-test --expected-available-records 997
pause
