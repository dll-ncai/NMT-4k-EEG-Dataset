@echo off
setlocal
call conda activate scnet_nmt4k
if errorlevel 1 exit /b 1

python check_setup.py --config config.yaml
if errorlevel 1 exit /b 1

python preprocess.py --config config.yaml
if errorlevel 1 exit /b 1

python train.py --config config.yaml --resume auto
if errorlevel 1 exit /b 1

python evaluate.py --config config.yaml --threshold auto
if errorlevel 1 exit /b 1

echo Pipeline complete.
