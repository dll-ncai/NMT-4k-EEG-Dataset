@echo off
setlocal

echo Creating conda environment scnet_nmt4k with Python 3.11...
call conda create -n scnet_nmt4k python=3.11 -y
if errorlevel 1 goto :error

call conda activate scnet_nmt4k
if errorlevel 1 goto :error

python -m pip install --upgrade pip
if errorlevel 1 goto :error

REM Stable CUDA-enabled PyTorch build known to support Windows and CUDA 12.8.
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error

python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Installation complete.
echo Activate later with: conda activate scnet_nmt4k
echo Then run: python check_setup.py --config config.yaml
exit /b 0

:error
echo.
echo Installation failed. Check the error above.
exit /b 1
