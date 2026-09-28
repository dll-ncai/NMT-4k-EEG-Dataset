@echo off
setlocal

echo ============================================================
echo BD-Deep4 NMT-4K-EEG Windows environment setup
echo ============================================================
echo.
echo This script assumes Miniconda/Anaconda is installed and conda is on PATH.
echo It creates the environment, then installs a CUDA-enabled PyTorch wheel.
echo If PyTorch changes its current CUDA wheel, use the selector at pytorch.org.
echo.

call conda create -n nmt-deep4 python=3.11 -y
if errorlevel 1 goto :error
call conda activate nmt-deep4
if errorlevel 1 goto :error
python -m pip install --upgrade pip
if errorlevel 1 goto :error

REM CUDA 12.8 wheel is a good Windows choice for modern NVIDIA drivers.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :torch_error

python -m pip install -r requirements.txt
if errorlevel 1 goto :error

python 00_check_gpu.py
if errorlevel 1 goto :error

echo.
echo Environment setup finished successfully.
echo Activate later with: conda activate nmt-deep4
goto :eof

:torch_error
echo.
echo PyTorch CUDA installation failed.
echo Open https://pytorch.org/get-started/locally/ and choose:
echo Windows / Pip / Python / a CUDA build supported by your NVIDIA driver.
goto :error

:error
echo.
echo Setup did not finish. Review the error above.
exit /b 1
