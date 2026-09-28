@echo off
setlocal
call conda activate nmt-deep4
if errorlevel 1 exit /b 1

python 00_check_gpu.py || exit /b 1
python 05_smoke_test_model.py || exit /b 1
python 01_prepare_manifest.py --config config.yaml || exit /b 1
python 02_preprocess_cache.py --config config.yaml || exit /b 1
python 03_train.py --config config.yaml || exit /b 1
python 04_evaluate.py --config config.yaml --split evaluation || exit /b 1

echo.
echo Complete. See the run artifacts under the work_dir configured in config.yaml.
