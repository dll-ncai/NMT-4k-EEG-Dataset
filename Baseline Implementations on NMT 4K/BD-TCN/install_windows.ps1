$ErrorActionPreference = "Stop"

Write-Host "Creating Conda environment: bdtcn-nmt"
conda create -n bdtcn-nmt python=3.11 -y

Write-Host "Installing CUDA-enabled PyTorch..."
conda run -n bdtcn-nmt python -m pip install --upgrade pip
conda run -n bdtcn-nmt python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing Braindecode and scientific Python dependencies..."
conda run -n bdtcn-nmt python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Installation finished."
Write-Host "Next:"
Write-Host "  conda activate bdtcn-nmt"
Write-Host "  python 01_check_setup.py"
