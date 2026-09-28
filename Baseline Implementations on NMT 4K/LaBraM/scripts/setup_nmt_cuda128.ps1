param(
    [switch]$ReinstallTorch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($ReinstallTorch) {
    python -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
else {
    python -c "import torch, torchvision, torchaudio; assert torch.__version__.startswith('2.10.0'); assert torchvision.__version__.startswith('0.25.0'); assert torchaudio.__version__.startswith('2.10.0'); assert str(torch.version.cuda).startswith('12.8')"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "The required PyTorch CUDA 12.8 trio is missing or mismatched. Rerun this script with -ReinstallTorch."
        exit $LASTEXITCODE
    }
}

python -m pip install --upgrade -r requirements_nmt_cuda128.txt -c constraints_nmt_cuda128.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python check_labram_environment.py
exit $LASTEXITCODE
