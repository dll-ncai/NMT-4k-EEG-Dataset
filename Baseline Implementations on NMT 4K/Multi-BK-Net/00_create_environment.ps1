[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$SupportedVersions = @("3.10", "3.11", "3.12", "3.13")
$PythonCommand = $null
$PythonArguments = @()
$SelectedVersion = $null
$DetectedDefault = $null

# Prefer the Windows Python launcher because it can select a supported Python
# even when the `python` command still points to a newer side-by-side install.
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($Candidate in $SupportedVersions) {
        try {
            $Probe = & py "-$Candidate" -c "import struct, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}', struct.calcsize('P') * 8, sep='|')" 2>$null
            if (($LASTEXITCODE -eq 0) -and ($Probe.Trim() -eq "$Candidate|64")) {
                $PythonCommand = "py"
                $PythonArguments = @("-$Candidate")
                $SelectedVersion = $Candidate
                break
            }
        } catch {}
    }
}

# Fall back to `python` only when it is itself a supported 64-bit interpreter.
if (-not $PythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
    try {
        $Probe = & python -c "import struct, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}', struct.calcsize('P') * 8, sep='|')"
        if ($LASTEXITCODE -eq 0) {
            $Parts = $Probe.Trim().Split("|")
            $DetectedDefault = "$($Parts[0]) ($($Parts[1])-bit)"
            if (($Parts[0] -in $SupportedVersions) -and ($Parts[1] -eq "64")) {
                $PythonCommand = "python"
                $SelectedVersion = $Parts[0]
            }
        }
    } catch {}
}

if (-not $PythonCommand) {
    if ($DetectedDefault) {
        Write-Host "Detected default Python: $DetectedDefault" -ForegroundColor Yellow
    }
    throw @"
No supported 64-bit Python installation was found.

Install a supported 64-bit Python (3.13 recommended for a new setup):
  winget install --exact --id Python.Python.3.13

Then close and reopen PowerShell, return to this folder, and run:
  py -3.13 --version
  .\00_create_environment.ps1

Supported versions: 3.10, 3.11, 3.12, or 3.13.
"@
}

Write-Host "Using 64-bit Python $SelectedVersion." -ForegroundColor Cyan

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $PythonCommand @PythonArguments -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

# CUDA 12.8 wheels include RTX 50-series (Blackwell) support. Torchvision and
# torchaudio are not needed by this project.
& $Python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed." }

& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }

& $Python -m multibknet_nmt.verify_install --config config.yaml --environment-only
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }

Write-Host ""
Write-Host "Environment created successfully." -ForegroundColor Green
Write-Host "Next command: .\01_verify_install.ps1"
