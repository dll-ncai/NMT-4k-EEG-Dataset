$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

& "$ProjectRoot\01_verify_install.ps1"
& "$ProjectRoot\02_inspect_dataset.ps1"
& "$ProjectRoot\03_preprocess.ps1"
& "$ProjectRoot\04_train.ps1"
& "$ProjectRoot\05_evaluate.ps1"

Write-Host "Full pipeline completed." -ForegroundColor Green
