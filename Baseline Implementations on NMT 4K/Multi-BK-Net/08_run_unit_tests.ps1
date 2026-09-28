$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run .\00_create_environment.ps1 first." }
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }
