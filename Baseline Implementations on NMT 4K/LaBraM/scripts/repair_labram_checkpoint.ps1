$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "LaBraM\.git")) {
    Write-Error "LaBraM\.git was not found. Run this from the original project that contains the cloned LaBraM repository."
    exit 1
}

git -C LaBraM cat-file -e "HEAD:checkpoints/labram-base.pth"
if ($LASTEXITCODE -ne 0) {
    Write-Error "The intact checkpoint is not present in the embedded LaBraM Git history."
    exit $LASTEXITCODE
}

git -C LaBraM restore checkpoints/labram-base.pth
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -c "from pathlib import Path; import zipfile; p=Path(r'LaBraM/checkpoints/labram-base.pth'); z=zipfile.ZipFile(p); bad=z.testzip(); z.close(); assert bad is None, f'Corrupt checkpoint member: {bad}'; assert p.stat().st_size == 96612769, f'Unexpected checkpoint size: {p.stat().st_size}'; print(f'Checkpoint repaired and verified: {p} ({p.stat().st_size:,} bytes)')"
exit $LASTEXITCODE
