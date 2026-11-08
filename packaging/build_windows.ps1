# Build SEMSimulator.exe on Windows (PowerShell).
# Run from the repository root:  .\packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "[1/4] Creating a clean virtual environment..."
if (Test-Path .buildenv) { Remove-Item -Recurse -Force .buildenv }
python -m venv .buildenv
& .\.buildenv\Scripts\Activate.ps1

Write-Host "[2/4] Installing dependencies..."
python -m pip install --upgrade pip | Out-Null
python -m pip install -r requirements.txt
python -m pip install pyinstaller pytest

Write-Host "[3/4] Running the test suite..."
python -m pytest tests -q

Write-Host "[4/4] Building the executable..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
pyinstaller packaging\semsim.spec --noconfirm --clean

Write-Host ""
Write-Host "Done. The executable is at dist\SEMSimulator.exe"
Write-Host "Smoke-testing it..."
& .\dist\SEMSimulator.exe --list-examples
Write-Host "Smoke test passed."
