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
Write-Host "Verifying the bundle..."
& .\dist\SEMSimulator.exe --list-examples
# A packaged build can open its window and list its examples while being
# quietly broken: PyInstaller misses the scipy submodules imported lazily, so
# the failure lands the first time a p-value is computed. --self-test runs a
# whole simulation headlessly, so a bad bundle fails here.
& .\dist\SEMSimulator.exe --self-test
if ($LASTEXITCODE -ne 0) { throw "self-test failed" }
Write-Host "Verification passed."
