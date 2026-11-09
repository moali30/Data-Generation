@echo off
REM Build SEMSimulator.exe on Windows.
REM Run from the repository root:  packaging\build_windows.bat

setlocal
cd /d "%~dp0\.."

echo [1/4] Creating a clean virtual environment...
if exist .buildenv rmdir /s /q .buildenv
python -m venv .buildenv || goto :fail
call .buildenv\Scripts\activate.bat

echo [2/4] Installing dependencies...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt || goto :fail
python -m pip install pyinstaller || goto :fail

echo [3/4] Running the test suite...
python -m pip install pytest >nul
python -m pytest tests -q || goto :fail

echo [4/4] Building the executable...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller packaging\semsim.spec --noconfirm --clean || goto :fail

echo.
echo Done. The executable is at dist\SEMSimulator.exe
echo Verifying the bundle...
dist\SEMSimulator.exe --list-examples || goto :fail
REM A packaged build can open its window and list its examples while being
REM quietly broken: PyInstaller misses the scipy submodules imported lazily,
REM so the failure lands the first time a p-value is computed. --self-test
REM runs a whole simulation headlessly, so a bad bundle fails here.
dist\SEMSimulator.exe --self-test || goto :fail
echo Verification passed.
goto :eof

:fail
echo.
echo BUILD FAILED. See the output above.
exit /b 1
