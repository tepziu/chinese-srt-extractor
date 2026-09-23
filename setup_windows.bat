@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,13)" >nul 2>&1
if errorlevel 1 (
    echo Python 3.10-3.12 is required for this pinned audio stack.
    goto :fail
)
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo Install FFmpeg and add it to PATH first.
    goto :fail
)
ffprobe -version >nul 2>&1
if errorlevel 1 goto :fail
if not exist "venv\Scripts\python.exe" python -m venv venv
if errorlevel 1 goto :fail
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
nvidia-smi >nul 2>&1
if errorlevel 1 (
    "venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    "venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
if errorlevel 1 goto :fail
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
"venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :fail
if not exist uploads mkdir uploads
if not exist outputs mkdir outputs
echo Setup completed. Use start_all.bat. Logs are in runtime.
exit /b 0
:fail
echo Setup failed. Review the error above; no existing environment was deleted.
pause
exit /b 1
