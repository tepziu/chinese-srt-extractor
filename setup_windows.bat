@echo off
chcp 65001 >nul
title Chinese SRT Extractor - Setup
cd /d "%~dp0"
python --version >nul 2>&1 || (echo Python 3.10+ is required.& pause& exit /b 1)
ffmpeg -version >nul 2>&1 || echo WARNING: install FFmpeg and add it to PATH before running.
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if errorlevel 1 rmdir /s /q venv
)
if not exist "venv\Scripts\python.exe" python -m venv venv
call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
) else (
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
)
if not exist uploads mkdir uploads
if not exist outputs mkdir outputs
echo Setup completed. Use start_all.bat to run services.
pause

