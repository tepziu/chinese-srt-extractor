@echo off
chcp 65001 >nul
title Chinese SRT Extractor - All Services
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo ============================================================
echo   Chinese SRT Extractor - All Services
echo ============================================================
echo.
echo   Web App:      http://127.0.0.1:5000
echo   Telegram Bot: Starting...
echo.
echo ============================================================
echo.

start "Web App" cmd /k "chcp 65001 >nul && cd /d \"%~dp0\" && if exist \"venv\Scripts\python.exe\" (venv\Scripts\python.exe app.py) else (python app.py)"
timeout /t 3 >nul
start "Telegram Bot" cmd /k "chcp 65001 >nul && cd /d \"%~dp0\" && if exist \"venv\Scripts\python.exe\" (venv\Scripts\python.exe bot.py) else (python bot.py)"

echo All services started!
pause >nul
