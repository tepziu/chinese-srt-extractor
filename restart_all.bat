@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\services.ps1" -Action restart -Service all
if errorlevel 1 (
    pause
    exit /b 1
)
