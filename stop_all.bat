@echo off
chcp 65001 >nul
title Chinese SRT Extractor - Stop
cd /d "%~dp0"
echo Stopping Chinese SRT Extractor services...
for /f "tokens=*" %%p in ('powershell -NoProfile -Command "Get-CimInstance Win32_Process ^| Where-Object { $_.Name -match '^(python|pythonw)\.exe$' -and $_.CommandLine -match '(app|bot)\.py' } ^| Select-Object -ExpandProperty ProcessId"') do (
    taskkill /PID %%p /T /F >nul 2>&1
    echo Stopped PID %%p
)
echo Done.
timeout /t 2 >nul
