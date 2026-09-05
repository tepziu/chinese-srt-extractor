@echo off
chcp 65001 >nul
title Chinese SRT Extractor - Stop
cd /d "%~dp0"
echo Dang tat tat ca dich vu...

taskkill /IM ffmpeg.exe /F >nul 2>&1

powershell -NoProfile -Command "Get-Process python*, py* -ErrorAction SilentlyContinue | Where-Object { $id = $_.Id; $cmd = (Get-CimInstance Win32_Process -Filter \"ProcessId = $id\" -ErrorAction SilentlyContinue).CommandLine; $cmd -match 'app\.py|bot\.py' } | ForEach-Object { Write-Host \"Tat Python PID $($_.Id)\"; Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }"

taskkill /FI "WINDOWTITLE eq Chinese SRT Extractor - Web*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Chinese SRT Extractor - Telegram*" /F >nul 2>&1

echo Hoan tat tat ca dich vu!
ping 127.0.0.1 -n 3 >nul
