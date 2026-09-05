@echo off
chcp 65001 >nul
title Chinese SRT Extractor - All Services
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:: Giai phong port 5000 neu dang bi chiem
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo ============================================================
echo   Chinese SRT Extractor - Khoi chay tat ca dich vu
echo ============================================================
echo.
echo   [1/2] Dang khoi chay Web Studio...
start "Chinese SRT Extractor - Web" "%~dp0start_web.bat"

ping 127.0.0.1 -n 3 >nul

echo   [2/2] Dang khoi chay Telegram Bot...
start "Chinese SRT Extractor - Telegram Bot" "%~dp0start_bot.bat"

echo.
echo ============================================================
echo   Tat ca dich vu da duoc khoi chay!
echo.
echo   - Web Studio:    http://127.0.0.1:5000
echo   - Telegram Bot:  Dang lang nghe lenh tren Telegram
echo ============================================================
echo.
echo Nhan phim bat ky de dong cua so nay (dich vu van chay tren 2 cua so rieng)...
pause >nul
