@echo off
chcp 65001 >nul
title Chinese SRT Extractor - Restart

echo.
echo ============================================================
echo   Dang khoi dong lai tat ca dich vu...
echo ============================================================
echo.

:: Stop everything first
call "%~dp0stop_all.bat"

echo.
echo   Dang khoi dong lai...
echo.
ping 127.0.0.1 -n 3 >nul

:: Start everything
call "%~dp0start_all.bat"
