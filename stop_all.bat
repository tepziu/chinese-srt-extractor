@echo off
chcp 65001 >nul
title Chinese SRT Extractor - Stop
cd /d "%~dp0"
echo Dang tat tat ca dich vu...

taskkill /IM ffmpeg.exe /F >nul 2>&1

for /f "tokens=2 delims==" %%a in ('wmic process where "commandline like '%%app.py%%' and name like '%%python%%'" get processid /value 2^>nul') do (
    if not "%%a"=="" (
        taskkill /PID %%a /T /F >nul 2>&1
        echo Da tat Web App [PID %%a]
    )
)

for /f "tokens=2 delims==" %%a in ('wmic process where "commandline like '%%bot.py%%' and name like '%%python%%'" get processid /value 2^>nul') do (
    if not "%%a"=="" (
        taskkill /PID %%a /T /F >nul 2>&1
        echo Da tat Telegram Bot [PID %%a]
    )
)

taskkill /FI "WINDOWTITLE eq Chinese SRT Extractor - Web*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Chinese SRT Extractor - Telegram*" /F >nul 2>&1

echo Hoan tat tat ca dich vu!
