@echo off
chcp 65001 >nul
title Chinese SRT Extractor - Telegram Bot
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe bot.py
) else (
    python bot.py
)
pause
