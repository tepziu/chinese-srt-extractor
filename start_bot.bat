@echo off
chcp 65001 >nul
title Chinese SRT Extractor - Telegram Bot
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
set PYTHONIOENCODING=utf-8
python bot.py
pause
