@echo off
chcp 65001 >nul
title 🎬 Chinese SRT Extractor - All Services
call venv\Scripts\activate.bat 2>nul
echo ============================================================
echo   🎬 Khởi động tất cả dịch vụ
echo ============================================================
echo.
echo 🌐 Web App:     http://localhost:5000
echo 🤖 Telegram Bot: @taovideoauto_bot
echo.
echo ============================================================
echo.

set PYTHONIOENCODING=utf-8
start "Web App" cmd /k "chcp 65001 >nul && call venv\Scripts\activate.bat && python app.py"
timeout /t 3 >nul
start "Telegram Bot" cmd /k "chcp 65001 >nul && call venv\Scripts\activate.bat && python bot.py"

echo ✅ Đã khởi động cả Web App và Telegram Bot!
echo    Nhấn phím bất kỳ để đóng cửa sổ này...
pause >nul
