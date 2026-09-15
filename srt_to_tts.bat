@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

title Tạo Audio TTS từ file .SRT

cd /d "%~dp0"

echo ====================================================================
echo   🎙️ CÔNG CỤ CHỈ TẠO FILE AUDIO TTS TỪ FILE PHỤ ĐỀ .SRT
echo ====================================================================

if "%~1"=="" (
    echo.
    echo   [HƯỚNG DẪN]:
    echo     1. Bạn có thể KÉO THẢ bất kỳ file .srt nào vào file .bat này!
    echo     2. Hoặc nhập trực tiếp đường dẫn file .srt dưới đây.
    echo.
    set /p "SRT_FILE=👉 Nhập đường dẫn file .srt: "
) else (
    set "SRT_FILE=%~1"
)

set "SRT_FILE=%SRT_FILE:"=%"

if "!SRT_FILE!"=="" (
    echo [LỖI] Chưa nhập đường dẫn file.
    pause
    exit /b 1
)

if not exist "!SRT_FILE!" (
    echo.
    echo [LỖI] Không tìm thấy file: "!SRT_FILE!"
    echo Vui lòng kiểm tra lại đường dẫn.
    echo.
    pause
    exit /b 1
)

echo.
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" srt_to_tts.py "!SRT_FILE!"
) else (
    python srt_to_tts.py "!SRT_FILE!"
)

echo.
echo ====================================================================
pause
