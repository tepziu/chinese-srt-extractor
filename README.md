# Chinese SRT Extractor & Translator

Công cụ local trên Windows/Linux để:

- Trích xuất tiếng Trung bằng Faster-Whisper (CUDA/CPU).
- Trích hardsub từ hình ảnh bằng Gemini.
- Dịch sang Tiếng Việt, English và Bahasa Indonesia với 2 phong cách chuyên biệt: Phim ảnh (đa nhân vật [M1, F1, M2, F2, N]) hoặc Dạy lái xe & Mẹo xe (chuẩn thuật ngữ ô tô/lái xe, 1 giọng hướng dẫn [M1]).
- Tạo TTS bằng Edge-TTS, Gemini TTS hoặc OmniVoice tùy cấu hình.
- Xử lý video đa chế độ: Dynamic Blur (chỉ làm mờ khi có thoại theo timeline SRT), AI Clean Plate (tẩy sạch 100% chữ tiếng Trung bằng Inpaint) hoặc Inpaint & Re-burn phụ đề mới.
- Nhận video/URL qua Web UI hoặc Telegram Bot.

## Cài đặt Windows

Yêu cầu: Python 3.10+, FFmpeg, Chrome (chỉ cần cho một số nền tảng Trung Quốc), NVIDIA driver nếu dùng CUDA.

```bat
setup_windows.bat
```

Script cài core dependencies và các dependency Hardsub/OCR. Với GPU, script cài PyTorch CUDA phù hợp trước khi khởi động.

## Cấu hình

Copy `.env.example` thành `.env`, sau đó đặt credential:

```env
TELEGRAM_BOT_TOKEN=
AI_TRANSLATE_BASE_URL=http://127.0.0.1:8317/v1
AI_TRANSLATE_API_KEY=
AI_TRANSLATE_MODEL=gpt-5.5
GEMINI_API_KEY=
```

Không commit `.env`, `config/google_media.json`, video upload hoặc output. Credential từng xuất hiện trong source cũ cần được thu hồi và tạo lại.

Mặc định Web chỉ bind vào `127.0.0.1` để không mở API cho toàn bộ LAN. Chỉ đặt `WEB_HOST=0.0.0.0` khi đã có lớp xác thực/reverse proxy bảo vệ phía trước.

Các giới hạn runtime:

- `MAX_UPLOAD_BYTES`: mặc định 2 GB.
- `MAX_DOWNLOAD_BYTES`: mặc định 2 GB.
- `MAX_VIDEO_DURATION_SECONDS`: mặc định 2 giờ.
- `FILE_MAX_AGE_SECONDS`: mặc định 2 giờ cho artifact tạm.

## Chạy

```bat
start_web.bat
start_bot.bat
start_all.bat
```

Web UI: `http://127.0.0.1:5000`

## Luồng xử lý

### Whisper

`upload/URL -> FFmpeg 16kHz mono -> Faster-Whisper -> split câu -> validate SRT -> dịch -> artifacts`

### Hardsub

`upload/URL -> Gemini video OCR -> validate SRT -> dịch -> artifacts`

Hardsub URL dùng downloader chung với Whisper nhưng không chạy nhầm Whisper.

### Artifact layout

Mỗi job có thư mục riêng:

```text
uploads/<job_id>/source.mp4
uploads/<job_id>/audio.wav
outputs/<job_id>/<subtitle-or-audio-files>
```

Background cleanup dọn recursive artifact cũ và không xóa job đang chạy.

## Telegram

```text
/lang vi en
/model large-v3-turbo
/mode movie|driving
/vmode blur|clean|inpaint_burn
/tts on
/voice edge|gemini|omnivoice
/burn on
/hardsub on
/status
```

Bot token phải được đặt qua `TELEGRAM_BOT_TOKEN`; code không còn fallback token hardcode.

## OmniVoice (tùy chọn)

OmniVoice sử dụng environment Python riêng. Cấu hình trong `.env`:

```env
OMNIVOICE_PYTHON=D:\Naldo\omnivoice-env\Scripts\python.exe
OMNIVOICE_REFERENCE_AUDIO=D:\path\to\reference.wav
OMNIVOICE_REFERENCE_TEXT_FILE=D:\path\to\reference.txt
```

Nếu thiếu runtime hoặc reference audio, UI/bot phải báo cấu hình thiếu thay vì báo thành công giả.

## Kiểm thử

```powershell
python -m compileall -q app.py bot.py config.py routes services tests
python -m unittest discover -s tests -v
```

Test pipeline cần chạy trong đúng virtual environment có dependencies tương ứng. Hardsub, OCR và OmniVoice là các tính năng optional nếu môi trường chưa cài dependency/model.
