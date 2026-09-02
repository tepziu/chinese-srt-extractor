#!/usr/bin/env python3
"""Central configuration and lightweight runtime state for the extractor."""

from __future__ import annotations

import ctypes
import glob
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_cuda_loaded = False
for pkg_name in ["nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"]:
    try:
        pkg = __import__(pkg_name, fromlist=[""])
        pkg_root = list(pkg.__path__)[0] if hasattr(pkg, "__path__") else ""
        for sub_dir in ["bin", "lib"]:
            lib_dir = os.path.join(pkg_root, sub_dir)
            if os.path.isdir(lib_dir):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(lib_dir)
                    except OSError:
                        pass
                os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
                lib_ext = "*.dll" if sys.platform == "win32" else "*.so*"
                for lib_file in sorted(glob.glob(os.path.join(lib_dir, lib_ext))):
                    try:
                        ctypes.CDLL(lib_file, mode=0 if sys.platform == "win32" else ctypes.RTLD_GLOBAL)
                        _cuda_loaded = True
                    except OSError:
                        pass
    except ImportError:
        pass
if _cuda_loaded:
    print("CUDA libraries preloaded successfully")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

LANGUAGES = {
    "vi": {"name": "Tiếng Việt", "flag": "🇻🇳"},
    "en": {"name": "English", "flag": "🇺🇸"},
    "id": {"name": "Bahasa Indonesia", "flag": "🇮🇩"},
}

MODEL_MAP = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "large-v1": "large-v1",
}
VALID_MODELS = list(MODEL_MAP)


def detect_device():
    """Detect the best CTranslate2 device without importing torch."""
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if "int8_float16" in cuda_types:
            return "cuda", "int8_float16"
        if "float16" in cuda_types:
            return "cuda", "float16"
        if cuda_types:
            return "cuda", "int8"
    except Exception as exc:
        print(f"GPU detection failed: {exc}")
    return "cpu", "int8"


DEVICE, COMPUTE_TYPE = detect_device()
print(f"Device: {DEVICE.upper()} | Compute: {COMPUTE_TYPE}")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
MAX_VIDEO_DURATION_SECONDS = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "7200"))
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "5000"))

jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.RLock()
JOB_MAX_AGE_SECONDS = int(os.getenv("JOB_MAX_AGE_SECONDS", "3600"))
JOB_MAX_COUNT = int(os.getenv("JOB_MAX_COUNT", "50"))
FILE_MAX_AGE_SECONDS = int(os.getenv("FILE_MAX_AGE_SECONDS", "7200"))
_CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "1800"))
_last_cleanup_time = 0.0
_cleanup_thread: threading.Thread | None = None
_cleanup_stop = threading.Event()


def create_job(job_id: str, **fields: Any) -> dict[str, Any]:
    """Create a consistent job record for Web and Telegram callers."""
    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Đang chờ xử lý...",
        "cancel": False,
        "srt_files": {},
        "translate_progress": {},
        "_created_at": time.time(),
        **fields,
    }
    with _jobs_lock:
        jobs[job_id] = job
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        return jobs.get(job_id)


def cleanup_old_jobs() -> None:
    """Remove expired records; recursive artifact cleanup runs separately."""
    global _last_cleanup_time
    now = time.time()
    if now - _last_cleanup_time < _CLEANUP_INTERVAL:
        return
    with _jobs_lock:
        _last_cleanup_time = now
        expired = [
            jid for jid, job in jobs.items()
            if job.get("_created_at") and now - job["_created_at"] > JOB_MAX_AGE_SECONDS
        ]
        if len(jobs) > JOB_MAX_COUNT:
            ordered = sorted(jobs.items(), key=lambda item: item[1].get("_created_at", 0))
            expired.extend(jid for jid, _ in ordered[: len(jobs) - JOB_MAX_COUNT])
        for jid in set(expired):
            job = jobs.get(jid, {})
            if job.get("status") not in {
                "queued", "downloading_video", "extracting", "loading_model",
                "transcribing", "translating", "processing", "uploading_video",
                "processing_video", "extracting_hardsub",
            }:
                jobs.pop(jid, None)
        if expired:
            print(f"Cleaned up {len(set(expired))} old job records")


def _is_active_path(path: Path) -> bool:
    with _jobs_lock:
        active_ids = {
            jid for jid, job in jobs.items()
            if job.get("status") not in {"done", "error", "cancelled"}
        }
    parts = set(path.parts)
    return any(jid in parts or path.name.startswith(f"{jid}_") for jid in active_ids)


def cleanup_old_files(force: bool = False) -> int:
    """Recursively remove expired artifacts without touching active jobs."""
    now = time.time()
    cleaned = 0
    for root in (UPLOAD_FOLDER, OUTPUT_FOLDER):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            try:
                if path.is_file():
                    if not force and now - path.stat().st_mtime <= FILE_MAX_AGE_SECONDS:
                        continue
                    if _is_active_path(path):
                        continue
                    path.unlink()
                    cleaned += 1
                elif path.is_dir() and path != root:
                    if not any(path.iterdir()) and (force or now - path.stat().st_mtime > FILE_MAX_AGE_SECONDS):
                        path.rmdir()
            except (OSError, PermissionError):
                continue
    if cleaned:
        print(f"Cleaned up {cleaned} old artifact files")
    return cleaned


def start_cleanup_worker() -> None:
    """Start one process-local background cleanup loop."""
    global _cleanup_thread
    if _cleanup_thread and _cleanup_thread.is_alive():
        return

    def _loop() -> None:
        while not _cleanup_stop.wait(_CLEANUP_INTERVAL):
            try:
                cleanup_old_jobs()
                cleanup_old_files()
            except Exception as exc:
                print(f"Cleanup worker error: {exc}")

    _cleanup_thread = threading.Thread(target=_loop, name="artifact-cleaner", daemon=True)
    _cleanup_thread.start()


AI_TRANSLATE_MODELS = {
    "gemini-3.7-flash-high": {
        "name": "Gemini 3.7 Flash High ⚡ (Khuyên dùng)",
        "description": "Nhanh, thông minh, dịch xưng hô và văn cảnh tiếng Trung chuẩn",
        "tier": "recommended",
    },
    "gpt-5.6-luna": {
        "name": "GPT 5.6 Luna 🧠 (Điện ảnh / Kịch tính)",
        "description": "Văn phong trau chuốt, giàu cảm xúc cho phim và kịch",
        "tier": "cinema",
    },
    "gpt-5.5": {
        "name": "GPT 5.5 🌟",
        "description": "Mô hình cân bằng, ổn định",
        "tier": "standard",
    },
    "claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6 🎭",
        "description": "Độ chính xác ngữ pháp và chuyển ngữ mượt mà",
        "tier": "high",
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash 💨",
        "description": "Siêu nhẹ, phản hồi nhanh",
        "tier": "fast",
    },
}
AI_DEFAULT_MODEL = os.getenv("AI_TRANSLATE_MODEL", "gemini-3.7-flash-high")

AI_TRANSLATE_CONFIG = {
    "base_url": os.getenv("AI_TRANSLATE_BASE_URL", "http://127.0.0.1:8317/v1").rstrip("/"),
    "api_key": os.getenv("AI_TRANSLATE_API_KEY", "").strip(),
    "model": AI_DEFAULT_MODEL,
}
AI_LANG_NAMES = {"vi": "Vietnamese", "en": "English", "id": "Indonesian"}

# Multi-speaker voice assignments per language and engine
SPEAKER_VOICE_MAPS = {
    "edge": {
        "vi": {
            "M1": {"voice": "vi-VN-NamMinhNeural", "name": "Nam chính (Nam Minh)", "gender": "male", "pitch": "+0Hz"},
            "F1": {"voice": "vi-VN-HoaiMyNeural", "name": "Nữ chính (Hoài My)", "gender": "female", "pitch": "+0Hz"},
            "M2": {"voice": "vi-VN-NamMinhNeural", "name": "Nam phụ (Trầm)", "gender": "male", "pitch": "-4Hz"},
            "F2": {"voice": "vi-VN-HoaiMyNeural", "name": "Nữ phụ (Trẻ)", "gender": "female", "pitch": "+4Hz"},
            "N": {"voice": "vi-VN-NamMinhNeural", "name": "Dẫn chuyện", "gender": "neutral", "pitch": "+0Hz"},
        },
        "en": {
            "M1": {"voice": "en-US-GuyNeural", "name": "Male Lead (Guy)", "gender": "male", "pitch": "+0Hz"},
            "F1": {"voice": "en-US-JennyNeural", "name": "Female Lead (Jenny)", "gender": "female", "pitch": "+0Hz"},
            "M2": {"voice": "en-US-ChristopherNeural", "name": "Male Secondary", "gender": "male", "pitch": "-4Hz"},
            "F2": {"voice": "en-US-AriaNeural", "name": "Female Secondary", "gender": "female", "pitch": "+4Hz"},
            "N": {"voice": "en-US-GuyNeural", "name": "Narrator", "gender": "neutral", "pitch": "+0Hz"},
        },
        "id": {
            "M1": {"voice": "id-ID-ArdiNeural", "name": "Pria Utama (Ardi)", "gender": "male", "pitch": "+0Hz"},
            "F1": {"voice": "id-ID-GadisNeural", "name": "Wanita Utama (Gadis)", "gender": "female", "pitch": "+0Hz"},
            "M2": {"voice": "id-ID-ArdiNeural", "name": "Pria Pendukung", "gender": "male", "pitch": "-4Hz"},
            "F2": {"voice": "id-ID-GadisNeural", "name": "Wanita Pendukung", "gender": "female", "pitch": "+4Hz"},
            "N": {"voice": "id-ID-ArdiNeural", "name": "Narator", "gender": "neutral", "pitch": "+0Hz"},
        },
        "zh": {
            "M1": {"voice": "zh-CN-YunxiNeural", "name": "男主 (云希)", "gender": "male", "pitch": "+0Hz"},
            "F1": {"voice": "zh-CN-XiaoxiaoNeural", "name": "女主 (晓晓)", "gender": "female", "pitch": "+0Hz"},
            "M2": {"voice": "zh-CN-YunjianNeural", "name": "男配 (云健)", "gender": "male", "pitch": "-4Hz"},
            "F2": {"voice": "zh-CN-XiaoyiNeural", "name": "女配 (晓伊)", "gender": "female", "pitch": "+4Hz"},
            "N": {"voice": "zh-CN-YunxiNeural", "name": "旁白", "gender": "neutral", "pitch": "+0Hz"},
        }
    },
    "gemini": {
        "vi": {
            "M1": {"voice": "Charon", "name": "Nam chính (Charon)", "gender": "male", "emotion": "warm"},
            "F1": {"voice": "Kore", "name": "Nữ chính (Kore)", "gender": "female", "emotion": "warm"},
            "M2": {"voice": "Fenrir", "name": "Nam phụ (Fenrir)", "gender": "male", "emotion": "dramatic"},
            "F2": {"voice": "Aoede", "name": "Nữ phụ (Aoede)", "gender": "female", "emotion": "intimate"},
            "N": {"voice": "Zephyr", "name": "Dẫn chuyện (Zephyr)", "gender": "neutral", "emotion": "neutral"},
        },
        "en": {
            "M1": {"voice": "Charon", "name": "Male Lead (Charon)", "gender": "male", "emotion": "warm"},
            "F1": {"voice": "Kore", "name": "Female Lead (Kore)", "gender": "female", "emotion": "warm"},
            "M2": {"voice": "Fenrir", "name": "Male Secondary", "gender": "male", "emotion": "dramatic"},
            "F2": {"voice": "Aoede", "name": "Female Secondary", "gender": "female", "emotion": "intimate"},
            "N": {"voice": "Zephyr", "name": "Narrator", "gender": "neutral", "emotion": "neutral"},
        }
    }
}

TTS_VOICES = {
    "vi": "vi-VN-NamMinhNeural",
    "en": "en-US-GuyNeural",
    "id": "id-ID-ArdiNeural",
    "zh": "zh-CN-YunxiNeural",
}

PRESETS_FILE = BASE_DIR / "presets.json"
MAX_BLUR_REGIONS = 3


def load_presets() -> dict:
    try:
        if PRESETS_FILE.exists():
            data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else _default_presets()
    except Exception as exc:
        print(f"Failed to load presets: {exc}")
    return _default_presets()


def save_presets(presets: dict) -> None:
    try:
        tmp = PRESETS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(PRESETS_FILE)
    except Exception as exc:
        print(f"Failed to save presets: {exc}")


def _default_presets() -> dict:
    return {
        "default": {
            "name": "Mặc định (Sub dưới)",
            "sub_region": {"x_ratio": 0, "y_ratio": 0.78, "w_ratio": 1.0, "h_ratio": 0.18},
            "extra_regions": [],
        }
    }


def validate_region(region: dict) -> bool:
    try:
        x = float(region.get("x_ratio", 0))
        y = float(region.get("y_ratio", 0))
        w = float(region.get("w_ratio", 1))
        h = float(region.get("h_ratio", 0))
        return 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.0 and y + h <= 1.0
    except (ValueError, TypeError, AttributeError):
        return False


def safe_stem(value: str, fallback: str = "video", max_length: int = 80) -> str:
    """Return a cross-platform safe filename stem."""
    value = Path(str(value or fallback)).stem
    value = re.sub(r"[^A-Za-z0-9À-ỹ一-龥_ .-]+", "_", value).strip(" ._")
    return (value or fallback)[:max_length]


GEMINI_MODELS = {
    "gemini-2.5-flash": {"name": "Gemini 2.5 Flash ⚡ (khuyên dùng)", "description": "Nhanh, giá rẻ, reasoning tốt", "tier": "stable"},
    "gemini-2.5-flash-lite": {"name": "Gemini 2.5 Flash-Lite 💨", "description": "Siêu nhanh, rẻ nhất", "tier": "stable"},
    "gemini-2.5-pro": {"name": "Gemini 2.5 Pro 🧠", "description": "Chính xác nhất, deep reasoning", "tier": "stable"},
    "gemini-3-flash-preview": {"name": "Gemini 3 Flash (Preview)", "description": "Thế hệ mới", "tier": "preview"},
    "gemini-3.1-pro-preview": {"name": "Gemini 3.1 Pro (Preview)", "description": "Mạnh nhất", "tier": "preview"},
}
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_api_key() -> str:
    from services.google_tts import load_google_media_credentials
    keys, _source = load_google_media_credentials()
    return keys[0] if keys else ""


def set_gemini_api_key(api_key: str) -> None:
    from services.google_tts import save_shared_google_key
    save_shared_google_key(api_key)
