"""Gemini-based extraction of hardcoded subtitles into validated SRT."""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path

from config import GEMINI_DEFAULT_MODEL, GEMINI_MODELS, LANGUAGES, OUTPUT_FOLDER, get_gemini_api_key
from services.srt_utils import validate_srt

HARDSUB_PROMPT = """You are an expert subtitle extractor. Analyze this video frame by frame and extract ALL hardcoded/burned-in subtitle text that appears on screen.

Rules:
1. Extract only subtitle text overlaid on the video; ignore watermarks, logos and UI.
2. Determine precise start and end timestamps.
3. Merge identical text across consecutive frames.
4. Keep the original language exactly as shown.
5. Return only standard SRT, with sequential numbering.
6. If no subtitle exists, return NO_SUBTITLES_FOUND.

Output format:
1
00:00:01,000 --> 00:00:03,500
Subtitle text

Begin extraction now:"""


def parse_srt_from_text(text: str) -> str:
    """Remove markdown wrappers and normalize entry numbering."""
    if not text or "NO_SUBTITLES_FOUND" in text:
        return ""
    text = re.sub(r"^\s*```(?:srt)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    lines = [line.rstrip() for line in text.strip().splitlines()]
    result = []
    counter = 1
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.isdigit() and index + 1 < len(lines) and "-->" in lines[index + 1]:
            result.append(str(counter))
            counter += 1
            index += 1
            continue
        result.append(line)
        index += 1
    return "\n".join(result).strip()


def count_srt_segments(srt_text: str) -> int:
    return len(re.findall(r"(?m)^\d+\s*$\n\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->", srt_text or ""))


def _friendly_gemini_error(error: Exception) -> str:
    msg = str(error).lower()
    if "api_key" in msg or "api key" in msg or ("invalid" in msg and "key" in msg):
        return "API key không hợp lệ. Vui lòng kiểm tra lại tại aistudio.google.com"
    if "quota" in msg or "rate limit" in msg or "resource_exhausted" in msg:
        return "Đã hết quota Gemini API. Vui lòng chờ hoặc nâng cấp plan."
    if "permission" in msg or "forbidden" in msg:
        return "API key không có quyền truy cập model này."
    if "not found" in msg and "model" in msg:
        return "Model Gemini không tồn tại hoặc chưa khả dụng."
    if "too large" in msg or "payload" in msg:
        return "File video quá lớn cho Gemini API."
    if "timeout" in msg or "deadline" in msg:
        return "Gemini API bị timeout. Thử video ngắn hơn hoặc model nhẹ hơn."
    if "connection" in msg or "network" in msg:
        return "Lỗi kết nối đến Gemini API. Kiểm tra mạng internet."
    return f"Lỗi Gemini: {error}"


def hardsub_worker(job: dict) -> None:
    """Upload, extract, validate, translate and publish Hardsub artifacts."""
    started = time.time()
    video_file = None
    client = None
    try:
        from google import genai

        job_id = job["job_id"]
        video_path = str(job["video_path"])
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError("Video không tồn tại")
        job["video_file"] = {"path": video_path, "filename": path.name, "size": path.stat().st_size}

        api_key = job.get("gemini_api_key") or get_gemini_api_key()
        if not api_key:
            raise RuntimeError("Chưa có Gemini API Key. Vui lòng cấu hình key.")
        model_name = job.get("gemini_model", GEMINI_DEFAULT_MODEL)
        if model_name not in GEMINI_MODELS:
            model_name = GEMINI_DEFAULT_MODEL
        translate_langs = [lang for lang in job.get("translate_langs", []) if lang in LANGUAGES]
        translate_method = job.get("translate_method", "google")

        if job.get("cancel"):
            job["status"] = "cancelled"
            job["message"] = "Đã hủy."
            return

        job.update({"status": "uploading_video", "progress": 5, "message": "Đang tải video lên Gemini..."})
        client = genai.Client(api_key=api_key)
        video_file = client.files.upload(file=video_path)
        job.update({"progress": 15, "message": "Đã tải video lên Gemini. Đang xử lý...", "status": "processing_video"})

        max_wait = 600
        waited = 0
        while video_file.state.name == "PROCESSING" and waited < max_wait:
            if job.get("cancel"):
                job["status"] = "cancelled"
                job["message"] = "Đã hủy."
                return
            time.sleep(5)
            waited += 5
            video_file = client.files.get(name=video_file.name)
            job["progress"] = min(40, 15 + int(waited / max_wait * 25))
            job["message"] = f"Gemini đang xử lý video... ({waited}s)"
        if video_file.state.name != "ACTIVE":
            raise RuntimeError(f"Xử lý video thất bại: {video_file.state.name}")

        job.update({"status": "extracting_hardsub", "progress": 45, "message": f"Đang trích xuất hardsub bằng {model_name}..."})
        response = client.models.generate_content(model=model_name, contents=[video_file, HARDSUB_PROMPT])
        raw_text = response.text or ""
        srt_content = parse_srt_from_text(raw_text)
        valid, errors = validate_srt(srt_content) if srt_content else (False, ["Không có SRT"])
        if not valid:
            raise RuntimeError("SRT Hardsub không hợp lệ: " + "; ".join(errors[:3]))

        output_dir = OUTPUT_FOLDER / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        zh_path = output_dir / "hardsub_zh.srt"
        zh_path.write_text(srt_content, encoding="utf-8")
        srt_files = {
            "zh": {
                "filename": zh_path.name, "path": str(zh_path), "preview": srt_content[:500],
                "lang_name": "中文 (原文)", "flag": "🇨🇳",
            }
        }
        job.update({"progress": 75, "status": "translating" if translate_langs else "done", "segment_count": count_srt_segments(srt_content), "srt_files": srt_files})

        if translate_langs:
            from services.translation import translate_srt, translate_srt_ai
            translation_mode = job.get("translation_mode", "movie")
            job["translate_progress"] = {lang: 0 for lang in translate_langs}
            for lang in translate_langs:
                if job.get("cancel"):
                    job["status"] = "cancelled"
                    job["message"] = "Đã hủy dịch thuật."
                    return
                try:
                    translated = translate_srt_ai(srt_content, lang, job_id, translation_mode=translation_mode) if translate_method == "ai" else translate_srt(srt_content, lang, job_id)
                    translated_valid, translated_errors = validate_srt(translated)
                    if not translated_valid:
                        raise RuntimeError("Bản dịch SRT không hợp lệ: " + "; ".join(translated_errors[:2]))
                    lang_path = output_dir / f"hardsub_{lang}.srt"
                    lang_path.write_text(translated, encoding="utf-8")
                    srt_files[lang] = {
                        "filename": lang_path.name, "path": str(lang_path), "preview": translated[:500],
                        "lang_name": LANGUAGES[lang]["name"], "flag": LANGUAGES[lang]["flag"],
                    }
                    job["translate_progress"][lang] = 100
                except Exception as exc:
                    srt_files[lang] = {"error": str(exc), "lang_name": LANGUAGES[lang]["name"], "flag": LANGUAGES[lang]["flag"]}

        job["srt_files"] = srt_files
        job["status"] = "done"
        job["progress"] = 100
        job["total_time"] = round(time.time() - started, 1)
        job["message"] = "Hoàn tất!"
    except Exception as exc:
        traceback.print_exc()
        job["status"] = "error"
        job["message"] = _friendly_gemini_error(exc)
    finally:
        if client is not None and video_file is not None:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass
