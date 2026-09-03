"""Whisper model loading, audio extraction and Chinese subtitle pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from config import DEVICE, COMPUTE_TYPE, LANGUAGES, MODEL_MAP, OUTPUT_FOLDER, UPLOAD_FOLDER, jobs, safe_stem
from services.srt_utils import format_timestamp, generate_srt, validate_srt
from services.translation import translate_srt, translate_srt_ai

_models = {}
_model_lock = threading.Lock()
_transcription_lock = threading.Lock()


def get_model(model_size: str = "large-v3-turbo"):
    """Lazy-load one Whisper model and reuse it between jobs."""
    if model_size not in _models:
        with _model_lock:
            if model_size not in _models:
                from faster_whisper import WhisperModel

                actual_model = MODEL_MAP.get(model_size, model_size)
                compute_type = "int8_float16" if DEVICE == "cuda" else "int8"
                print(f"Loading model '{model_size}' on {DEVICE.upper()} ({compute_type})...")
                try:
                    _models[model_size] = WhisperModel(
                        actual_model,
                        device=DEVICE,
                        compute_type=compute_type,
                        cpu_threads=os.cpu_count() or 4,
                    )
                except Exception as exc:
                    if DEVICE != "cuda":
                        raise
                    print(f"GPU model load failed ({exc}); falling back to CPU")
                    _models[model_size] = WhisperModel(
                        actual_model,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=os.cpu_count() or 4,
                    )
                print(f"Model '{model_size}' is ready")
    return _models[model_size]


def extract_audio(video_path: str, job_id: str) -> str:
    """Extract 16 kHz mono PCM audio for Whisper."""
    job = jobs[job_id]
    job["status"] = "extracting"
    job["message"] = "Đang tách audio từ video..."
    audio_path = str(UPLOAD_FOLDER / job_id / "audio.wav")
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-print_format", "json", video_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        video_duration = float(json.loads(probe.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        video_duration = 0.0

    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", "-threads", str(min(os.cpu_count() or 4, 8)), audio_path,
    ]
    started = time.time()
    process = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    elapsed = time.time() - started
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"FFmpeg lỗi: {detail}")

    video_size = Path(video_path).stat().st_size
    audio_size = Path(audio_path).stat().st_size
    reduction = (1 - audio_size / video_size) * 100 if video_size else 0
    job.update({
        "extract_time": round(elapsed, 1),
        "video_size": video_size,
        "audio_size": audio_size,
        "video_duration_probe": video_duration,
        "message": f"Tách audio xong ({elapsed:.1f}s) - giảm {reduction:.0f}%",
    })
    return audio_path


_SENTENCE_END = set("。！？；!?;")
_CLAUSE_BREAK = set("，,、：:")


def split_segments_by_sentence(segments, max_chars: int = 30):
    """Split long Whisper segments using word timestamps and Chinese punctuation."""
    result = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        words = getattr(seg, "words", None)
        if not words or len(text) <= max_chars:
            result.append({"start": seg.start, "end": seg.end, "text": text})
            continue

        sentence_parts = []
        current = []
        for word in words:
            current.append(word)
            word_text = word.word.strip()
            if word_text and word_text[-1] in _SENTENCE_END:
                joined = "".join(item.word for item in current).strip()
                if joined:
                    sentence_parts.append({"start": current[0].start, "end": current[-1].end, "text": joined})
                current = []
        if current:
            joined = "".join(item.word for item in current).strip()
            if joined:
                sentence_parts.append({"start": current[0].start, "end": current[-1].end, "text": joined})

        final_parts = []
        for part in sentence_parts:
            if len(part["text"]) <= max_chars:
                final_parts.append(part)
                continue
            sub_words = [
                word for word in words
                if word.start >= part["start"] - 0.01 and word.end <= part["end"] + 0.01
            ]
            chunk = []
            chunk_len = 0
            for word in sub_words:
                word_text = word.word.strip()
                chunk.append(word)
                chunk_len += len(word_text)
                if word_text and word_text[-1] in _CLAUSE_BREAK and chunk_len >= 8:
                    joined = "".join(item.word for item in chunk).strip()
                    if joined:
                        final_parts.append({"start": chunk[0].start, "end": chunk[-1].end, "text": joined})
                    chunk = []
                    chunk_len = 0
            if chunk:
                joined = "".join(item.word for item in chunk).strip()
                if joined:
                    final_parts.append({"start": chunk[0].start, "end": chunk[-1].end, "text": joined})
        result.extend(final_parts or [{"start": seg.start, "end": seg.end, "text": text}])
    return result


def process_video(job_id: str, video_path: str, model_size: str, translate_langs: list[str], translate_method: str = "ai", translation_mode: str = "movie") -> None:
    """Run the complete Whisper -> SRT -> translation pipeline in a worker."""
    audio_path = None
    total_started = time.time()
    job = jobs[job_id]
    job["device"] = DEVICE.upper()
    job["compute_type"] = COMPUTE_TYPE
    translation_mode = translation_mode or job.get("translation_mode", "movie")
    job["translation_mode"] = translation_mode
    try:
        audio_path = extract_audio(video_path, job_id)
        job["status"] = "loading_model"
        job["message"] = f"Đang tải AI model trên {DEVICE.upper()}..."
        model = get_model(model_size)

        job["status"] = "transcribing"
        job["message"] = f"Đang nhận dạng giọng nói [{DEVICE.upper()}]..."
        transcribe_started = time.time()
        segments = []
        with _transcription_lock:
            segments_gen, info = model.transcribe(
                audio_path,
                language="zh",
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 250,
                    "speech_pad_ms": 100,
                    "max_speech_duration_s": 10,
                    "threshold": 0.35,
                },
                word_timestamps=True,
                condition_on_previous_text=True,
            )
            duration = info.duration if info.duration else 1
            for segment in segments_gen:
                if job.get("cancel"):
                    job["status"] = "cancelled"
                    job["message"] = "Đã hủy nhận dạng"
                    return
                segments.append(segment)
                progress = min(int(segment.end / duration * 100), 99)
                elapsed = time.time() - transcribe_started
                speed = segment.end / elapsed if elapsed > 0 else 0
                job["progress"] = progress
                job["message"] = f"[{DEVICE.upper()}] Nhận dạng: {progress}% ({format_timestamp(segment.end)}) - {speed:.1f}x"

        transcribe_time = time.time() - transcribe_started
        split_segments = split_segments_by_sentence(segments, max_chars=30)
        if not split_segments:
            job.update({
                "status": "done",
                "progress": 100,
                "segment_count": 0,
                "raw_segment_count": len(segments),
                "duration": info.duration,
                "transcribe_time": round(transcribe_time, 1),
                "language_probability": info.language_probability,
                "message": "Không phát hiện giọng nói trong video",
                "srt_files": {},
            })
            return
        srt_content = generate_srt(split_segments)
        valid, errors = validate_srt(srt_content)
        if not valid:
            raise RuntimeError(f"SRT Whisper không hợp lệ: {'; '.join(errors[:3])}")

        output_dir = OUTPUT_FOLDER / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        video_stem = safe_stem(job.get("original_name", "video"))
        zh_filename = f"{video_stem}_zh.srt"
        zh_path = output_dir / zh_filename
        zh_path.write_text(srt_content, encoding="utf-8")

        job.update({
            "status": "translating" if translate_langs else "done",
            "progress": 100,
            "segment_count": len(split_segments),
            "raw_segment_count": len(segments),
            "duration": info.duration,
            "transcribe_time": round(transcribe_time, 1),
            "language_probability": info.language_probability,
            "srt_files": {
                "zh": {
                    "path": str(zh_path), "filename": zh_filename, "preview": srt_content[:1500],
                    "lang_name": "中文 (原文)", "flag": "🇨🇳",
                }
            },
        })

        for lang_code in translate_langs:
            if job.get("cancel"):
                job["status"] = "cancelled"
                job["message"] = "Đã hủy dịch thuật"
                return
            lang_info = LANGUAGES[lang_code]
            job["message"] = f"Đang dịch sang {lang_info['name']}..."
            try:
                translated = translate_srt_ai(srt_content, lang_code, job_id, translation_mode=translation_mode) if translate_method == "ai" else translate_srt(srt_content, lang_code, job_id)
                translated_valid, translated_errors = validate_srt(translated)
                if not translated_valid:
                    raise RuntimeError(f"Bản dịch SRT không hợp lệ: {'; '.join(translated_errors[:2])}")
                filename = f"{video_stem}_{lang_code}.srt"
                path = output_dir / filename
                path.write_text(translated, encoding="utf-8")
                job["srt_files"][lang_code] = {
                    "path": str(path), "filename": filename, "preview": translated[:1500],
                    "lang_name": lang_info["name"], "flag": lang_info["flag"],
                }
                job["translate_progress"][lang_code] = 100
            except Exception as exc:
                job["srt_files"][lang_code] = {
                    "error": str(exc), "lang_name": lang_info["name"], "flag": lang_info["flag"],
                }

        job["total_time"] = round(time.time() - total_started, 1)
        job["status"] = "done"
        job["message"] = f"Hoàn thành trong {job['total_time']:.0f}s!"
        if Path(video_path).exists():
            job["video_file"] = {
                "path": str(video_path), "filename": Path(video_path).name, "size": Path(video_path).stat().st_size,
            }
    except Exception as exc:
        job["status"] = "error"
        job["message"] = f"Lỗi: {exc}"
        import traceback
        job["traceback"] = traceback.format_exc()
    finally:
        if audio_path:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                pass

