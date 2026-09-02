"""
api.py — Flask Blueprint with all API routes
"""

import os
import json
import uuid
import time
import subprocess
import threading
from pathlib import Path

from flask import Blueprint, request, jsonify, send_file, make_response

from config import (
    LANGUAGES, VALID_MODELS, DEVICE, COMPUTE_TYPE,
    UPLOAD_FOLDER, OUTPUT_FOLDER,
    MAX_UPLOAD_BYTES, MAX_VIDEO_DURATION_SECONDS,
    jobs, cleanup_old_jobs, create_job, get_job, safe_stem,
    load_presets, save_presets, validate_region, MAX_BLUR_REGIONS,
    GEMINI_MODELS, GEMINI_DEFAULT_MODEL,
    AI_TRANSLATE_MODELS, AI_DEFAULT_MODEL, SPEAKER_VOICE_MAPS,
    get_gemini_api_key, set_gemini_api_key,
)
from services.whisper_engine import process_video
from services.tts import tts_worker
from services.google_tts import get_google_tts_health
from services.burn_sub import burnsub_worker
from services.downloader import download_from_url, process_url_video, validate_download_url
from services.hardsub_gemini import hardsub_worker

api_bp = Blueprint("api", __name__)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _parse_languages(value) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except (TypeError, ValueError):
            value = []
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(lang) for lang in value if str(lang) in LANGUAGES))


def _validate_uploaded_video(video):
    filename = str(video.filename or "").strip()
    if not filename:
        return None, "Chưa chọn file"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return None, f"Định dạng {ext or 'không xác định'} không được hỗ trợ"
    return ext, ""


def _video_info(path: str) -> dict:
    file_path = Path(path)
    return {"path": str(file_path), "filename": file_path.name, "size": file_path.stat().st_size}


def _validate_media_path(path: Path) -> tuple[bool, str]:
    try:
        size = path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            return False, f"File quá lớn ({MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB tối đa)"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-print_format", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        if duration > MAX_VIDEO_DURATION_SECONDS:
            return False, f"Video quá dài ({MAX_VIDEO_DURATION_SECONDS // 3600} giờ tối đa)"
        return True, ""
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, "File không phải video hợp lệ hoặc FFprobe không đọc được"
# ── Index ──────────────────────────────────────────────────────────────────

@api_bp.route("/")
def index():
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    html = template_path.read_text(encoding="utf-8")
    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    return resp


# ── Device Info ────────────────────────────────────────────────────────────

@api_bp.route("/api/device")
def device_info():
    info = {
        "device": DEVICE.upper(),
        "compute_type": COMPUTE_TYPE,
        "cpu_threads": os.cpu_count() or 4,
    }
    if DEVICE == "cuda":
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            info["gpu_name"] = result.stdout.strip()
        except:
            info["gpu_name"] = "NVIDIA GPU"
    return jsonify(info)


# ── Upload Video ───────────────────────────────────────────────────────────

@api_bp.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Không tìm thấy file video"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    ext, validation_error = _validate_uploaded_video(video)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"source{ext}"
    video.save(str(video_path))
    if not video_path.exists() or video_path.stat().st_size == 0:
        return jsonify({"error": "File upload rỗng hoặc không thể ghi"}), 400

    valid_media, media_error = _validate_media_path(video_path)
    if not valid_media:
        video_path.unlink(missing_ok=True)
        return jsonify({"error": media_error}), 400
    model_size = request.form.get("model_size", "large-v3-turbo")
    if model_size not in VALID_MODELS:
        model_size = "large-v3-turbo"

    translate_langs = _parse_languages(request.form.get("translate_langs", "[]"))
    ai_model = request.form.get("ai_model", AI_DEFAULT_MODEL)
    if ai_model not in AI_TRANSLATE_MODELS and not ai_model.strip():
        ai_model = AI_DEFAULT_MODEL
    translate_method = request.form.get("translate_method", "ai")
    if translate_method not in ("ai", "google"):
        translate_method = "ai"

    create_job(
        job_id,
        original_name=str(video.filename),
        video_path=str(video_path),
        translate_langs=translate_langs,
        translate_method=translate_method,
        ai_model=ai_model,
    )

    cleanup_old_jobs()

    thread = threading.Thread(
        target=process_video,
        args=(job_id, str(video_path), model_size, translate_langs, translate_method),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


# ── URL Download ───────────────────────────────────────────────────────────

@api_bp.route("/api/url", methods=["POST"])
def url_download():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Thiếu URL"}), 400

    url = str(data["url"]).strip()
    valid_url, url_error = validate_download_url(url)
    if not valid_url:
        return jsonify({"error": url_error}), 400

    model_size = data.get("model_size", "large-v3-turbo")
    if model_size not in VALID_MODELS:
        model_size = "large-v3-turbo"

    translate_langs = _parse_languages(data.get("translate_langs", []))
    job_id = uuid.uuid4().hex[:12]
    create_job(
        job_id,
        status="downloading_video",
        message="Đang chuẩn bị tải video...",
        original_name=url[:60],
        translate_langs=translate_langs,
    )

    cleanup_old_jobs()

    ai_model = data.get("ai_model", AI_DEFAULT_MODEL)
    if ai_model not in AI_TRANSLATE_MODELS and not str(ai_model).strip():
        ai_model = AI_DEFAULT_MODEL
    translate_method = data.get("translate_method", "ai")
    if translate_method not in ("ai", "google"):
        translate_method = "ai"
    jobs[job_id]["translate_method"] = translate_method
    jobs[job_id]["ai_model"] = ai_model

    thread = threading.Thread(
        target=process_url_video,
        args=(job_id, url, model_size, translate_langs, translate_method),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "downloading_video"})


# ── Job Status & Control ──────────────────────────────────────────────────

@api_bp.route("/api/status/<job_id>")
def get_status(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job không tồn tại"}), 404
    if "speakers" not in job:
        speakers_file = OUTPUT_FOLDER / job_id / "speakers.json"
        if speakers_file.exists():
            try:
                data = json.loads(speakers_file.read_text(encoding="utf-8"))
                job["speakers"] = data.get("speakers", {"M1": 1})
                job["segment_speakers"] = data.get("segment_speakers", [])
            except Exception:
                pass
    # Filter out internal keys and sensitive data
    _HIDDEN_KEYS = {"_ffmpeg_process", "_download_process", "_tts_process", "_created_at", "gemini_api_key", "video_path"}
    safe_data = {k: v for k, v in job.items() if k not in _HIDDEN_KEYS and not k.startswith("_")}
    return jsonify(safe_data)


@api_bp.route("/api/stop/<job_id>", methods=["POST"])
def stop_job(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job không tồn tại"}), 404
    job["cancel"] = True

    for process_key in ("_ffmpeg_process", "_download_process"):
        process = job.get(process_key)
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    return jsonify({"status": "ok"})
@api_bp.route("/api/shutdown", methods=["POST"])
def shutdown():
    # Disabled by default: an unauthenticated shutdown endpoint is unsafe on LAN.
    allow = os.getenv("ALLOW_SHUTDOWN", "0").lower() in {"1", "true", "yes"}
    local = request.remote_addr in {"127.0.0.1", "::1"}
    if not allow or not local:
        return jsonify({"error": "Shutdown endpoint đang bị vô hiệu hóa"}), 403
    func = request.environ.get("werkzeug.server.shutdown")
    if func is not None:
        func()
    return jsonify({"status": "ok", "message": "Server is shutting down..."})


# ── File Downloads ─────────────────────────────────────────────────────────

@api_bp.route("/api/download/<job_id>/<lang>")
def download_srt(job_id, lang):
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    job = jobs[job_id]
    if job["status"] != "done":
        return jsonify({"error": "File chưa sẵn sàng"}), 400

    srt_files = job.get("srt_files", {})
    if lang not in srt_files:
        return jsonify({"error": f"Ngôn ngữ '{lang}' không tồn tại"}), 404

    file_info = srt_files[lang]
    if "error" in file_info:
        return jsonify({"error": file_info["error"]}), 400

    return send_file(
        file_info["path"],
        as_attachment=True,
        download_name=file_info["filename"],
        mimetype="text/plain; charset=utf-8",
    )


@api_bp.route("/api/download-video/<job_id>")
def download_video_file(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    job = jobs[job_id]
    video_file = job.get("video_file")
    if not video_file or not os.path.exists(video_file["path"]):
        return jsonify({"error": "Video không còn khả dụng"}), 404

    original_name = job.get("original_name", "video")
    safe_name = "".join(c for c in original_name if c.isalnum() or c in " _-").strip()[:80]
    if not safe_name:
        safe_name = "video"
    download_name = f"{safe_name}.mp4"


    return send_file(
        video_file["path"],
        as_attachment=True,
        download_name=download_name,
        mimetype="video/mp4",
    )


# ── AI Translation Models & Multi-Speaker ──────────────────────────────

@api_bp.route("/api/translation/models")
def get_translation_models():
    """Return available AI translation models."""
    return jsonify({
        "models": AI_TRANSLATE_MODELS,
        "default": AI_DEFAULT_MODEL,
    })


@api_bp.route("/api/speakers/<job_id>")
def get_job_speakers(job_id):
    """Return detected speakers and current voice assignments."""
    job = get_job(job_id) or {}
    speakers = job.get("speakers")
    if not speakers:
        speakers_file = OUTPUT_FOLDER / job_id / "speakers.json"
        if speakers_file.exists():
            try:
                data = json.loads(speakers_file.read_text(encoding="utf-8"))
                speakers = data.get("speakers")
                if speakers:
                    job["speakers"] = speakers
                    job["segment_speakers"] = data.get("segment_speakers")
            except Exception:
                pass
    if not speakers and not job:
        return jsonify({"error": "Job không tồn tại"}), 404
    if not speakers:
        speakers = {"M1": 1}

    return jsonify({
        "speakers": speakers,
        "default_voice_maps": SPEAKER_VOICE_MAPS,
    })




# ── TTS ────────────────────────────────────────────────────────────────────

@api_bp.route("/api/tts/health")
def tts_health():
    try:
        return jsonify(get_google_tts_health())
    except Exception as exc:
        return jsonify({"ok": False, "configured": False, "error": str(exc)}), 500

@api_bp.route("/api/tts/<job_id>/<lang>", methods=["POST"])
def trigger_tts(job_id, lang):
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    job = jobs[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Job chưa hoàn thành"}), 400

    tts_key = f"tts_{lang}"
    req_data = request.get_json(silent=True) or {}
    force_retry = bool(req_data.get("retry") or req_data.get("force"))
    if tts_key in job:
        tts_status = job[tts_key].get("status", "")
        if tts_status == "generating":
            return jsonify({"message": "Đang tạo TTS...", "status": "generating"})
        elif tts_status in {"done", "partial"} and not force_retry:
            return jsonify({"message": "TTS đã sẵn sàng", "status": "done"})

    srt_files = job.get("srt_files", {})
    if lang not in srt_files:
        return jsonify({"error": f"Không tìm thấy SRT cho ngôn ngữ '{lang}'"}), 404

    srt_info = srt_files[lang]
    if "error" in srt_info:
        return jsonify({"error": srt_info["error"]}), 400

    srt_path = srt_info.get("path", "")
    if not srt_path or not os.path.exists(srt_path):
        return jsonify({"error": "File SRT không tồn tại"}), 404

    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    # Retrieve engine parameter from request
    req_data = request.get_json(silent=True) or {}
    tts_engine = req_data.get("engine", "edge")
    if tts_engine not in {"edge", "omnivoice", "gemini"}:
        return jsonify({"error": "Engine TTS không hợp lệ"}), 400
    tts_options = {
        "voice": str(req_data.get("voice") or "Charon")[:40],
        "emotion": str(req_data.get("emotion") or "warm")[:30],
        "style_prompt": str(req_data.get("style_prompt") or "")[:500],
    }

    speaker_voices = req_data.get("speaker_voices")
    thread = threading.Thread(
        target=tts_worker,
        args=(job_id, lang, srt_content, tts_engine, tts_options, speaker_voices),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started", "message": "Bắt đầu tạo TTS..."})


@api_bp.route("/api/download-audio/<job_id>/<lang>")
def download_tts_audio(job_id, lang):
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    job = jobs[job_id]
    tts_key = f"tts_{lang}"
    tts_info = job.get(tts_key, {})

    if tts_info.get("status") not in {"done", "partial"}:
        return jsonify({"error": "Audio TTS chưa sẵn sàng"}), 400

    audio_path = tts_info.get("path", "")
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"error": "File audio không tồn tại"}), 404

    return send_file(
        audio_path,
        as_attachment=True,
        download_name=tts_info.get("filename", f"tts_{lang}.mp3"),
        mimetype="audio/mpeg",
    )


# ── Video Frame & Burn Sub ─────────────────────────────────────────────────

@api_bp.route("/api/frame/<job_id>")
def get_video_frame(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    job = jobs[job_id]
    video_file = job.get("video_file", {})
    video_path = video_file.get("path", "")

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video không tồn tại"}), 404

    frame_path = str(OUTPUT_FOLDER / f"{job_id}_frame.jpg")

    dur_cmd = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-print_format', 'json', video_path],
        capture_output=True, text=True, timeout=30
    )
    duration = 10
    try:
        duration = float(json.loads(dur_cmd.stdout)["format"]["duration"])
    except:
        pass

    seek_time = min(duration * 0.3, duration - 1)

    subprocess.run([
        'ffmpeg', '-y', '-ss', str(seek_time), '-i', video_path,
        '-frames:v', '1', '-q:v', '3',
        frame_path,
    ], capture_output=True, timeout=30)

    if not os.path.exists(frame_path):
        return jsonify({"error": "Không trích được frame"}), 500

    return send_file(frame_path, mimetype="image/jpeg")


@api_bp.route("/api/burnsub/<job_id>/<lang>", methods=["POST"])
def trigger_burnsub(job_id, lang):
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    job = jobs[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Job chưa hoàn thành"}), 400

    burn_key = f"burn_{lang}"
    if burn_key in job:
        burn_status = job[burn_key].get("status", "")
        if burn_status in {"processing", "generating"}:
            return jsonify({"message": "Đang burn sub...", "status": "processing"})

        if burn_status == "done":
            return jsonify({"message": "Burn sub đã sẵn sàng", "status": "done"})
    video_file = job.get("video_file", {})
    if not video_file.get("path") or not os.path.exists(video_file["path"]):
        return jsonify({"error": "Không tìm thấy video gốc"}), 404

    srt_files = job.get("srt_files", {})
    if lang not in srt_files:
        return jsonify({"error": f"Không tìm thấy SRT cho '{lang}'"}), 404

    srt_info = srt_files[lang]
    if "error" in srt_info:
        return jsonify({"error": srt_info["error"]}), 400

    srt_path = srt_info.get("path", "")
    if not srt_path or not os.path.exists(srt_path):
        return jsonify({"error": "File SRT không tồn tại"}), 404

    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    # Parse multi-region data
    data = request.get_json(silent=True) or {}
    sub_region = None
    extra_regions = []

    # Sub region (where new subtitle goes)
    sr = data.get("sub_region")
    if sr and validate_region(sr):
        sub_region = {
            "x_ratio": float(sr.get("x_ratio", 0)),
            "y_ratio": float(sr["y_ratio"]),
            "w_ratio": float(sr.get("w_ratio", 1.0)),
            "h_ratio": float(sr["h_ratio"]),
        }

    # Extra blur regions (logos, watermarks, etc.)
    for er in data.get("extra_regions", [])[:MAX_BLUR_REGIONS - 1]:
        if validate_region(er):
            extra_regions.append({
                "x_ratio": float(er.get("x_ratio", 0)),
                "y_ratio": float(er["y_ratio"]),
                "w_ratio": float(er.get("w_ratio", 1.0)),
                "h_ratio": float(er["h_ratio"]),
            })

    # Legacy support: old {y_ratio, h_ratio} format
    if not sub_region and "y_ratio" in data and "h_ratio" in data:
        try:
            y_r = float(data["y_ratio"])
            h_r = float(data["h_ratio"])
            if 0 <= y_r <= 1 and 0 < h_r <= 1:
                sub_region = {"x_ratio": 0, "y_ratio": y_r, "w_ratio": 1.0, "h_ratio": h_r}
        except (ValueError, TypeError):
            pass

    thread = threading.Thread(
        target=burnsub_worker,
        args=(job_id, lang, srt_content, sub_region, extra_regions),
        daemon=True,
    )
    thread.start()

    region_count = 1 + len(extra_regions)
    mode = f"thủ công ({region_count} vùng)" if sub_region else "tự động (OCR)"
    return jsonify({"status": "started", "message": f"Bắt đầu burn sub ({mode})..."})


# ── Region Presets ─────────────────────────────────────────────────────────

@api_bp.route("/api/presets")
def get_presets():
    return jsonify(load_presets())


@api_bp.route("/api/presets", methods=["POST"])
def save_preset():
    data = request.get_json()
    if not data or "key" not in data or "name" not in data:
        return jsonify({"error": "Thiếu key hoặc name"}), 400

    key = data["key"].strip().lower().replace(" ", "_")
    if not key or len(key) > 30:
        return jsonify({"error": "Key không hợp lệ"}), 400

    sr = data.get("sub_region", {})
    if not validate_region(sr):
        return jsonify({"error": "sub_region không hợp lệ"}), 400

    extra = []
    for er in data.get("extra_regions", [])[:MAX_BLUR_REGIONS - 1]:
        if validate_region(er):
            extra.append(er)

    presets = load_presets()
    presets[key] = {
        "name": data["name"][:50],
        "sub_region": sr,
        "extra_regions": extra,
    }
    save_presets(presets)

    return jsonify({"status": "ok", "key": key, "total": len(presets)})


@api_bp.route("/api/presets/<key>", methods=["DELETE"])
def delete_preset(key):
    presets = load_presets()
    if key not in presets:
        return jsonify({"error": "Preset không tồn tại"}), 404
    if key == "default":
        return jsonify({"error": "Không thể xóa preset mặc định"}), 400

    del presets[key]
    save_presets(presets)
    return jsonify({"status": "ok", "remaining": len(presets)})


@api_bp.route("/api/download-burned-video/<job_id>/<lang>")
def download_burned_video(job_id, lang):
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    job = jobs[job_id]
    burn_key = f"burn_{lang}"
    burn_info = job.get(burn_key, {})

    if burn_info.get("status") != "done":
        return jsonify({"error": "Video chưa sẵn sàng"}), 400

    video_path = burn_info.get("path", "")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "File video không tồn tại"}), 404

    return send_file(
        video_path,
        as_attachment=True,
        download_name=burn_info.get("filename", f"video_{lang}_sub.mp4"),
        mimetype="video/mp4",
    )


# ── Gemini API Key ─────────────────────────────────────────────────────────

@api_bp.route("/api/gemini/key", methods=["GET"])
def get_gemini_key_status():
    """Check shared Gemini credential state without exposing key fragments."""
    health = get_google_tts_health()
    return jsonify({
        "has_key": health["configured"],
        "credential_source": health["credential_source"],
    })


@api_bp.route("/api/gemini/key", methods=["POST"])
def save_gemini_key():
    """Save Gemini API key after verifying it works"""
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "API key không được để trống"}), 400

    # Verify key by listing models
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Quick test: list models (lightweight call)
        models = list(client.models.list())
        if not models:
            return jsonify({"error": "API key không hợp lệ - không tìm thấy models"}), 400
    except Exception as e:
        err_msg = str(e).lower()
        if "api_key" in err_msg or "invalid" in err_msg or "permission" in err_msg:
            return jsonify({"error": "API key không hợp lệ. Kiểm tra lại tại aistudio.google.com"}), 400
        # Network error — save anyway but warn
        print(f"⚠️ Cannot verify Gemini key (network?): {e}")

    set_gemini_api_key(api_key)
    return jsonify({"status": "ok", "verified": True})


# ── Gemini Models ──────────────────────────────────────────────────────────

@api_bp.route("/api/gemini/models")
def get_gemini_models():
    """Return list of available Gemini models for hardsub extraction"""
    return jsonify({
        "models": GEMINI_MODELS,
        "default": GEMINI_DEFAULT_MODEL,
    })


# ── Hardsub Extraction ─────────────────────────────────────────────────────

@api_bp.route("/api/hardsub", methods=["POST"])
def start_hardsub():
    """Start hardsub extraction from uploaded video using Gemini API"""
    if "video" not in request.files:
        return jsonify({"error": "Không tìm thấy file video"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    ext, validation_error = _validate_uploaded_video(video)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"source{ext}"
    video.save(str(video_path))
    if not video_path.exists() or video_path.stat().st_size == 0:
        return jsonify({"error": "File upload rỗng hoặc không thể ghi"}), 400

    valid_media, media_error = _validate_media_path(video_path)
    if not valid_media:
        video_path.unlink(missing_ok=True)
        return jsonify({"error": media_error}), 400
    gemini_model = request.form.get("gemini_model", GEMINI_DEFAULT_MODEL)
    gemini_api_key = request.form.get("gemini_api_key", "").strip()

    # If key provided, also save it for future use
    if gemini_api_key:
        set_gemini_api_key(gemini_api_key)
    else:
        gemini_api_key = get_gemini_api_key()

    if not gemini_api_key:
        return jsonify({"error": "Chưa có Gemini API Key"}), 400

    translate_langs = _parse_languages(request.form.get("translate_langs", "[]"))
    translate_method = request.form.get("translate_method", "google")
    if translate_method not in ("ai", "google"):
        translate_method = "google"
    if gemini_model not in GEMINI_MODELS:
        gemini_model = GEMINI_DEFAULT_MODEL

    create_job(
        job_id,
        original_name=str(video.filename),
        video_path=str(video_path),
        video_file=_video_info(str(video_path)),
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        translate_langs=translate_langs,
        translate_method=translate_method,
        mode="hardsub",
    )

    cleanup_old_jobs()

    thread = threading.Thread(
        target=hardsub_worker,
        args=(jobs[job_id],),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@api_bp.route("/api/hardsub-url", methods=["POST"])
def start_hardsub_url():
    """Start hardsub extraction from URL video using Gemini API"""
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    valid_url, url_error = validate_download_url(url)
    if not valid_url:
        return jsonify({"error": url_error}), 400

    gemini_model = data.get("gemini_model", GEMINI_DEFAULT_MODEL)
    gemini_api_key = data.get("gemini_api_key", "").strip()

    if gemini_api_key:
        set_gemini_api_key(gemini_api_key)
    else:
        gemini_api_key = get_gemini_api_key()

    if not gemini_api_key:
        video_path.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        return jsonify({"error": "Chưa có Gemini API Key"}), 400

    translate_langs = data.get("translate_langs", [])
    translate_langs = [l for l in translate_langs if l in LANGUAGES]

    translate_method = data.get("translate_method", "google")
    if translate_method not in ("ai", "google"):
        translate_method = "google"

    job_id = uuid.uuid4().hex[:12]

    if gemini_model not in GEMINI_MODELS:
        gemini_model = GEMINI_DEFAULT_MODEL
    create_job(
        job_id,
        status="downloading_video",
        message="Đang tải video từ URL...",
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        translate_langs=_parse_languages(data.get("translate_langs", [])),
        translate_method=translate_method,
        mode="hardsub",
    )

    cleanup_old_jobs()

    def download_then_hardsub():
        job = jobs[job_id]
        try:
            video_path = download_from_url(job_id, url)
            if not video_path:
                job["status"] = "error"
                job["message"] = "Tải video thất bại"
                return
            job["video_path"] = video_path
            job["video_file"] = _video_info(video_path)
            hardsub_worker(job)
        except Exception as e:
            job["status"] = "error"
            job["message"] = f"Lỗi: {str(e)}"

    thread = threading.Thread(target=download_then_hardsub, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})





