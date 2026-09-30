"""
api.py — Flask Blueprint with all API routes
"""

import os
import re
import json
import uuid
import time
import subprocess
import threading
import shutil
from pathlib import Path
from typing import Any

from flask import Blueprint, request, jsonify, send_file as _send_file, make_response
from services.runtime_state import valid_job_id, resolve_media_path, host_lock, public_data, TERMINAL
from services.pipeline_options import validate_tts_options

from config import (
    parse_bool,
    LANGUAGES, VALID_MODELS, DEVICE, COMPUTE_TYPE,
    UPLOAD_FOLDER, OUTPUT_FOLDER,
    MAX_UPLOAD_BYTES, MAX_VIDEO_DURATION_SECONDS,
    jobs, cleanup_old_jobs, create_job, get_job, safe_stem,
    load_presets, save_presets, validate_region, MAX_BLUR_REGIONS,
    GEMINI_MODELS, GEMINI_DEFAULT_MODEL,
    AI_TRANSLATE_MODELS, AI_DEFAULT_MODEL, SPEAKER_VOICE_MAPS, TTS_VOICES,
    get_gemini_api_key, set_gemini_api_key,
    TRANSLATION_MODES, DEFAULT_TRANSLATION_MODE,
    BATCH_ALLOWED_ROOTS, BATCH_MAX_FILES,
)
from services.whisper_engine import process_video
from services.tts import tts_worker
from services.srt_to_tts import parse_srt_data, process_srt_to_tts
from services.google_tts import get_google_tts_health
from services.burn_sub import burnsub_worker
from services.downloader import download_from_url, process_url_video, validate_download_url
from services.hardsub_gemini import hardsub_worker
from services.job_runner import submit
from services.batch_pipeline import (
    BatchRunner,
    create_batch_manifest,
    load_batch_manifest,
    recover_stale_batch,
    run_batch,
    scan_video_folder,
    validate_batch_options,
)

api_bp = Blueprint("api", __name__)


@api_bp.before_request
def validate_request_boundary():
    jid = (request.view_args or {}).get("job_id")
    if jid is not None and not valid_job_id(jid):
        return jsonify({"error": "job_id không hợp lệ"}), 400
    if request.is_json and not isinstance(request.get_json(silent=True), dict):
        return jsonify({"error": "Dữ liệu JSON phải là object"}), 400
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return jsonify({"error": "Origin không được phép"}), 403
    if jid:
        get_job(jid)


@api_bp.errorhandler(ValueError)
def invalid_option(error):
    return jsonify({"error": str(error)}), 400


@api_bp.errorhandler(TimeoutError)
def resource_busy(error):
    return jsonify({"error": str(error)}), 409


def send_file(path, **kwargs):
    path = resolve_media_path(path, UPLOAD_FOLDER, OUTPUT_FOLDER)
    return _send_file(path, **kwargs)

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
        if MAX_UPLOAD_BYTES > 0 and size > MAX_UPLOAD_BYTES:
            return False, f"File quá lớn ({MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB tối đa)"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-print_format", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        if MAX_VIDEO_DURATION_SECONDS > 0 and duration > MAX_VIDEO_DURATION_SECONDS:
            return False, f"Video quá dài ({MAX_VIDEO_DURATION_SECONDS // 3600} giờ tối đa)"
        return True, ""
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, "File không phải video hợp lệ hoặc FFprobe không đọc được"
# ── Index ──────────────────────────────────────────────────────────────────

@api_bp.route("/srt-to-tts")
def srt_to_tts_view():
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    html = template_path.read_text(encoding="utf-8")
    script = "<script>window.addEventListener('DOMContentLoaded', () => switchInputMode('srttts'));</script>"
    html = html.replace("</body>", f"{script}\r\n</body>")
    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    return resp


@api_bp.route("/")
def index():
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    html = template_path.read_text(encoding="utf-8")
    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# ── Device Info ────────────────────────────────────────────────────────────

@api_bp.route("/api/device")
def device_info():
    info = {
        "device": DEVICE.upper(),
        "compute_type": COMPUTE_TYPE,
        "cpu_threads": os.cpu_count() or 4,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "unlimited_upload": MAX_UPLOAD_BYTES <= 0,
        "max_duration_seconds": MAX_VIDEO_DURATION_SECONDS,
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



def process_clean_only_video(
    job_id: str,
    video_path: str,
    engine: str = "opencv",
    trim_intro: str = "off",
    clean_hardsub: bool = True,
    clean_logo: bool = False,
    clean_title: bool = False,
    sub_region: dict | None = None,
    extra_regions: list | None = None,
):
    """Trực tiếp xóa phụ đề cũ và giữ nguyên 100% âm thanh gốc, không cần qua bước dịch thuật."""
    job = jobs[job_id]
    job["status"] = "processing"
    job["progress"] = 15
    job["message"] = f"🧹 Đang chuẩn bị xóa phụ đề ({engine})..."
    try:
        total_started = time.time()
        # Step 0: Cắt bìa nếu yêu cầu
        if trim_intro and str(trim_intro).lower() != "off":
            from services.video.trimmer import preprocess_trim_video
            output_dir = OUTPUT_FOLDER / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            trimmed_dest = str(output_dir / "video_clean_cover.mp4")
            clean_vid, trim_sec = preprocess_trim_video(video_path, output_path=trimmed_dest, trim_mode=str(trim_intro))
            if trim_sec > 0.05:
                video_path = clean_vid
                job["trimmed_seconds"] = trim_sec
                job["video_path"] = video_path

        # Step 1: Xác định vùng phụ đề
        if not sub_region:
            from services.burn_sub import detect_hardsub_region
            sub_region = detect_hardsub_region(video_path, job_id=job_id, srt_content=None)

        if not extra_regions:
            extra_regions = []
            if clean_logo:
                extra_regions.append({"x_ratio": 0.02, "y_ratio": 0.02, "w_ratio": 0.20, "h_ratio": 0.06})
            if clean_title:
                extra_regions.append({"x_ratio": 0.05, "y_ratio": 0.04, "w_ratio": 0.90, "h_ratio": 0.08})

        output_dir = OUTPUT_FOLDER / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        video_stem = safe_stem(job.get("original_name", "video"))
        out_file = str(output_dir / f"{video_stem}_clean.mp4")

        from services.video.clean_pipeline import clean_video_pipeline
        clean_res = clean_video_pipeline(
            video_path=video_path,
            sub_region=sub_region,
            srt_content="",
            output_path=out_file,
            job_id=job_id,
            burn_key="burn_clean",
            engine=engine,
            re_burn_ass_path=None,
            tts_audio_path=None,  # Bắt buộc giữ nguyên 100% âm thanh gốc!
            extra_regions=extra_regions if extra_regions else None,
        )

        total_time = round(time.time() - total_started, 1)
        job.update({
            "status": "done",
            "progress": 100,
            "total_time": total_time,
            "duration": clean_res.get("duration", 0),
            "segment_count": 0,
            "message": f"Hoàn tất xóa phụ đề! Âm thanh gốc được giữ nguyên 100%.",
            "burn_clean": clean_res,
            "clean_video": clean_res,
            "mode": "clean_only",
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        job["status"] = "error"
        job["message"] = f"Lỗi xóa phụ đề: {exc}"


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
    if ai_model not in AI_TRANSLATE_MODELS:
        ai_model = AI_DEFAULT_MODEL
    translate_method = request.form.get("translate_method", "ai")
    if translate_method not in ("ai", "google"):
        translate_method = "ai"
    translation_mode = request.form.get("translation_mode", DEFAULT_TRANSLATION_MODE)
    if translation_mode not in TRANSLATION_MODES:
        translation_mode = DEFAULT_TRANSLATION_MODE

    trim_intro = request.form.get("trim_intro", "auto")
    action_mode = request.form.get("action_mode", "translate")

    if action_mode == "clean_only":
        clean_engine = request.form.get("clean_engine", "opencv")
        clean_hardsub = parse_bool(request.form.get("clean_hardsub"), default=True)
        clean_logo = parse_bool(request.form.get("clean_logo"), default=False)
        clean_title = parse_bool(request.form.get("clean_title"), default=False)

        create_job(
            job_id,
            original_name=str(video.filename),
            video_path=str(video_path),
            video_file=_video_info(str(video_path)),
            mode="clean_only",
            status="processing",
            message="🧹 Đang chuẩn bị xóa phụ đề...",
            trim_intro=trim_intro,
        )
        cleanup_old_jobs()
        thread = threading.Thread(
            target=process_clean_only_video,
            args=(job_id, str(video_path), clean_engine, trim_intro, clean_hardsub, clean_logo, clean_title),
            daemon=True,
        )
        thread.start()
        return jsonify({"job_id": job_id, "status": "processing", "mode": "clean_only"})

    create_job(
        job_id,
        original_name=str(video.filename),
        video_path=str(video_path),
        translate_langs=translate_langs,
        translate_method=translate_method,
        translation_mode=translation_mode,
        ai_model=ai_model,
        trim_intro=trim_intro,
    )

    cleanup_old_jobs()

    thread = threading.Thread(
        target=process_video,
        args=(job_id, str(video_path), model_size, translate_langs, translate_method, translation_mode),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})



# ── Chunked Upload (Phân đoạn xử lý file lớn / không giới hạn > 2GB) ──────────

_chunk_lock = threading.RLock()
_chunk_uploads: dict[str, dict[str, Any]] = {}


def _save_chunk_session(session):
    path = UPLOAD_FOLDER / session['job_id'] / 'upload-session.json'
    data = public_data(session)
    path.with_suffix('.tmp').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    path.with_suffix('.tmp').replace(path)


def _load_chunk_session(job_id):
    if not valid_job_id(job_id):
        raise ValueError('job_id không hợp lệ')
    path = UPLOAD_FOLDER / job_id / 'upload-session.json'
    if path.is_file():
        session = json.loads(path.read_text(encoding='utf-8'))
        session['received_chunks'] = set(session['received_chunks'])
        session['video_path'] = str(resolve_media_path(session['video_path'], UPLOAD_FOLDER))
        _chunk_uploads[job_id] = session
        return session
    return _chunk_uploads.get(job_id)


def _cleanup_expired_chunk_sessions():
    now = time.time()
    with _chunk_lock:
        expired = [
            jid for jid, sess in _chunk_uploads.items()
            if now - sess.get("created_at", 0) > 7200
        ]
        for jid in expired:
            _chunk_uploads.pop(jid, None)


@api_bp.route("/api/upload/init", methods=["POST"])
def upload_init():
    """Khởi tạo phiên upload phân đoạn (chunked upload) cho file bất kỳ kích thước."""
    _cleanup_expired_chunk_sessions()
    data = request.get_json(silent=True) or request.form.to_dict()
    filename = str(data.get("filename", "")).strip()
    if not filename:
        return jsonify({"error": "Chưa cung cấp tên file"}), 400
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": f"Định dạng {ext or 'không xác định'} không được hỗ trợ"}), 400

    try:
        total_size = int(data.get("total_size", 0))
    except (ValueError, TypeError):
        total_size = 0

    if MAX_UPLOAD_BYTES > 0 and total_size > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File vượt giới hạn cho phép ({MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB)"}), 400

    try:
        chunk_size = int(data.get("chunk_size", 20 * 1024 * 1024))
    except (ValueError, TypeError):
        chunk_size = 20 * 1024 * 1024

    if chunk_size <= 0:
        chunk_size = 20 * 1024 * 1024

    try:
        total_chunks = int(data.get("total_chunks", 0))
    except (ValueError, TypeError):
        total_chunks = 0

    if total_chunks <= 0:
        total_chunks = max(1, (total_size + chunk_size - 1) // chunk_size) if total_size > 0 else 1

    if total_size <= 0 or not 1 <= chunk_size <= 64 * 1024 * 1024:
        return jsonify({'error': 'Dung lượng file phải dương; chunk tối đa 64 MiB'}), 400
    if total_chunks != (total_size + chunk_size - 1) // chunk_size or total_chunks > 1000000:
        return jsonify({'error': 'Số chunk không khớp dung lượng file'}), 400
    if shutil.disk_usage(UPLOAD_FOLDER).free < total_size + 256 * 1024 * 1024:
        return jsonify({'error': 'Không đủ dung lượng đĩa cho file upload'}), 507

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"source{ext}"

    # Tạo trước file rỗng
    with open(video_path, "wb") as f:
        pass

    with _chunk_lock:
        _chunk_uploads[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "ext": ext,
            "video_path": str(video_path),
            "total_size": total_size,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "received_chunks": set(),
            "created_at": time.time(),
            "metadata": dict(data),
        }
        _save_chunk_session(_chunk_uploads[job_id])
    create_job(job_id, status='uploading', original_name=filename, video_path=str(video_path))

    return jsonify({
        "job_id": job_id,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "status": "initialized",
    })


@api_bp.route("/api/upload/chunk", methods=["POST"])
def upload_chunk():
    """Nhận và ghi một phần dữ liệu (chunk) trực tiếp vào file đĩa."""
    job_id = request.form.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "Thiếu job_id"}), 400

    with host_lock('upload-' + job_id, timeout=30):
        return _receive_chunk(job_id)


def _receive_chunk(job_id):
    session = _load_chunk_session(job_id)
    if not session:
        return jsonify({"error": "Phiên upload không tồn tại hoặc đã hết hạn"}), 404

    try:
        chunk_index = int(request.form.get("chunk_index", -1))
    except (ValueError, TypeError):
        return jsonify({"error": "chunk_index không hợp lệ"}), 400

    if chunk_index < 0 or chunk_index >= session["total_chunks"]:
        return jsonify({"error": f"chunk_index {chunk_index} ngoài phạm vi (0..{session['total_chunks']-1})"}), 400

    if "chunk" not in request.files:
        return jsonify({"error": "Thiếu dữ liệu chunk"}), 400

    chunk_file = request.files["chunk"]
    if session.get('finished_response'):
        return jsonify({'error': 'Upload đã hoàn tất'}), 409

    chunk_size = session["chunk_size"]
    video_path = Path(session["video_path"])
    offset = chunk_index * chunk_size

    temp_part = video_path.with_suffix('.chunk')
    expected = min(chunk_size, session['total_size'] - offset)
    try:
        received = 0
        with temp_part.open('wb') as part:
            while True:
                buf = chunk_file.stream.read(min(1024 * 1024, expected - received + 1))
                if not buf:
                    break
                received += len(buf)
                if received > expected:
                    return jsonify({'error': 'Chunk vượt kích thước khai báo'}), 400
                part.write(buf)
        if received != expected:
            return jsonify({'error': 'Chunk chưa đủ kích thước khai báo'}), 400
        with video_path.open('r+b') as f, temp_part.open('rb') as part:
            f.seek(offset)
            shutil.copyfileobj(part, f, 1024 * 1024)
    except Exception as e:
        return jsonify({"error": f"Lỗi ghi chunk: {e}"}), 500
    finally:
        temp_part.unlink(missing_ok=True)

    with _chunk_lock:
        session["received_chunks"].add(chunk_index)
        received_count = len(session["received_chunks"])
        session['last_activity'] = time.time()
        _save_chunk_session(session)

    return jsonify({
        "job_id": job_id,
        "chunk_index": chunk_index,
        "received_chunks": received_count,
        "total_chunks": session["total_chunks"],
        "status": "ok",
    })


@api_bp.route("/api/upload/finish", methods=["POST"])
def upload_finish():
    data = request.get_json(silent=True) or request.form.to_dict()
    jid = str(data.get('job_id', '')).strip()
    if not valid_job_id(jid):
        raise ValueError('job_id không hợp lệ')
    with host_lock('upload-' + jid, timeout=60):
        session = _load_chunk_session(jid)
        if session and session.get('finished_response'):
            return jsonify(session['finished_response'])
        response = _finish_upload()
        if not isinstance(response, tuple) and response.status_code == 200 and session:
            session['finished_response'] = response.get_json()
            _save_chunk_session(session)
        return response


def _finish_upload():
    """Kiểm tra toàn vẹn file sau upload phân đoạn và bắt đầu tiến trình xử lý."""
    data = request.get_json(silent=True) or request.form.to_dict()
    job_id = str(data.get("job_id", "")).strip()
    if not job_id:
        return jsonify({"error": "Thiếu job_id"}), 400

    with _chunk_lock:
        session = _load_chunk_session(job_id)
    if not session:
        return jsonify({"error": "Phiên upload không tồn tại hoặc đã hoàn tất"}), 404

    total_chunks = session["total_chunks"]
    missing_chunks = set(range(total_chunks)) - session["received_chunks"]
    if missing_chunks:
        return jsonify({
            "error": f"Chưa nhận đủ dữ liệu. Thiếu {len(missing_chunks)} phần ({sorted(list(missing_chunks))[:10]})"
        }), 400

    video_path = Path(session["video_path"])
    if not video_path.exists() or video_path.stat().st_size != session["total_size"]:
        return jsonify({"error": "File sau khi ghép rỗng hoặc không hợp lệ"}), 400

    valid_media, media_error = _validate_media_path(video_path)
    if not valid_media:
        video_path.unlink(missing_ok=True)
        return jsonify({"error": media_error}), 400

    meta = session.get("metadata", {})
    upload_type = meta.get("upload_type", "whisper")

    if upload_type == 'pipeline':
        create_job(job_id, status='uploaded', video_path=str(video_path), original_name=session['filename'], video_file=_video_info(str(video_path)))
        return jsonify({'job_id': job_id, 'status': 'uploaded', 'video_path': str(video_path)})

    if upload_type == "hardsub":
        gemini_model = meta.get("gemini_model", GEMINI_DEFAULT_MODEL)
        gemini_api_key = str(meta.get("gemini_api_key", "")).strip()
        if gemini_api_key:
            set_gemini_api_key(gemini_api_key)
        else:
            gemini_api_key = get_gemini_api_key()

        if not gemini_api_key and gemini_model not in ("gemini-3.8-flash-high", "gemini-3.7-flash-high"):
            return jsonify({"error": "Chưa có Gemini API Key"}), 400

        translate_langs = _parse_languages(meta.get("translate_langs", "[]"))
        ai_model = meta.get("ai_model", AI_DEFAULT_MODEL)
        if ai_model not in AI_TRANSLATE_MODELS:
            ai_model = AI_DEFAULT_MODEL
        translate_method = meta.get("translate_method", "google")
        if translate_method not in ("ai", "google"):
            translate_method = "google"
        if gemini_model not in GEMINI_MODELS:
            gemini_model = GEMINI_DEFAULT_MODEL

        translation_mode = meta.get("translation_mode", DEFAULT_TRANSLATION_MODE)
        if translation_mode not in TRANSLATION_MODES:
            translation_mode = DEFAULT_TRANSLATION_MODE

        trim_intro = meta.get("trim_intro", "auto")
        create_job(
            job_id,
            original_name=session["filename"],
            video_path=str(video_path),
            video_file=_video_info(str(video_path)),
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            visual_timing_enabled=parse_bool(meta.get('visual_timing'), default=False),
            translate_langs=translate_langs,
            translate_method=translate_method,
            translation_mode=translation_mode,
            ai_model=ai_model,
            trim_intro=trim_intro,
        )

        cleanup_old_jobs()
        job = get_job(job_id)
        thread = threading.Thread(target=hardsub_worker, args=(job,), daemon=True)
        thread.start()
        return jsonify({"job_id": job_id, "status": "queued"})
    else:
        model_size = meta.get("model_size", "large-v3-turbo")
        if model_size not in VALID_MODELS:
            model_size = "large-v3-turbo"

        translate_langs = _parse_languages(meta.get("translate_langs", "[]"))
        ai_model = meta.get("ai_model", AI_DEFAULT_MODEL)
        if ai_model not in AI_TRANSLATE_MODELS:
            ai_model = AI_DEFAULT_MODEL
        translate_method = meta.get("translate_method", "ai")
        if translate_method not in ("ai", "google"):
            translate_method = "ai"
        translation_mode = meta.get("translation_mode", DEFAULT_TRANSLATION_MODE)
        if translation_mode not in TRANSLATION_MODES:
            translation_mode = DEFAULT_TRANSLATION_MODE

        trim_intro = meta.get("trim_intro", "auto")
        action_mode = meta.get("action_mode", "translate")

        if action_mode == "clean_only":
            clean_engine = meta.get("clean_engine", "opencv")
            clean_hardsub = parse_bool(meta.get("clean_hardsub"), default=True)
            clean_logo = parse_bool(meta.get("clean_logo"), default=False)
            clean_title = parse_bool(meta.get("clean_title"), default=False)

            create_job(
                job_id,
                original_name=session["filename"],
                video_path=str(video_path),
                video_file=_video_info(str(video_path)),
                mode="clean_only",
                status="processing",
                message="🧹 Đang khởi tạo xóa phụ đề...",
                trim_intro=trim_intro,
            )
            cleanup_old_jobs()
            thread = threading.Thread(
                target=process_clean_only_video,
                args=(job_id, str(video_path), clean_engine, trim_intro, clean_hardsub, clean_logo, clean_title),
                daemon=True,
            )
            thread.start()
            return jsonify({"job_id": job_id, "status": "processing", "mode": "clean_only"})

        create_job(
            job_id,
            original_name=session["filename"],
            video_path=str(video_path),
            translate_langs=translate_langs,
            translate_method=translate_method,
            translation_mode=translation_mode,
            ai_model=ai_model,
            trim_intro=trim_intro,
        )

        cleanup_old_jobs()
        thread = threading.Thread(
            target=process_video,
            args=(job_id, str(video_path), model_size, translate_langs, translate_method, translation_mode),
            daemon=True,
        )
        thread.start()
        return jsonify({"job_id": job_id, "status": "queued"})


@api_bp.route("/api/upload/cancel", methods=["POST"])
def upload_cancel():
    data = request.get_json(silent=True) or request.form.to_dict()
    job_id = str(data.get("job_id", "")).strip()
    if not valid_job_id(job_id):
        raise ValueError('job_id không hợp lệ')
    if not job_id:
        return jsonify({"error": "Thiếu job_id"}), 400

    with host_lock('upload-' + job_id, timeout=5):
        session = _load_chunk_session(job_id)
        if session:
            if session.get('finished_response'):
                raise TimeoutError('Upload đã hoàn tất; dùng chức năng dừng job để hủy xử lý.')
            target = (UPLOAD_FOLDER / job_id).resolve()
            if target.parent != UPLOAD_FOLDER.resolve():
                raise ValueError('Đường dẫn upload không hợp lệ')
            shutil.rmtree(target, ignore_errors=False)
            _chunk_uploads.pop(job_id, None)
            job = get_job(job_id)
            if job:
                job.update(status='cancelled', cancel=True, message='Đã hủy upload')
            return jsonify({"status": "cancelled", "job_id": job_id})
    return jsonify({"status": "not_found", "job_id": job_id})


# ── URL Download ───────────────────────────────────────────────────────────

@api_bp.route("/api/url", methods=["POST"])
def url_download():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Thiếu URL"}), 400

    from services.downloader import clean_download_url
    url = clean_download_url(data.get("url", ""))
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
    if ai_model not in AI_TRANSLATE_MODELS:
        ai_model = AI_DEFAULT_MODEL
    translate_method = data.get("translate_method", "ai")
    if translate_method not in ("ai", "google"):
        translate_method = "ai"
    translation_mode = data.get("translation_mode", DEFAULT_TRANSLATION_MODE)
    if translation_mode not in TRANSLATION_MODES:
        translation_mode = DEFAULT_TRANSLATION_MODE
    jobs[job_id]["translate_method"] = translate_method
    jobs[job_id]["translation_mode"] = translation_mode
    jobs[job_id]["ai_model"] = ai_model
    jobs[job_id]["trim_intro"] = data.get("trim_intro", "auto")
    action_mode = data.get("action_mode", "translate")

    if action_mode == "clean_only":
        clean_engine = data.get("clean_engine", "opencv")
        clean_hardsub = parse_bool(data.get("clean_hardsub"), default=True)
        clean_logo = parse_bool(data.get("clean_logo"), default=False)
        clean_title = parse_bool(data.get("clean_title"), default=False)
        trim_intro = data.get("trim_intro", "auto")

        def download_then_clean():
            job = jobs[job_id]
            try:
                v_path = download_from_url(job_id, url)
                if not v_path:
                    job["status"] = "error"
                    job["message"] = "Tải video thất bại"
                    return
                job["video_path"] = v_path
                job["video_file"] = _video_info(v_path)
                process_clean_only_video(job_id, v_path, clean_engine, trim_intro, clean_hardsub, clean_logo, clean_title)
            except Exception as e:
                job["status"] = "error"
                job["message"] = f"Lỗi: {e}"

        job = jobs[job_id]
        job["mode"] = "clean_only"
        thread = threading.Thread(target=download_then_clean, daemon=True)
        thread.start()
        return jsonify({"job_id": job_id, "status": "queued", "mode": "clean_only"})

    thread = threading.Thread(
        target=process_url_video,
        args=(job_id, url, model_size, translate_langs, translate_method, translation_mode),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "downloading_video"})


# ── Job Status & Control ──────────────────────────────────────────────────

def _declared_artifact_paths(job):
    """Return output artifacts promised by a job, without treating source media as output."""
    found = []
    containers = [job.get('srt_files', {}), job.get('step_results', {})]
    containers.extend(value for key, value in job.items()
                      if isinstance(value, dict) and (key.startswith('tts_') or key.startswith('burn_')
                                                     or key in {'clean_video', 'burned_video', 'aligned_srt'}))
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {'path', 'audio_path', 'srt_path'} and isinstance(child, str):
                    try:
                        resolved = Path(child).resolve()
                        if resolved.is_relative_to(OUTPUT_FOLDER.resolve()):
                            found.append(resolved)
                    except (OSError, ValueError):
                        pass
                else:
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    for container in containers:
        visit(container)
    return list(dict.fromkeys(found))


def _strip_missing_artifact_paths(value):
    """Avoid returning broken download links while preserving status metadata."""
    if isinstance(value, dict):
        result = {key: _strip_missing_artifact_paths(child) for key, child in value.items()}
        raw_path = value.get('path')
        if isinstance(raw_path, str):
            try:
                path = Path(raw_path).resolve()
                if path.is_relative_to(OUTPUT_FOLDER.resolve()) and not path.is_file():
                    result.pop('path', None)
                    result['artifact_missing'] = True
            except (OSError, ValueError):
                pass
        return result
    if isinstance(value, list):
        return [_strip_missing_artifact_paths(item) for item in value]
    return value

@api_bp.route("/api/status/<job_id>")
def get_status(job_id):
    job = get_job(job_id)
    if job is None:
        output_dir = OUTPUT_FOLDER / job_id
        if output_dir.exists() and output_dir.is_dir():
            srt_files = {}
            for srt_path in output_dir.glob("*.srt"):
                stem = srt_path.stem
                lang = "zh"
                for l in ("vi", "en", "id"):
                    if stem.endswith(f"_{l}") or stem.endswith(f".{l}"):
                        lang = l
                        break
                meta = LANGUAGES.get(lang, {"name": "中文 (原文)" if lang == "zh" else lang, "flag": "🇨🇳" if lang == "zh" else "🏳️"})
                try:
                    content = srt_path.read_text(encoding="utf-8")
                except Exception:
                    content = ""
                srt_files[lang] = {
                    "path": str(srt_path),
                    "filename": srt_path.name,
                    "preview": content[:1500],
                    "lang_name": meta.get("name", lang),
                    "flag": meta.get("flag", "🏳️"),
                }
            if srt_files:
                speakers = {}
                speakers_file = output_dir / "speakers.json"
                if speakers_file.exists():
                    try:
                        data = json.loads(speakers_file.read_text(encoding="utf-8"))
                        speakers = data.get("speakers", {})
                    except Exception:
                        pass
                upload_dir = UPLOAD_FOLDER / job_id
                video_file = None
                if upload_dir.exists():
                    for v_path in upload_dir.glob("source.*"):
                        video_file = {"path": str(v_path), "filename": v_path.name, "size": v_path.stat().st_size}
                        break
                job = create_job(
                    job_id,
                    status="interrupted",
                    progress=100,
                    message="Đã tìm thấy file cũ; chưa xác minh quy trình đã hoàn tất.",
                    legacy_unverified=True,
                    segment_count=len(srt_files.get("zh", {}).get("preview", "").split("\n\n")),
                    srt_files=srt_files,
                    speakers=speakers,
                    video_file=video_file,
                )
                for tts_file in output_dir.glob("tts_*.mp3"):
                    t_lang = tts_file.stem.removeprefix("tts_").removeprefix("aligned_")
                    job[f"tts_{t_lang}"] = {
                        "status": "done",
                        "progress": 100,
                        "message": f"Hoàn thành ({tts_file.stat().st_size / 1048576:.1f}MB)",
                        "path": str(tts_file),
                        "filename": tts_file.name,
                        "size": tts_file.stat().st_size,
                    }

                # Restore burned/clean videos from disk if present
                burn_candidates = list(output_dir.glob("*_sub.mp4")) + list(output_dir.glob("burned_*.mp4")) + list(output_dir.glob("*_clean.mp4")) + list(OUTPUT_FOLDER.glob(f"{job_id}_burned_*.mp4"))
                for b_path in burn_candidates:
                    if b_path.exists() and b_path.stat().st_size > 0:
                        b_lang = "vi"
                        for l in ("vi", "en", "id"):
                            if f"_{l}" in b_path.name:
                                b_lang = l
                                break
                        job[f"burn_{b_lang}"] = {
                            "status": "done",
                            "progress": 100,
                            "message": f"Hoàn thành ({b_path.stat().st_size / 1048576:.1f}MB)",
                            "path": str(b_path),
                            "filename": b_path.name,
                            "size": b_path.stat().st_size,
                        }
    if job is None:
        return jsonify({"error": "Job không tồn tại"}), 404

    from services.pipeline_completion import reconcile_final_render
    reconcile_final_render(job, job_id, OUTPUT_FOLDER)

    missing_artifacts = [path for path in _declared_artifact_paths(job) if not path.is_file()]
    if missing_artifacts:
        job['artifact_warnings'] = [path.name for path in missing_artifacts]
        if job.get('status') == 'done':
            job.update(status='partial', active_step='artifact_missing',
                       message=f'Tác vụ từng hoàn tất nhưng hiện thiếu {len(missing_artifacts)} tệp kết quả; cần chạy lại hoặc khôi phục file.')

    if "speakers" not in job:
        speakers_file = OUTPUT_FOLDER / job_id / "speakers.json"
        if speakers_file.exists():
            try:
                data = json.loads(speakers_file.read_text(encoding="utf-8"))
                job["speakers"] = data.get("speakers", {"M1": 1})
                job["segment_speakers"] = data.get("segment_speakers", [])
            except Exception:
                pass

    if hasattr(job, 'persist') and job.get('_owner_pid') == os.getpid():
        job.persist()
    safe_data = _strip_missing_artifact_paths(public_data(job))
    return jsonify(safe_data)


@api_bp.route("/api/stop/<job_id>", methods=["POST"])
def stop_job(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job không tồn tại"}), 404
    if job.get('status') == 'cancelled':
        return jsonify({"status": "ok", "cancel_requested": True,
                        "already_requested": True, "worker_stopped": False})
    if job.get('status') in TERMINAL:
        return jsonify({"error": "Công việc đã kết thúc; không thể dừng kết quả đã hoàn tất."}), 409
    # A remote cancel must not overwrite a newer worker snapshot.
    jobs.store.cancel(job_id)
    if job.get('_owner_pid') == os.getpid():
        job["cancel"] = True
    job.update(status='cancelled', message='Đã ghi nhận yêu cầu dừng; công đoạn hiện tại có thể cần thời gian để kết thúc.')

    for process_key in ("_ffmpeg_process", "_download_process", "_tts_process"):
        process = job.get(process_key)
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    return jsonify({"status": "ok", "cancel_requested": True, "worker_stopped": False})
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
    if job["status"] not in {"done", "partial", "awaiting_review", "interrupted", "processing"}:
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

@api_bp.route("/api/render-modes")
def get_render_modes():
    """Get available video processing / inpainting modes."""
    return jsonify({
        "modes": {
            "blur": {
                "id": "blur",
                "name": "Dynamic Blur & Burn (Nhanh & Tự nhiên)",
                "description": "Làm mờ viền mềm chỉ khi có phụ đề, đè phụ đề mới (3–5s)",
                "icon": "⚡",
            },
            "clean": {
                "id": "clean",
                "name": "AI Clean Plate (Tẩy sạch chữ hoàn toàn)",
                "description": "Xóa sạch phụ đề tiếng Trung, giữ video nguyên bản không tì vết",
                "icon": "🧹",
            },
            "inpaint_burn": {
                "id": "inpaint_burn",
                "name": "Inpaint & Re-burn (Xóa sạch rồi đè sub mới)",
                "description": "Tẩy sạch chữ cũ trước, sau đó chèn phụ đề tiếng Việt mới lên",
                "icon": "🌟",
            },
        },
        "default": "blur",
    })


@api_bp.route("/api/translation-modes")
def get_translation_modes():
    """Get available translation modes/genres"""
    return jsonify({
        "modes": TRANSLATION_MODES,
        "default": DEFAULT_TRANSLATION_MODE,
    })


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


# ── Standalone SRT to TTS Engine ──────────────────────────────────────────

def _srt_to_tts_worker(job_id: str, srt_content: str, lang: str, engine: str, voice: str | None, options: dict):
    try:
        process_srt_to_tts(
            srt_content=srt_content,
            lang=lang,
            engine=engine,
            voice=voice,
            options=options,
            job_id=job_id,
        )
    except Exception as exc:
        print(f"[SRT_TO_TTS Worker] Lỗi job {job_id}: {exc}")
        if job_id in jobs:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = f"Lỗi tạo TTS từ phụ đề: {exc}"
            tts_key = f"tts_{lang}"
            jobs[job_id][tts_key] = {"status": "error", "progress": 0, "message": str(exc)}


@api_bp.route("/api/srt-to-tts", methods=["POST"])
def srt_to_tts_endpoint():
    srt_content = ""
    lang = "vi"
    engine = "edge"
    voice = None
    options = {}

    if request.content_type and "multipart/form-data" in request.content_type:
        file = request.files.get("file")
        if file and file.filename:
            try:
                srt_content = file.read().decode("utf-8", errors="replace")
            except Exception as e:
                return jsonify({"error": f"Không thể đọc file SRT: {e}"}), 400
        else:
            srt_content = request.form.get("srt_content", "")

        lang = request.form.get("lang", "vi")
        engine = request.form.get("engine", "edge")
        voice = request.form.get("voice") or None
        options = {
            "align_mode": request.form.get("align_mode", "smart_sync"),
            "safety_margin_ms": request.form.get("safety_margin_ms", 60),
            "max_speed_ratio": request.form.get("max_speed_ratio", 1.65),
            "target_duration_ms": request.form.get("target_duration_ms") or None,
        }
    else:
        data = request.get_json(silent=True) or {}
        srt_content = data.get("srt_content", "")
        lang = data.get("lang", "vi")
        engine = data.get("engine", "edge")
        voice = data.get("voice") or None
        options = {
            "align_mode": data.get("align_mode", "smart_sync"),
            "safety_margin_ms": data.get("safety_margin_ms", 60),
            "max_speed_ratio": data.get("max_speed_ratio", 1.65),
            "target_duration_ms": data.get("target_duration_ms") or None,
            "speaker_voices": data.get("speaker_voices"),
            "style_prompt": data.get("style_prompt"),
        }

    options = validate_tts_options(options)
    if engine not in {'edge', 'gemini', 'omnivoice'}:
        raise ValueError('Engine TTS không được hỗ trợ')
    if not isinstance(srt_content, str) or not srt_content.strip():
        return jsonify({"error": "Vui lòng tải lên file .srt hoặc nhập nội dung phụ đề"}), 400

    try:
        segments = parse_srt_data(srt_content)
    except Exception as exc:
        return jsonify({"error": f"Lỗi phân tích file SRT: {exc}"}), 400

    if not segments:
        return jsonify({"error": "Nội dung phụ đề .srt không hợp lệ hoặc không có dòng nào"}), 400

    job_id = f"srttts_{uuid.uuid4().hex[:10]}"
    sub_duration_s = round(segments[-1]["end_ms"] / 1000.0, 2)

    create_job(
        job_id,
        status="generating",
        progress=5,
        message=f"Đang chuẩn bị tạo thuyết minh cho {len(segments)} câu phụ đề...",
        job_type="srt_to_tts",
        lang=lang,
        engine=engine,
        total_segments=len(segments),
        sub_duration_s=sub_duration_s,
    )

    thread = threading.Thread(
        target=_srt_to_tts_worker,
        args=(job_id, srt_content, lang, engine, voice, options),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "started",
        "job_id": job_id,
        "message": f"Bắt đầu tạo thuyết minh ({len(segments)} câu, thời lượng phụ đề {sub_duration_s}s)...",
        "total_segments": len(segments),
        "sub_duration_s": sub_duration_s,
    })


@api_bp.route("/api/srt-to-tts/download/<job_id>/<file_type>")
def download_srt_to_tts_file(job_id, file_type):
    job = get_job(job_id)
    out_dir = OUTPUT_FOLDER / job_id
    if not out_dir.exists():
        return jsonify({"error": "Không tìm thấy thư mục kết quả cho job này"}), 404

    lang = job.get("lang", "vi") if job else "vi"

    if file_type == "audio":
        audio_path = out_dir / f"tts_aligned_{lang}.mp3"
        if not audio_path.exists():
            mp3s = list(out_dir.glob("*.mp3"))
            if mp3s:
                audio_path = mp3s[0]
            else:
                return jsonify({"error": "File audio chưa sẵn sàng"}), 404
        return send_file(
            audio_path,
            as_attachment=True,
            download_name=f"tts_{lang}_{job_id}.mp3",
            mimetype="audio/mpeg",
        )
    elif file_type == "srt":
        srt_path = out_dir / f"subtitles_aligned_{lang}.srt"
        if not srt_path.exists():
            srts = list(out_dir.glob("*aligned*.srt"))
            if srts:
                srt_path = srts[0]
            else:
                return jsonify({"error": "File SRT đồng bộ chưa sẵn sàng"}), 404
        return send_file(
            srt_path,
            as_attachment=True,
            download_name=f"aligned_{lang}_{job_id}.srt",
            mimetype="text/plain; charset=utf-8",
        )
    else:
        return jsonify({"error": "file_type không hợp lệ (hỗ trợ 'audio' hoặc 'srt')"}), 400


@api_bp.route("/api/srt-to-tts/voices")
def get_srt_to_tts_voices():
    return jsonify({
        "edge": SPEAKER_VOICE_MAPS.get("edge", {}),
        "gemini": SPEAKER_VOICE_MAPS.get("gemini", {}),
        "defaults": TTS_VOICES,
    })



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

    render_mode = str(data.get("render_mode", "blur")).lower().strip()
    if render_mode not in ("blur", "clean", "inpaint_burn"):
        render_mode = "blur"
    inpaint_engine = str(data.get("inpaint_engine", "opencv")).lower().strip()
    if inpaint_engine not in ("opencv", "lama"):
        inpaint_engine = "opencv"

    trim_intro = str(data.get("trim_intro", "auto")).lower().strip()
    translate_title = parse_bool(data.get("translate_title"), default=False)
    title_lang = str(data.get("title_lang", lang))
    brand_name = str(data.get("brand_name", "")).strip()
    bgm_mode = str(data.get("bgm_mode", "auto")).lower().strip()
    try:
        bgm_volume = float(data.get("bgm_volume", 0.8))
    except (ValueError, TypeError):
        bgm_volume = 0.8

    clean_hardsub = parse_bool(data.get("clean_hardsub"), default=True)
    clean_logo = parse_bool(data.get("clean_logo"), default=False)
    clean_title = parse_bool(data.get("clean_title"), default=False)
    burn_new_sub = parse_bool(data.get("burn_new_sub"), default=True)
    if lang == "clean":
        burn_new_sub = False

    keep_original_audio = parse_bool(data.get("keep_original_audio"), default=False)
    if bgm_mode in ("keep_original", "original", "orig_only", "none_tts") or lang == "clean":
        keep_original_audio = True

    thread = threading.Thread(
        target=burnsub_worker,
        args=(
            job_id, lang, srt_content, sub_region, extra_regions,
            render_mode, inpaint_engine, trim_intro, translate_title,
            title_lang, brand_name, bgm_mode, bgm_volume,
            clean_hardsub, clean_logo, clean_title, burn_new_sub,
            keep_original_audio
        ),
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
    job = get_job(job_id)
    video_path = ""
    download_filename = f"video_clean.mp4" if lang == "clean" else f"video_{lang}_sub.mp4"

    if job:
        burn_key = f"burn_{lang}"
        burn_info = job.get(burn_key, {})
        video_path = burn_info.get("path", "")
        download_filename = burn_info.get("filename", download_filename)

    # Disk fallback if in-memory job state is lost or not done
    if not video_path or not os.path.exists(video_path):
        candidates = [
            OUTPUT_FOLDER / job_id / f"burned_{lang}.mp4",
            OUTPUT_FOLDER / f"{job_id}_burned_{lang}.mp4",
            OUTPUT_FOLDER / job_id / f"{lang}_sub.mp4",
        ]
        output_dir = OUTPUT_FOLDER / job_id
        if output_dir.exists():
            candidates.extend(output_dir.glob(f"*{lang}*.mp4"))
            candidates.extend(output_dir.glob("*clean*.mp4"))
        for cand in candidates:
            if cand.exists() and cand.stat().st_size > 0:
                video_path = str(cand)
                download_filename = cand.name
                break

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "File video đã burn không tồn tại"}), 404

    return send_file(
        video_path,
        as_attachment=True,
        download_name=download_filename,
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

    if not gemini_api_key and gemini_model not in ("gemini-3.8-flash-high", "gemini-3.7-flash-high"):
        return jsonify({"error": "Chưa có Gemini API Key"}), 400

    translate_langs = _parse_languages(request.form.get("translate_langs", "[]"))
    ai_model = request.form.get("ai_model", AI_DEFAULT_MODEL)
    if ai_model not in AI_TRANSLATE_MODELS:
        ai_model = AI_DEFAULT_MODEL
    translate_method = request.form.get("translate_method", "google")
    if translate_method not in ("ai", "google"):
        translate_method = "google"
    if gemini_model not in GEMINI_MODELS:
        gemini_model = GEMINI_DEFAULT_MODEL

    translation_mode = request.form.get("translation_mode", DEFAULT_TRANSLATION_MODE)
    if translation_mode not in TRANSLATION_MODES:
        translation_mode = DEFAULT_TRANSLATION_MODE

    trim_intro = request.form.get("trim_intro", "auto")
    create_job(
        job_id,
        original_name=str(video.filename),
        video_path=str(video_path),
        video_file=_video_info(str(video_path)),
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        visual_timing_enabled=parse_bool(request.form.get('visual_timing'), default=False),
        translate_langs=translate_langs,
        translate_method=translate_method,
        translation_mode=translation_mode,
        ai_model=ai_model,
        mode="hardsub",
        trim_intro=trim_intro,
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
    from services.downloader import clean_download_url
    url = clean_download_url(data.get("url", ""))
    valid_url, url_error = validate_download_url(url)
    if not valid_url:
        return jsonify({"error": url_error}), 400

    gemini_model = data.get("gemini_model", GEMINI_DEFAULT_MODEL)
    gemini_api_key = data.get("gemini_api_key", "").strip()

    if gemini_api_key:
        set_gemini_api_key(gemini_api_key)
    else:
        gemini_api_key = get_gemini_api_key()

    if not gemini_api_key and gemini_model not in ("gemini-3.8-flash-high", "gemini-3.7-flash-high"):
        return jsonify({"error": "Chưa có Gemini API Key"}), 400

    translate_langs = data.get("translate_langs", [])
    translate_langs = [l for l in translate_langs if l in LANGUAGES]

    ai_model = data.get("ai_model", AI_DEFAULT_MODEL)
    if ai_model not in AI_TRANSLATE_MODELS:
        ai_model = AI_DEFAULT_MODEL
    translate_method = data.get("translate_method", "google")
    if translate_method not in ("ai", "google"):
        translate_method = "google"

    job_id = uuid.uuid4().hex[:12]

    if gemini_model not in GEMINI_MODELS:
        gemini_model = GEMINI_DEFAULT_MODEL
    translation_mode = data.get("translation_mode", DEFAULT_TRANSLATION_MODE)
    if translation_mode not in TRANSLATION_MODES:
        translation_mode = DEFAULT_TRANSLATION_MODE

    create_job(
        job_id,
        status="downloading_video",
        message="Đang tải video từ URL...",
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        visual_timing_enabled=parse_bool(data.get('visual_timing'), default=False),
        translate_langs=_parse_languages(data.get("translate_langs", [])),
        translate_method=translate_method,
        translation_mode=translation_mode,
        ai_model=ai_model,
        mode="hardsub",
        trim_intro=data.get("trim_intro", "auto"),
    )

    cleanup_old_jobs()

    jobs[job_id]["auto_burn"] = parse_bool(data.get("auto_burn"), default=False)
    jobs[job_id]["burn_region_mode"] = str(data.get("burn_region_mode", "auto"))
    sub_region = data.get("sub_region")
    if sub_region and validate_region(sub_region):
        jobs[job_id]["sub_region"] = sub_region
    jobs[job_id]["render_mode"] = str(data.get("render_mode", "inpaint_burn"))
    jobs[job_id]["keep_original_audio"] = parse_bool(data.get("keep_original_audio"), default=True)

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







# ── Douyin Channel Monitor Routes ──────────────────────────────────────────

@api_bp.route("/api/douyin/channels", methods=["GET"])
def get_douyin_channels():
    from services.douyin_monitor import get_channels
    return jsonify({"channels": get_channels()})


@api_bp.route("/api/douyin/channels", methods=["POST"])
def add_douyin_channel():
    from services.douyin_monitor import add_channel, resolve_channel_sec_uid

    data = request.get_json() or {}
    channel_id = str(data.get("channel_id", "")).strip()
    if not channel_id:
        return jsonify({"error": "Channel ID không được để trống"}), 400

    nickname = str(data.get("nickname", "")).strip()
    sec_uid = str(data.get("sec_uid", "")).strip()
    target_lang = str(data.get("target_lang", "vi")).strip()
    style = str(data.get("style", "driving")).strip()
    bgm_mode = str(data.get("bgm_mode", "ai")).strip()
    auto_burn = parse_bool(data.get("auto_burn"), default=True)
    clean_hardsub = parse_bool(data.get("clean_hardsub"), default=True)
    clean_logo = parse_bool(data.get("clean_logo"), default=True)
    translate_title = parse_bool(data.get("translate_title"), default=True)
    audio_policy = str(data.get("audio_policy", "keep_original")).strip()
    tts_enabled = parse_bool(data.get("tts_enabled"), default=False)
    subtitle_cleanup_mode = str(data.get("subtitle_cleanup_mode", "inpaint_burn")).strip()
    target_profile_id = str(data.get("target_profile_id", "")).strip()
    approval_policy = str(data.get("approval_policy", "manual")).strip()
    rights_status = str(data.get("rights_status", "review_required")).strip()
    publish_outbox_dir = str(data.get("publish_outbox_dir", "")).strip()

    if not sec_uid or sec_uid.startswith("MS4wLjABAAAA_rP") or len(sec_uid) < 30:
        try:
            resolved_uid, resolved_nick, meta = resolve_channel_sec_uid(channel_id)
            if resolved_uid:
                sec_uid = resolved_uid
                if not nickname:
                    nickname = resolved_nick or meta.get("nickname", channel_id)
                if meta.get("unique_id"):
                    channel_id = meta.get("unique_id")
        except Exception as exc:
            print(f"[API] Error resolving channel sec_uid: {exc}")

    ch = add_channel(
        channel_id=channel_id,
        nickname=nickname,
        sec_uid=sec_uid,
        target_lang=target_lang,
        style=style,
        bgm_mode=bgm_mode,
        auto_burn=auto_burn,
        clean_hardsub=clean_hardsub,
        clean_logo=clean_logo,
        translate_title=translate_title,
        audio_policy=audio_policy,
        tts_enabled=tts_enabled,
        subtitle_cleanup_mode=subtitle_cleanup_mode,
        target_profile_id=target_profile_id,
        approval_policy=approval_policy,
        rights_status=rights_status,
        publish_outbox_dir=publish_outbox_dir,
    )
    return jsonify({"status": "success", "channel": ch})


@api_bp.route("/api/douyin/channels/<channel_id>", methods=["DELETE"])
def delete_douyin_channel(channel_id):
    from services.douyin_monitor import remove_channel
    success = remove_channel(channel_id)
    return jsonify({"success": success})


@api_bp.route("/api/douyin/channels/<channel_id>/toggle", methods=["POST"])
def toggle_douyin_channel(channel_id):
    from services.douyin_monitor import toggle_channel
    data = request.get_json() or {}
    enabled = parse_bool(data.get("enabled"), default=True)
    success = toggle_channel(channel_id, enabled)
    return jsonify({"success": success})


@api_bp.route("/api/douyin/resolve", methods=["POST"])
def resolve_douyin_channel():
    from services.douyin_monitor import resolve_channel_sec_uid
    data = request.get_json() or {}
    channel_input = str(data.get("channel_input", "")).strip()
    if not channel_input:
        return jsonify({"error": "Vui lòng nhập Douyin ID hoặc link kênh"}), 400

    sec_uid, nickname, meta = resolve_channel_sec_uid(channel_input)
    if not sec_uid:
        return jsonify({"error": "Không tìm thấy thông tin kênh"}), 404

    return jsonify({
        "sec_uid": sec_uid,
        "nickname": nickname,
        "meta": meta,
    })


@api_bp.route("/api/douyin/monitor/status", methods=["GET"])
def get_douyin_monitor_status():
    from services.douyin_monitor import get_monitor_status, get_channels, get_downloaded_history
    status = get_monitor_status()
    channels = get_channels()
    history = get_downloaded_history()
    status["total_channels"] = len(channels)
    status["enabled_channels"] = len([c for c in channels if c.get("enabled")])
    status["downloaded_count"] = len(history)
    return jsonify(status)


@api_bp.route("/api/douyin/monitor/toggle", methods=["POST"])
def toggle_douyin_monitor():
    from services.douyin_monitor import start_monitor, stop_monitor, get_monitor_status
    data = request.get_json() or {}
    enable = parse_bool(data.get("enable"), default=True)
    interval = int(data.get("interval", 180))

    if enable:
        start_monitor(interval=interval)
    else:
        stop_monitor()

    return jsonify(get_monitor_status())


@api_bp.route("/api/douyin/monitor/scan", methods=["POST"])
def scan_douyin_monitor():
    from services.douyin_monitor import scan_now
    res = scan_now()
    return jsonify(res)

# =========================================================================
# UNIFIED STUDIO PIPELINE ENDPOINTS
# =========================================================================

@api_bp.route("/api/pipeline/presets", methods=["GET"])
def get_pipeline_presets():
    """Tra ve cac kich ban mau (Full Dub, Pure Vietsub, Editor Kit, Custom)."""
    return jsonify({
        "full_dub": {
            "name": "🎬 1. Trọn gói Lồng tiếng & In Sub (Full Dub)",
            "desc": "Tải video Douyin -> Nhận diện & Dịch -> Tạo thuyết minh TTS -> In sub kèm Ducking hạ nhỏ tiếng gốc.",
            "steps": {"download": True, "extract_sub": True, "clean_video": False, "tts": True, "burn_sub": True},
            "sub_options": {"engine": "hybrid", "model": "gemini-3.8-flash-high", "target_lang": "vi", "style": "movie", "pause_for_review": False},
            "clean_options": {"region_mode": "auto", "engine": "opencv", "clean_hardsub": True},
            "tts_options": {"engine": "edge", "voice": "vi-VN-NamMinhNeural", "align_mode": "smart_sync", "margin": 60, "max_speed": 1.45},
            "burn_options": {"audio_mode": "tts_ducking", "render_mode": "inpaint_burn", "bgm_volume": 0.8}
        },
        "pure_vietsub": {
            "name": "🇻🇳 2. Chỉ làm Vietsub (Giữ 100% âm thanh gốc)",
            "desc": "Tải video Douyin -> Nhận diện & Dịch -> In sub mới đè lên sub cũ, giữ nguyên toàn bộ âm thanh gốc.",
            "steps": {"download": True, "extract_sub": True, "clean_video": False, "tts": False, "burn_sub": True},
            "sub_options": {"engine": "hybrid", "model": "gemini-3.8-flash-high", "target_lang": "vi", "style": "movie", "pause_for_review": False},
            "clean_options": {"region_mode": "auto", "engine": "opencv", "clean_hardsub": True},
            "tts_options": {},
            "burn_options": {"audio_mode": "keep_original", "render_mode": "inpaint_burn", "bgm_volume": 1.0}
        },
        "editor_kit": {
            "name": "🧹 3. Bộ dựng tự do (Xuất Video sạch chữ & File .SRT)",
            "desc": "Tải video Douyin -> Nhận diện & Dịch ra file .SRT -> Xóa sạch chữ trên video (Clean Plate) để tự dựng CapCut/Premiere.",
            "steps": {"download": True, "extract_sub": True, "clean_video": True, "tts": False, "burn_sub": False},
            "sub_options": {"engine": "hybrid", "model": "gemini-3.8-flash-high", "target_lang": "vi", "style": "movie", "pause_for_review": False},
            "clean_options": {"region_mode": "auto", "engine": "opencv", "clean_hardsub": True},
            "tts_options": {},
            "burn_options": {}
        }
    })


@api_bp.route("/api/jobs", methods=["GET"])
def get_all_jobs():
    """Liệt kê danh sách tất cả các jobs đang có trên server."""
    active = {}
    for jid, j in jobs.snapshots().items():
        if j is None:
            continue
        active[jid] = {
            "job_id": jid,
            "status": j.get("status"),
            "progress": j.get("progress", 0),
            "message": j.get("message", ""),
            "active_step": j.get("active_step", ""),
            "original_name": j.get("original_name", ""),
            "created_at": j.get("_created_at", 0),
        }
    return jsonify(active)


@api_bp.route("/api/batch/region-preview", methods=["POST"])
def api_batch_region_preview():
    """Detect or preview the shared subtitle region on the first batch video."""
    data = request.get_json(silent=True) or {}
    folder_path = str(data.get("folder_path") or data.get("folder") or "").strip()
    if not folder_path:
        return jsonify({"error": "Chưa cung cấp folder_path"}), 400

    mode = str(data.get("region_mode") or "auto").strip().lower()
    option_payload = {
        "region_mode": mode,
        "sub_region": data.get("sub_region"),
        "output_mode": "srt_and_video",
    }
    region_options = validate_batch_options(option_payload)
    files = scan_video_folder(
        folder_path,
        recursive=parse_bool(data.get("recursive"), True),
        extensions=data.get("extensions"),
        allowed_roots=BATCH_ALLOWED_ROOTS,
    )
    if not files:
        return jsonify({"error": "Không tìm thấy video hợp lệ trong thư mục"}), 400
    video_path = files[0]

    with host_lock("batch-region-preview", timeout=600):
        if mode == "auto":
            from services.burn_sub import detect_hardsub_region
            detected = detect_hardsub_region(str(video_path), job_id=None, srt_content=None)
            region_method = str(detected.get("method") or "auto")
            region = validate_batch_options({
                "region_mode": "manual",
                "sub_region": detected,
                "output_mode": "srt_and_video",
            })["sub_region"]
        else:
            region = region_options["sub_region"]
            region_method = "manual"

        import cv2
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError("Không đọc được video để tạo preview")
            total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            target_frame = min(total_frames - 1, max(0, int(total_frames * 0.25)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError("Không trích xuất được frame preview")
        finally:
            cap.release()

        height, width = frame.shape[:2]
        x = max(0, min(width - 1, int(width * region["x_ratio"])))
        y = max(0, min(height - 1, int(height * region["y_ratio"])))
        w = max(2, min(width - x, int(width * region["w_ratio"])))
        h = max(2, min(height - y, int(height * region["h_ratio"])))
        color = (40, 220, 40) if mode == "auto" else (0, 180, 255)
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), color, 3)
        cv2.putText(
            frame,
            f"Subtitle region: {region_method}",
            (max(4, x), max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        preview_id = uuid.uuid4().hex[:20]
        preview_dir = OUTPUT_FOLDER / "batch_region_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{preview_id}.jpg"
        if not cv2.imwrite(str(preview_path), frame):
            raise RuntimeError("Không ghi được ảnh preview")

    return jsonify({
        "video_name": video_path.name,
        "video_path": str(video_path),
        "region_mode": mode,
        "region_method": region_method,
        "region": region,
        "preview_url": f"/api/batch/region-preview/file/{preview_id}",
    })


@api_bp.route("/api/batch/region-preview/file/<preview_id>", methods=["GET"])
def api_batch_region_preview_file(preview_id):
    if not re.fullmatch(r"[a-f0-9]{20}", str(preview_id or "")):
        return jsonify({"error": "preview_id không hợp lệ"}), 400
    preview_path = OUTPUT_FOLDER / "batch_region_previews" / f"{preview_id}.jpg"
    if not preview_path.is_file():
        return jsonify({"error": "Preview không tồn tại"}), 404
    return send_file(preview_path, mimetype="image/jpeg", max_age=0)


@api_bp.route("/api/batch/run", methods=["POST"])
def api_batch_run():
    """Start a resumable translation/render batch for a local video folder."""
    data = request.get_json(silent=True) or {}
    folder_path = str(data.get("folder_path") or data.get("folder") or "").strip()
    if not folder_path:
        return jsonify({"error": "Chưa cung cấp folder_path"}), 400

    options = validate_batch_options(data)
    files = scan_video_folder(
        folder_path,
        recursive=options["recursive"],
        extensions=options["extensions"],
        allowed_roots=BATCH_ALLOWED_ROOTS,
    )
    if not files:
        return jsonify({"error": "Không tìm thấy video hợp lệ trong thư mục"}), 400
    if len(files) > BATCH_MAX_FILES:
        return jsonify({"error": f"Batch vượt giới hạn {BATCH_MAX_FILES} video"}), 400

    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    batch_dir = OUTPUT_FOLDER / "batches" / batch_id
    manifest_path, manifest = create_batch_manifest(
        batch_id,
        Path(folder_path).expanduser().resolve(),
        files,
        batch_dir,
        options,
    )
    create_job(
        batch_id,
        mode="batch",
        original_name=Path(folder_path).name,
        batch_id=batch_id,
        batch_folder=str(Path(folder_path).expanduser().resolve()),
        batch_manifest_path=str(manifest_path),
        status="queued",
        progress=0,
        batch_counts=manifest["counts"],
    )
    try:
        submit(run_batch, batch_id, str(manifest_path), options)
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error"] = str(exc)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"error": str(exc), "batch_id": batch_id}), 409
    return jsonify({
        "batch_id": batch_id,
        "status": "queued",
        "total_files": len(files),
        "manifest_path": str(manifest_path),
    }), 202


@api_bp.route("/api/batch/<batch_id>", methods=["GET"])
def api_batch_status(batch_id):
    job = get_job(batch_id)
    manifest_path = (job or {}).get("batch_manifest_path")
    if not manifest_path:
        manifest_path = str(OUTPUT_FOLDER / "batches" / batch_id / "manifest.json")
    try:
        manifest = recover_stale_batch(manifest_path, job)
    except FileNotFoundError:
        return jsonify({"error": "Không tìm thấy batch"}), 404
    return jsonify(manifest)


@api_bp.route("/api/batch/<batch_id>/stop", methods=["POST"])
def api_batch_stop(batch_id):
    job = get_job(batch_id)
    manifest_path = (job or {}).get("batch_manifest_path")
    if not manifest_path:
        manifest_path = str(OUTPUT_FOLDER / "batches" / batch_id / "manifest.json")
    try:
        manifest = BatchRunner(batch_id, manifest_path, {}).request_cancel()
    except FileNotFoundError:
        return jsonify({"error": "Không tìm thấy batch"}), 404
    return jsonify({"batch_id": batch_id, "status": manifest.get("status"), "cancel_requested": True})


def _batch_manifest_for_route(batch_id: str):
    job = get_job(batch_id)
    manifest_path = (job or {}).get("batch_manifest_path")
    if not manifest_path:
        manifest_path = str(OUTPUT_FOLDER / "batches" / batch_id / "manifest.json")
    return manifest_path, recover_stale_batch(manifest_path, job)


@api_bp.route("/api/batch/<batch_id>/resume", methods=["POST"])
def api_batch_resume(batch_id):
    try:
        manifest_path, manifest = _batch_manifest_for_route(batch_id)
    except FileNotFoundError:
        return jsonify({"error": "Không tìm thấy batch"}), 404
    if manifest.get("status") == "processing":
        return jsonify({"error": "Batch đang chạy"}), 409
    manifest["cancel_requested"] = False
    for item in manifest.get("items", []):
        if item.get("status") in {"failed", "partial", "cancelled", "error"}:
            item.update(status="queued", stage="queued", progress=0, error=None)
    Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    options = validate_batch_options(manifest.get("options") or {})
    if not get_job(batch_id):
        create_job(batch_id, mode="batch", batch_manifest_path=str(manifest_path), status="queued")
    else:
        get_job(batch_id)["cancel"] = False
    submit(run_batch, batch_id, manifest_path, options)
    return jsonify({"batch_id": batch_id, "status": "queued"}), 202


@api_bp.route("/api/batch/<batch_id>/items/<item_id>/retry", methods=["POST"])
def api_batch_item_retry(batch_id, item_id):
    try:
        manifest_path, manifest = _batch_manifest_for_route(batch_id)
    except FileNotFoundError:
        return jsonify({"error": "Không tìm thấy batch"}), 404
    item = next((value for value in manifest.get("items", []) if value.get("item_id") == item_id), None)
    if not item:
        return jsonify({"error": "Không tìm thấy item"}), 404
    if manifest.get("status") == "processing":
        return jsonify({"error": "Batch đang chạy"}), 409
    item.update(status="queued", stage="queued", progress=0, error=None)
    manifest["cancel_requested"] = False
    Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    options = validate_batch_options(manifest.get("options") or {})
    if not get_job(batch_id):
        create_job(batch_id, mode="batch", batch_manifest_path=str(manifest_path), status="queued")
    else:
        get_job(batch_id)["cancel"] = False
    submit(run_batch, batch_id, manifest_path, options)
    return jsonify({"batch_id": batch_id, "item_id": item_id, "status": "queued"}), 202


@api_bp.route("/api/batch/<batch_id>/manifest", methods=["GET"])
def api_batch_manifest_download(batch_id):
    try:
        manifest_path, _manifest = _batch_manifest_for_route(batch_id)
    except FileNotFoundError:
        return jsonify({"error": "Không tìm thấy batch"}), 404
    return send_file(manifest_path, as_attachment=True, download_name=f"{batch_id}-manifest.json")


@api_bp.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    """Khởi động quy trình tổng hợp với checklist các công đoạn tùy chọn."""
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        raw_config = request.form.get("config") or request.form.get("payload")
        if raw_config:
            try:
                data = json.loads(raw_config)
            except Exception:
                data = request.form.to_dict()
        else:
            data = request.form.to_dict()

    raw_jid = str(data.get("job_id") or "").strip()
    if raw_jid:
        if not re.match(r"^[A-Za-z0-9_-]{6,64}$", raw_jid):
            return jsonify({"error": "Mã job_id không hợp lệ (chỉ chấp nhận ký tự a-z, 0-9, _, -)"}), 400
        job_id = raw_jid
    else:
        job_id = f"pipe_{uuid.uuid4().hex[:10]}"
    data["job_id"] = job_id

    # Kiểm tra bảo mật nếu client truyền đường dẫn video_path
    if data.get("video_path"):
        raw_vpath = Path(str(data["video_path"])).resolve()
        upload_root = UPLOAD_FOLDER.resolve()
        output_root = OUTPUT_FOLDER.resolve()
        if not (upload_root in raw_vpath.parents or output_root in raw_vpath.parents):
            return jsonify({"error": "Đường dẫn video không hợp lệ (phải nằm trong uploads/ hoặc outputs/)"}), 400
        if not raw_vpath.exists():
            return jsonify({"error": "File video không tồn tại trên máy chủ"}), 404
        data["video_path"] = str(raw_vpath)

    # Xử lý nếu có file upload trực tiếp qua multipart/form-data
    if "file" in request.files:
        f = request.files["file"]
        if f and f.filename:
            existing = get_job(job_id)
            if existing and existing.get('status') not in {'uploaded','done','partial','error','cancelled','interrupted'}:
                return jsonify({'error': 'Job đang hoạt động'}), 409
            ext, error = _validate_uploaded_video(f)
            if error:
                raise ValueError(error)
            dest_dir = UPLOAD_FOLDER / job_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            vpath = dest_dir / ('source' + ext)
            f.save(str(vpath))
            valid, error = _validate_media_path(vpath)
            if not valid:
                vpath.unlink(missing_ok=True)
                raise ValueError(error)
            data["video_path"] = str(vpath)
            data["source_type"] = "upload"
            data["original_name"] = f.filename

    # Khôi phục video_path từ job đã upload trước đó nếu có
    if not data.get("video_path") and job_id:
        existing_job = get_job(job_id)
        if existing_job and existing_job.get("video_path"):
            data["video_path"] = existing_job["video_path"]
        else:
            cand = UPLOAD_FOLDER / job_id / "source.mp4"
            if cand.exists():
                data["video_path"] = str(cand)

    from services.pipeline_orchestrator import start_pipeline_job
    try:
        res = start_pipeline_job(job_id, data)
        return jsonify(res)
    except TimeoutError as exc:
        return jsonify({'error': str(exc)}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.route("/api/pipeline/continue/<job_id>", methods=["POST"])
def api_pipeline_continue(job_id):
    """Tiếp tục quy trình sau khi người dùng đã xem trước/sửa file phụ đề .srt."""
    job_id = str(job_id or "").strip()
    if not re.match(r"^[A-Za-z0-9_-]{6,64}$", job_id):
        return jsonify({"error": "Mã job_id không hợp lệ"}), 400

    data = request.get_json(silent=True) or {}
    updated_srt = str(data.get("updated_srt") or "").strip()
    if updated_srt:
        from services.srt_utils import validate_srt
        valid_srt, srt_errs = validate_srt(updated_srt)
        if not valid_srt:
            return jsonify({"error": f"Phụ đề chỉnh sửa không hợp lệ: {'; '.join(srt_errs[:2])}"}), 400

    from services.pipeline_orchestrator import resume_pipeline_job
    try:
        res = resume_pipeline_job(job_id, updated_srt, expected_revision=data.get('revision'))
        return jsonify(res)
    except TimeoutError as exc:
        return jsonify({'error': str(exc)}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.route('/api/pipeline/review/<job_id>', methods=['GET', 'PUT'])
def pipeline_review(job_id):
    with host_lock('review-' + job_id, timeout=5):
        job = get_job(job_id)
        if not job or job.get('status') != 'awaiting_review' or not job.get('pending_pipeline'):
            return jsonify({'error': 'Job không ở bước chờ duyệt'}), 409
        if request.method == 'PUT':
            data = request.get_json(silent=True) or {}
            if data.get('revision') != job.get('review_revision', 0):
                return jsonify({'error': 'Bản phụ đề đã được cập nhật ở nơi khác'}), 409
            content = data.get('content')
            if not isinstance(content, str) or len(content) > 10 * 1024 * 1024:
                raise ValueError('Nội dung bản nháp không hợp lệ')
            job.update(review_draft=content, review_revision=job.get('review_revision', 0)+1)
            job.persist()
        pending = job['pending_pipeline']
        return jsonify({'content': job.get('review_draft', pending['target_srt_content']),
                        'revision': job.get('review_revision', 0), 'lang': pending['target_lang']})


@api_bp.route('/api/pipeline/retry/<job_id>', methods=['POST'])
def retry_pipeline(job_id):
    from services.pipeline_orchestrator import start_pipeline_job
    job = get_job(job_id)
    if not job or not job.get('pipeline_config'):
        return jsonify({'error': 'Không tìm thấy cấu hình Studio để thử lại'}), 404
    if job.get('status') not in {'error', 'partial', 'cancelled', 'interrupted'}:
        raise TimeoutError('Chỉ thử lại tác vụ đã dừng hoặc chưa hoàn tất')
    options = json.loads(json.dumps(job['pipeline_config']))
    data = request.get_json(silent=True) or {}
    lang = options.get('sub_options', {}).get('target_lang', 'vi')
    srt_info = job.get('srt_files', {}).get(lang, {})
    translation_incomplete = job.get('translation_quality', {}).get(lang, {}).get('unchanged_segments')
    if srt_info.get('path') and Path(srt_info['path']).is_file() and (not translation_incomplete or data.get('target_srt_content')):
        options['target_srt_content'] = data.get('target_srt_content') or Path(srt_info['path']).read_text(encoding='utf-8-sig')
        options.pop('target_srt_path', None)
        options.pop('srt_path', None)
        options['steps']['extract_sub'] = False
        original_timing = job.get('source_visual_srt')
        if not original_timing:
            original_info = job.get('srt_files', {}).get('zh', {})
            if original_info.get('path') and Path(original_info['path']).is_file():
                original_timing = Path(original_info['path']).read_text(encoding='utf-8-sig')
        if original_timing:
            options['zh_srt_content'] = original_timing
    if job.get('video_path') and Path(job['video_path']).is_file():
        options.update(video_path=job['video_path'], source_type='upload', trim_intro='off')
        options['steps']['download'] = False
        if job.get('clean_video', {}).get('path') == job['video_path']:
            options['steps']['clean_video'] = False
            options.setdefault('burn_options', {})['render_mode'] = 'pure_burn'
            if job.get('source_sub_region'):
                options['burn_options'].update(region_mode='manual', sub_region=job['source_sub_region'])
    if data.get('tts_options') is not None:
        if not isinstance(data['tts_options'], dict):
            raise ValueError('tts_options phải là object')
        options.setdefault('tts_options', {}).update(data['tts_options'])
    return jsonify(start_pipeline_job(job_id, options))


@api_bp.route('/api/upload/session/<job_id>')
def upload_session_status(job_id):
    with host_lock('upload-' + job_id, timeout=5):
        session = _load_chunk_session(job_id)
        if not session:
            return jsonify({'error': 'Phiên upload không tồn tại'}), 404
        return jsonify({k: sorted(v) if isinstance(v, set) else v for k, v in session.items()
                        if k in {'job_id','filename','total_size','chunk_size','total_chunks','received_chunks','finished_response'}})
