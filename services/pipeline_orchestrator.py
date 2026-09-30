# -*- coding: utf-8 -*-
"""
services/pipeline_orchestrator.py
==================================
Bộ điều phối quy trình thống nhất (Unified Studio Pipeline Orchestrator).
Cho phép kích hoạt linh hoạt 1 hoặc nhiều tính năng trong 5 công đoạn:
  - Bước 1: Tải video gốc Douyin (Master CDN, không watermark) / nhận video upload.
  - Bước 2: Nhận diện phụ đề (Gemini 3.8 Vision hoặc Whisper ASR) -> Dịch sang VI/EN.
    (Hỗ trợ tạm dừng xem trước/chỉnh sửa SRT trước khi chạy tiếp).
  - Bước 3: Xử lý phụ đề gốc trên video (Auto OCR / Preset -> Inpaint / Blur -> Clean Plate).
  - Bước 4: Chuyển đổi phụ đề .srt sang thuyết minh TTS (Edge-TTS / Gemini TTS, Smart Sync).
  - Bước 5: In hardsub đã dịch lên video (Giữ 100% âm thanh gốc hoặc lồng tiếng TTS + BGM).
"""

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from services.srt_utils import validate_srt
from services.runtime_state import host_lock, serial_task, valid_job_id
from services.pipeline_options import validate_pipeline_options
from services.job_runner import submit
from config import (
    AI_DEFAULT_MODEL,
    AI_TRANSLATE_CONFIG,
    DEFAULT_TRANSLATION_MODE,
    GEMINI_DEFAULT_MODEL,
    GEMINI_MODELS,
    LANGUAGES,
    OUTPUT_FOLDER,
    TTS_VOICES,
    UPLOAD_FOLDER,
    create_job,
    get_gemini_api_key,
    get_job,
    jobs,
)



def _extract_whisper_source(job_id: str, video_path: str, sub_opts: dict) -> tuple[str, str]:
    from services.whisper_engine import process_video
    model = sub_opts.get("whisper_model", "large-v3-turbo")
    process_video(job_id=job_id, video_path=video_path, model_size=model,
                  translate_langs=[], translate_method="ai",
                  translation_mode=sub_opts.get("style", "movie"), manage_status=False)
    current = get_job(job_id)
    if not current or current.get("status") == "error":
        raise RuntimeError((current or {}).get("message", "Whisper thất bại"))
    info = (current.get("srt_files") or {}).get("zh") or {}
    path = info.get("path", "")
    if not path or not Path(path).is_file():
        raise RuntimeError("Whisper không tạo được SRT tiếng Trung")
    return Path(path).read_text(encoding="utf-8-sig"), path


def _extract_gemini_source(job_id: str, video_path: str, sub_opts: dict) -> str:
    job = get_job(job_id)
    model = sub_opts.get("model", "gemini-3.8-flash-high")
    from services.hardsub_gemini import (create_gemini_proxy_video,
        extract_hardsub_via_local_gateway, parse_srt_from_text, HARDSUB_PROMPT)
    api_key = get_gemini_api_key()
    use_local = bool(AI_TRANSLATE_CONFIG.get("api_key"))
    if use_local or model in ("gemini-3.8-flash-high", "gemini-3.7-flash-high"):
        raw = extract_hardsub_via_local_gateway(video_path, job_id, model_name=model)
    else:
        from google import genai
        upload_path, is_proxy = create_gemini_proxy_video(video_path, job_id)
        client = genai.Client(api_key=api_key)
        uploaded = None
        try:
            uploaded = client.files.upload(file=upload_path)
            deadline = time.monotonic() + 600
            while uploaded.state.name == "PROCESSING":
                _check_cancel(job)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Gemini xử lý file quá thời gian cho phép")
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)
            if uploaded.state.name != "ACTIVE":
                raise RuntimeError("Gemini không thể xử lý video")
            response = client.models.generate_content(model=model, contents=[uploaded, HARDSUB_PROMPT])
            if any(str(getattr(c, "finish_reason", "")).split(".")[-1]
                   in {"MAX_TOKENS", "SAFETY", "RECITATION"} for c in (response.candidates or [])):
                raise RuntimeError("Gemini OCR bị cắt hoặc chặn")
            raw = response.text or ""
        finally:
            if uploaded:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass
            if is_proxy:
                Path(upload_path).unlink(missing_ok=True)
    content = parse_srt_from_text(raw)
    valid, errors = validate_srt(content)
    if not valid:
        raise RuntimeError("Gemini không trích xuất được SRT hợp lệ: " + "; ".join(errors[:2]))
    return content

def start_pipeline_job(job_id: str, config: Dict[str, Any]) -> dict:
    if not valid_job_id(job_id):
        raise ValueError('job_id không hợp lệ')
    with host_lock('submit-' + job_id, timeout=5):
        existing = get_job(job_id)
        if existing and existing.get('status') not in {'uploaded', 'interrupted', 'done', 'partial', 'error', 'cancelled'}:
            raise TimeoutError('Job đang chạy hoặc chờ duyệt; không thể khởi động trùng.')
        return _start_pipeline_job(job_id, validate_pipeline_options(config))


def _start_pipeline_job(job_id: str, config: Dict[str, Any]) -> dict:
    """Khởi tạo và chạy pipeline trong background thread."""
    job_id = str(job_id or "").strip()
    if not re.match(r"^[A-Za-z0-9_-]{6,64}$", job_id):
        raise ValueError(f"job_id '{job_id}' không hợp lệ (chỉ chấp nhận a-z, 0-9, _, -)")

    source_type = config.get("source_type", "upload")
    url = config.get("url", "").strip()
    original_name = config.get("original_name") or ("douyin_video" if "douyin" in url.lower() else "studio_video")

    initial_fields = {
        
        "source_type": source_type,
        "url": url,
        "original_name": original_name,
        "pipeline_config": config,
        "status": "queued",
        "progress": 2,
        "message": "Đang khởi tạo quy trình Studio...",
        "active_step": "init",
        "steps_selected": config.get("steps", {}),
        "srt_files": {},
        "artifacts": {},
        "step_results": {},
        "review_revision": 0,
        "trim_intro": "off",
        "ai_model": config.get('sub_options', {}).get('ai_model', AI_DEFAULT_MODEL),
    }

    if "video_path" in config and config["video_path"]:
        vpath = Path(config["video_path"]).resolve()
        upload_root = UPLOAD_FOLDER.resolve()
        output_root = OUTPUT_FOLDER.resolve()
        if not (upload_root in vpath.parents or output_root in vpath.parents):
            raise ValueError(f"Đường dẫn video '{vpath}' nằm ngoài phạm vi an toàn (uploads/ hoặc outputs/)")
        if vpath.exists():
            initial_fields["video_path"] = str(vpath)
            initial_fields["video_file"] = {
                "path": str(vpath),
                "filename": vpath.name,
                "size": vpath.stat().st_size,
            }

    create_job(job_id, **initial_fields)

    try:
        submit(_pipeline_worker, job_id, config)
    except Exception:
        jobs[job_id].update(status='error', message='Không thể đưa công việc vào hàng đợi')
        raise

    return {"job_id": job_id, "status": "queued", "message": "Pipeline đã bắt đầu chạy ngầm."}


def resume_pipeline_job(job_id: str, updated_srt: Optional[str] = None, expected_revision=None) -> dict:
    if not valid_job_id(job_id):
        raise ValueError('job_id không hợp lệ')
    with host_lock('review-' + job_id, timeout=5):
        job = get_job(job_id)
        if expected_revision is not None and job and expected_revision != job.get('review_revision', 0):
            raise TimeoutError('Bản phụ đề đã thay đổi. Hãy tải lại trước khi xác nhận.')
        return _resume_pipeline_job(job_id, updated_srt)


def _resume_pipeline_job(job_id: str, updated_srt: Optional[str] = None) -> dict:
    """Tiếp tục pipeline sau khi người dùng đã duyệt/sửa phụ đề."""
    job = get_job(job_id)
    if not job:
        raise RuntimeError(f"Không tìm thấy job: {job_id}")

    pending = job.get("pending_pipeline")
    if not pending or job.get('status') != 'awaiting_review':
        raise RuntimeError(f"Job {job_id} không ở trạng thái chờ duyệt (awaiting_review).")

    # Cập nhật lại nội dung SRT đã được người dùng chỉnh sửa
    target_lang = pending.get("target_lang", "vi")
    unresolved = list(job.get('translation_quality', {}).get(target_lang, {}).get('unchanged_segments', []))
    submitted_srt = (updated_srt or '').strip()
    if unresolved:
        from services.srt_utils import parse_srt
        before = parse_srt(pending.get('target_srt_content', ''))
        after = parse_srt(submitted_srt)
        still_untranslated = []
        for position in unresolved:
            offset = int(position) - 1
            if offset < 0 or offset >= len(after):
                still_untranslated.append(position)
                continue
            previous_text = before[offset][2].strip() if offset < len(before) else ''
            if after[offset][2].strip() == previous_text:
                still_untranslated.append(position)
        if still_untranslated:
            preview = ', '.join(map(str, still_untranslated[:12]))
            suffix = '…' if len(still_untranslated) > 12 else ''
            raise ValueError(f'Còn {len(still_untranslated)} câu chưa dịch (mục {preview}{suffix}). Hãy sửa các câu này trước khi tiếp tục.')
    if submitted_srt:
        valid_srt, srt_errs = validate_srt(submitted_srt)
        if not valid_srt:
            raise ValueError(f"Phụ đề chỉnh sửa không hợp lệ: {'; '.join(srt_errs[:2])}")
        output_dir = OUTPUT_FOLDER / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        srt_path = output_dir / f"hardsub_{target_lang}.srt"
        srt_path.write_text(submitted_srt, encoding="utf-8")
        if "srt_files" not in job:
            job["srt_files"] = {}
        meta = LANGUAGES.get(target_lang, {"name": target_lang, "flag": "🏳️"})
        job["srt_files"][target_lang] = {
            "filename": srt_path.name,
            "path": str(srt_path),
            "preview": submitted_srt[:500],
            "lang_name": meta.get("name", target_lang),
            "flag": meta.get("flag", "🏳️"),
        }
        pending["target_srt_content"] = submitted_srt
        if job.get("semantic_fusion"):
            from services.hybrid_subtitles import display_srt_to_tts_srt
            tts_srt, tts_speakers = display_srt_to_tts_srt(submitted_srt, job.get("segment_speakers"))
            job["tts_source_srt"] = tts_srt
            job["tts_segment_speakers"] = tts_speakers
            tts_path = output_dir / f"tts_script_{target_lang}.srt"
            tts_path.write_text(tts_srt, encoding="utf-8")
            job["tts_script"] = {"path": str(tts_path), "filename": tts_path.name,
                                 "segments": len(tts_speakers), "regenerated_after_review": True}
        if unresolved:
            quality = job.setdefault('translation_quality', {}).setdefault(target_lang, {})
            quality['unchanged_segments'] = []
            quality['reviewed_by_user'] = True
        pending["target_srt_path"] = str(srt_path)

    job["cancel"] = False
    job['_owner_pid'] = os.getpid()
    dict.__setitem__(jobs, job_id, job)
    job["review_revision"] = job.get('review_revision', 0) + 1
    job["status"] = "processing"
    job["message"] = "Đã xác nhận phụ đề! Đang tiếp tục các công đoạn tiếp theo..."
    job["pending_pipeline"] = None
    job.persist()

    try:
        submit(_resume_worker, job_id, dict(pending))
    except Exception:
        job.update(status='awaiting_review', pending_pipeline=pending)
        raise

    return {"job_id": job_id, "status": "processing", "message": "Đang tiếp tục các bước tiếp theo."}


@serial_task
def _pipeline_worker(job_id: str, config: Dict[str, Any]) -> None:
    """Thực thi tuần tự các bước 1 -> 5 theo cấu hình đã chọn."""
    job = get_job(job_id)
    if not job:
        return

    steps = config.get("steps", {
        "download": True,
        "extract_sub": True,
        "clean_video": False,
        "tts": True,
        "burn_sub": True,
    })

    sub_opts = config.get("sub_options", {})
    clean_opts = config.get("clean_options", {})
    tts_opts = config.get("tts_options", {})
    burn_opts = config.get("burn_options", {})

    target_lang = sub_opts.get("target_lang", "vi")
    if target_lang not in LANGUAGES:
        target_lang = "vi"

    output_dir = OUTPUT_FOLDER / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _check_cancel(job)
        job['status'] = 'processing'
        # =========================================================================
        # BƯỚC 1: XỬ LÝ NGUỒN VIDEO & TẢI DOUYIN
        # =========================================================================
        video_path = job.get("video_path")
        source_type = config.get("source_type", "upload")
        from services.downloader import clean_download_url
        url = clean_download_url(config.get("url", ""))

        if source_type == "url" and url:
            job.update({"active_step": "step_1_download", "progress": 5, "message": "⬇️ [Bước 1/5] Đang phân tích link video..."})
            from services.downloader import download_from_url

            is_douyin = bool(re.search(r"(douyin\.com|iesdouyin\.com|v\.douyin)", url, re.IGNORECASE))
            if is_douyin:
                job["message"] = "⬇️ [Bước 1/5] Đang tải video gốc Master Douyin (không nén, không watermark)..."
                video_path = download_from_url(job_id, url)
            else:
                job["message"] = f"⬇️ [Bước 1/5] Đang tải video từ URL qua yt-dlp..."
                video_path = download_from_url(job_id, url)

            job["video_path"] = str(video_path)
            vpath = Path(video_path)
            job["video_file"] = {"path": str(vpath), "filename": vpath.name, "size": vpath.stat().st_size}
            job["artifacts"]["source_video"] = str(vpath)
            job["progress"] = 20
            job["message"] = f"✅ [Bước 1/5] Đã tải xong video gốc ({vpath.stat().st_size / 1048576:.1f}MB)"

        if not video_path or not os.path.exists(video_path):
            if source_type != "srt_only":
                raise FileNotFoundError("Không tìm thấy file video đầu vào để xử lý.")

        # Bước 1b: Cắt ảnh bìa tiếng Trung nếu được yêu cầu
        trim_intro = config.get("trim_intro", "auto")
        if video_path and trim_intro and str(trim_intro).lower() != "off":
            from services.video.trimmer import preprocess_trim_video
            trimmed_dest = str(output_dir / "video_clean_cover.mp4")
            job["message"] = "✂️ [Bước 1b] Đang kiểm tra và cắt bỏ ảnh bìa tiếng Trung đầu video..."
            clean_vid, trim_sec = preprocess_trim_video(video_path, output_path=trimmed_dest, trim_mode=str(trim_intro))
            if trim_sec > 0.05:
                video_path = clean_vid
                job["video_path"] = video_path
                job["trimmed_seconds"] = trim_sec
                job["message"] = f"✂️ Đã cắt bỏ {trim_sec:.2f}s bìa tiếng Trung đầu video."
                print(f"✂️ [Pipeline {job_id}] Cut {trim_sec:.2f}s intro cover")

        # =========================================================================
        # BƯỚC 2: NHẬN DIỆN PHỤ ĐỀ (GEMINI 3.8 / WHISPER) & DỊCH THUẬT
        # =========================================================================
        zh_srt_content = config.get("zh_srt_content", "")
        target_srt_content = config.get("target_srt_content") or config.get("srt_content", "")
        target_srt_path = config.get("target_srt_path") or config.get("srt_path", "")
        if zh_srt_content and not steps.get('extract_sub', True):
            valid, errors = validate_srt(zh_srt_content)
            if not valid:
                raise ValueError('Nguồn phụ đề gốc không hợp lệ: ' + '; '.join(errors[:2]))
            job['source_visual_srt'] = zh_srt_content
            job['source_timing_origin'] = 'provided_estimate'

        if target_srt_content and not target_srt_path:
            target_srt_path = str(output_dir / f"hardsub_{target_lang}.srt")
            Path(target_srt_path).write_text(target_srt_content, encoding="utf-8")
            meta = LANGUAGES.get(target_lang, {"name": target_lang, "flag": "🏳️"})
            job["srt_files"][target_lang] = {
                "filename": Path(target_srt_path).name,
                "path": target_srt_path,
                "preview": target_srt_content[:500],
                "lang_name": meta.get("name", target_lang),
                "flag": meta.get("flag", "🏳️"),
            }

        if steps.get("extract_sub", True) and video_path:
            job.update({"active_step": "step_2_subtitles", "progress": 25})
            sub_engine = sub_opts.get("engine", "hybrid").lower()
            style = sub_opts.get("style", "movie")
            translate_method = sub_opts.get("translate_method", "ai")

            whisper_context = None
            whisper_path = None
            gemini_error = None

            if sub_engine in {"whisper", "hybrid"}:
                job["message"] = "🎙️ [Bước 2/5] Whisper đang tạo transcript đối chiếu..." if sub_engine == "hybrid" else "🎙️ [Bước 2/5] Đang nhận diện bằng Whisper..."
                job["progress"] = 30
                try:
                    whisper_context, whisper_path = _extract_whisper_source(job_id, video_path, sub_opts)
                    job.setdefault("subtitle_source_tracks", {})["audio"] = {
                        "engine": "faster-whisper", "path": whisper_path,
                    }
                except Exception as exc:
                    if sub_engine == "whisper":
                        raise
                    job.setdefault("subtitle_source_warnings", {})["audio"] = str(exc)[:500]

            if sub_engine in {"gemini", "hybrid"}:
                model = sub_opts.get("model", "gemini-3.8-flash-high")
                job["message"] = f"🔍 [Bước 2/5] Gemini Vision đang đọc hardsub ({model})..."
                job["progress"] = 40
                try:
                    zh_srt_content = _extract_gemini_source(job_id, video_path, sub_opts)
                    job.setdefault("subtitle_source_tracks", {})["vision"] = {
                        "engine": model, "segments": len(__import__('services.srt_utils', fromlist=['parse_srt']).parse_srt(zh_srt_content)),
                    }
                except Exception as exc:
                    gemini_error = exc
                    job.setdefault("subtitle_source_warnings", {})["vision"] = str(exc)[:500]
                    if sub_engine == "gemini" or not whisper_context:
                        raise
                    zh_srt_content = whisper_context

            if sub_engine == "whisper":
                zh_srt_content = whisper_context or ""

            valid, errors = validate_srt(zh_srt_content)
            if not valid:
                raise RuntimeError("Nguồn phụ đề không hợp lệ: " + "; ".join(errors[:2]))
            if sub_opts.get('visual_timing') and sub_engine in {'gemini', 'hybrid'} and not gemini_error:
                from services.visual_timing import refine_and_store
                job['message'] = '🔎 Đang đối chiếu ranh giới phụ đề với frame gốc...'
                zh_srt_content = refine_and_store(job, video_path, zh_srt_content, output_dir)
            # Keep original-language timing even when optional OCR refinement is
            # disabled or fails. Translation/TTS timelines never replace it.
            job['source_visual_srt'] = zh_srt_content
            job['source_timing_origin'] = 'audio_estimate' if sub_engine == 'whisper' or gemini_error else 'vision_estimate'
            zh_path = output_dir / ("hybrid_zh.srt" if sub_engine == "hybrid" else f"{sub_engine}_zh.srt")
            zh_path.write_text(zh_srt_content, encoding="utf-8")
            job["srt_files"]["zh"] = {
                "filename": zh_path.name, "path": str(zh_path),
                "preview": zh_srt_content[:1500], "lang_name": "中文 (原文)", "flag": "🇨🇳",
            }
            job["artifacts"]["srt_zh"] = str(zh_path)

            job["message"] = f"🌐 [Bước 2/5] Đang gộp ngữ nghĩa và dịch sang {LANGUAGES.get(target_lang, {}).get('name', target_lang)}..."
            job["progress"] = 55
            if translate_method == "ai":
                from services.hybrid_subtitles import semantic_fuse_translate
                fusion = semantic_fuse_translate(
                    zh_srt_content, target_lang, job_id, translation_mode=style,
                    whisper_context=whisper_context if sub_engine == "hybrid" else None,
                    model=sub_opts.get("ai_model"),
                )
                target_srt_content = fusion["display_srt"]
                tts_source_srt = fusion["tts_srt"]
                tts_script_path = output_dir / f"tts_script_{target_lang}.srt"
                tts_script_path.write_text(tts_source_srt, encoding="utf-8")
                job["tts_source_srt"] = tts_source_srt
                job["tts_script"] = {
                    "path": str(tts_script_path), "filename": tts_script_path.name,
                    "segments": len(fusion["tts_groups"]),
                }
                job["artifacts"][f"tts_script_{target_lang}"] = str(tts_script_path)
                if gemini_error:
                    job["semantic_fusion"]["vision_fallback"] = True
            else:
                from services.translation import translate_srt
                target_srt_content = translate_srt(zh_srt_content, target_lang, job_id)
                job["tts_source_srt"] = target_srt_content

            target_srt_path = str(output_dir / f"hardsub_{target_lang}.srt")
            Path(target_srt_path).write_text(target_srt_content, encoding="utf-8")
            meta = LANGUAGES.get(target_lang, {"name": target_lang, "flag": "🏳️"})
            job["srt_files"][target_lang] = {
                "filename": Path(target_srt_path).name, "path": target_srt_path,
                "preview": target_srt_content[:1500],
                "lang_name": meta.get("name", target_lang), "flag": meta.get("flag", "🏳️"),
            }
            job["artifacts"][f"srt_{target_lang}"] = target_srt_path

            if not target_srt_content:
                raise RuntimeError('Không có phụ đề đích hợp lệ sau bước nhận diện/dịch')
            job['step_results']['subtitles'] = {'status': 'done', 'path': target_srt_path}
            _check_cancel(job)

            # ĐIỂM DỪNG DUYỆT PHỤ ĐỀ (HUMAN-IN-THE-LOOP)
            untranslated = job.get('translation_quality', {}).get(target_lang, {}).get('unchanged_segments', [])
            fallback_lines = job.get('translation_fallbacks', {}).get(target_lang, [])
            provider_warning = job.get('translation_provider_warnings', {}).get(target_lang)
            fusion_fallback = bool(job.get('semantic_fusion', {}).get('used_fallback'))
            fusion_needs_review = bool(job.get('semantic_fusion', {}).get('needs_review'))
            pause_for_review = (sub_opts.get("pause_for_review", False) or bool(untranslated)
                                or bool(fallback_lines) or fusion_fallback or fusion_needs_review)
            if untranslated and not (steps.get('tts') or steps.get('burn_sub')):
                job.update(status='partial', message=f'{len(untranslated)} câu chưa được dịch; cần kiểm tra SRT')
                return
            if pause_for_review and (steps.get("tts") or steps.get("burn_sub")):
                if untranslated:
                    review_message = f'⚠️ Còn {len(untranslated)} câu chưa dịch. Hãy sửa trước khi tiếp tục.'
                elif fallback_lines:
                    review_message = f'⚠️ AI dịch gặp lỗi; {len(fallback_lines)} câu đã dùng Google fallback. Hãy duyệt trước khi tiếp tục.'
                elif fusion_fallback:
                    review_message = '⚠️ Semantic Fusion không trả contract hợp lệ; hệ thống đã dùng gộp/dịch fallback. Hãy duyệt trước khi tiếp tục.'
                elif fusion_needs_review:
                    review_message = '⚠️ Semantic Fusion phát hiện câu còn lỗi ngôn ngữ hoặc bị cụt ý. Hãy duyệt trước khi tiếp tục.'
                else:
                    review_message = '⏳ Đã nhận diện & dịch xong! Vui lòng xem trước/chỉnh sửa phụ đề bên dưới rồi nhấn Tiếp tục.'
                job.update({
                    "status": "awaiting_review",
                    "progress": 60,
                    "message": review_message,
                    "pending_pipeline": {
                        "video_path": video_path,
                        "target_lang": target_lang,
                        "target_srt_content": target_srt_content,
                        "target_srt_path": target_srt_path,
                        "steps": steps,
                        "clean_opts": clean_opts,
                        "tts_opts": tts_opts,
                        "burn_opts": burn_opts,
                    },
                    "review_draft": target_srt_content,
                })
                print(f"⏸️ [Pipeline {job_id}] Paused for subtitle review by user.")
                return

        # Nếu không tạm dừng, tiếp tục thực thi các bước 3, 4, 5
        _execute_remaining_steps(
            job_id=job_id,
            video_path=video_path,
            target_lang=target_lang,
            target_srt_content=target_srt_content,
            target_srt_path=target_srt_path,
            steps=steps,
            clean_opts=clean_opts,
            tts_opts=tts_opts,
            burn_opts=burn_opts,
        )

    except Exception as exc:
        print(f"❌ [Pipeline {job_id}] Lỗi: {exc}")
        import traceback
        traceback.print_exc()
        job["status"] = "cancelled" if job.get('cancel') else "error"
        job["message"] = f"Lỗi xử lý: {exc}"


@serial_task
def _resume_worker(job_id: str, pending: dict) -> None:
    """Worker tiếp tục sau khi người dùng bấm Tiếp tục."""
    try:
        _execute_remaining_steps(
            job_id=job_id,
            video_path=pending["video_path"],
            target_lang=pending["target_lang"],
            target_srt_content=pending["target_srt_content"],
            target_srt_path=pending["target_srt_path"],
            steps=pending["steps"],
            clean_opts=pending["clean_opts"],
            tts_opts=pending["tts_opts"],
            burn_opts=pending["burn_opts"],
        )
    except Exception as exc:
        print(f"❌ [Pipeline Resume {job_id}] Lỗi: {exc}")
        job = get_job(job_id)
        if job:
            job["status"] = "cancelled" if job.get('cancel') else "error"
            job["message"] = f"Lỗi tiếp tục quy trình: {exc}"


def _check_cancel(job):
    if job.get('cancel'):
        raise RuntimeError('Đã hủy quy trình')


def _execute_remaining_steps(
    job_id: str,
    video_path: str,
    target_lang: str,
    target_srt_content: str,
    target_srt_path: str,
    steps: dict,
    clean_opts: dict,
    tts_opts: dict,
    burn_opts: dict,
) -> None:
    """Thực thi các bước 3 (Xóa chữ), 4 (Lồng tiếng TTS), 5 (In sub)."""
    job = get_job(job_id)
    if not job:
        return
    _check_cancel(job)
    job['status'] = 'processing'
    job.setdefault('step_results', {})
    if not target_srt_content and target_srt_path:
        target_srt_content = Path(target_srt_path).read_text(encoding='utf-8-sig')
    if (steps.get('tts') or steps.get('burn_sub')) and not target_srt_content:
        raise ValueError('Công đoạn TTS/in phụ đề cần file SRT hợp lệ')
    if (steps.get('clean_video') or steps.get('burn_sub')) and not video_path:
        raise ValueError('Công đoạn xử lý hình cần video đầu vào')

    output_dir = OUTPUT_FOLDER / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_video_path = None

    # =========================================================================
    # BƯỚC 3: XỬ LÝ PHỤ ĐỀ GỐC (CLEAN PLATE - XÓA CHỮ XUẤT VIDEO SẠCH)
    # =========================================================================
    if steps.get("clean_video", False) and video_path:
        job.update({"active_step": "step_3_clean", "progress": 65, "message": "🧹 [Bước 3/5] Đang xóa phụ đề tiếng Trung / Logo / Banner..."})
        from services.burn_sub import detect_hardsub_region
        from services.video.clean_pipeline import clean_video_pipeline

        reg_mode = clean_opts.get("region_mode", "auto")
        sub_reg = None
        if reg_mode == "manual" and clean_opts.get("sub_region"):
            sub_reg = clean_opts["sub_region"]
        else:
            job["message"] = "🔍 [Bước 3/5] Đang tự động quét tọa độ hardsub bằng AI OCR..."
            sub_reg = detect_hardsub_region(video_path, job_id=job_id, srt_content=job.get('source_visual_srt') or target_srt_content)

        job['source_sub_region'] = dict(sub_reg)
        job['source_geometry_video_path'] = video_path
        job.setdefault('artifacts', {}).setdefault('source_video', video_path)
        clean_out = str(output_dir / "video_clean_plate.mp4")
        engine = clean_opts.get("engine", "opencv")
        clean_result = clean_video_pipeline(
            video_path=video_path,
            sub_region=sub_reg,
            # Do not gate original-text removal by a translated/display/TTS timeline.
            srt_content=job.get('source_visual_srt') or target_srt_content or "",
            output_path=clean_out,
            job_id=job_id,
            burn_key="clean_plate",
            engine=engine,
            re_burn_ass_path=None,
            tts_audio_path=None,
            extra_regions=clean_opts.get("extra_regions"),
            timing_guard=clean_opts.get('visual_guard', True),
        )
        _check_cancel(job)
        if not os.path.isfile(clean_out) or os.path.getsize(clean_out) == 0:
            raise RuntimeError('Không tạo được video sạch')
        clean_result.setdefault('sub_region', dict(sub_reg))
        job['source_sub_region'] = dict(clean_result['sub_region'])
        job['step_results']['clean'] = clean_result
        if os.path.exists(clean_out):
            clean_video_path = clean_out
            csize = os.path.getsize(clean_out)
            job["clean_video"] = {"path": clean_out, "filename": "video_clean_plate.mp4", "size": csize}
            job["burn_clean"] = {"status": "done", "path": clean_out, "size": csize}
            job["artifacts"]["clean_video"] = clean_out
            job["message"] = f"✅ [Bước 3/5] Đã xuất video sạch chữ ({csize / 1048576:.1f}MB)."

            # ĐỒNG BỘ: Cập nhật video_path của pipeline sang video sạch cho các bước tiếp theo
            video_path = clean_out
            job["video_path"] = clean_out
            job["video_file"] = {"path": clean_out, "filename": Path(clean_out).name, "size": csize}

    # =========================================================================
    # BƯỚC 4: TỪ FILE .SRT -> THUYẾT MINH TTS (SMART SYNC)
    # =========================================================================
    tts_audio_path = None
    if steps.get("tts", False) and (target_srt_path or target_srt_content):
        job.update({"active_step": "step_4_tts", "progress": 75, "message": "🎙️ [Bước 4/5] Đang tạo giọng đọc thuyết minh khớp timeline phụ đề..."})
        from services.srt_to_tts import process_srt_to_tts

        tts_engine = tts_opts.get("engine", "edge")
        voice = tts_opts.get("voice") or None
        align_mode = tts_opts.get("align_mode", "smart_sync")
        margin = int(tts_opts.get("margin", 60))
        max_speed = float(tts_opts.get("max_speed", 1.45))

        tts_out = str(output_dir / f"tts_{target_lang}.mp3")
        tts_input_srt = job.get("tts_source_srt") or target_srt_content
        res_tts = process_srt_to_tts(
            srt_content=tts_input_srt,
            output_audio_path=tts_out,
            job_id=job_id,
            manage_status=False,
            lang=target_lang,
            engine=tts_engine,
            voice=voice,
            options={'align_mode': align_mode, 'safety_margin_ms': margin, 'max_speed_ratio': max_speed,
                     'segment_speakers': job.get('tts_segment_speakers') or job.get('segment_speakers'), 'speaker_voices': tts_opts.get('speaker_voices'),
                     'style_prompt': tts_opts.get('style_prompt')},
        )
        _check_cancel(job)
        job['step_results']['tts'] = res_tts
        if res_tts.get('status') != 'done':
            job.update(status='partial', progress=100, active_step='needs_review', message='TTS còn đoạn thiếu hoặc cần chỉnh thời gian. Kiểm tra audio trước khi xuất video.')
            return
        if not os.path.isfile(tts_out) or os.path.getsize(tts_out) == 0:
            raise RuntimeError('Không tạo được audio TTS')
        if res_tts.get('srt_path'):
            job["aligned_tts_srt"] = {"path": res_tts['srt_path'], "filename": Path(res_tts['srt_path']).name}
        if (not job.get("semantic_fusion") and burn_opts.get('audio_mode', 'keep_original') != 'keep_original'
                and res_tts.get('srt_path')):
            target_srt_path = res_tts['srt_path']
            target_srt_content = Path(target_srt_path).read_text(encoding='utf-8')

        if os.path.exists(tts_out):
            tts_audio_path = tts_out
            asize = os.path.getsize(tts_out)
            tts_status = res_tts.get("status", "done")
            failed_segs = res_tts.get("failed_segments", [])
            total_segs = res_tts.get("total_segments", 0)
            if failed_segs:
                msg = f"Hoàn thành một phần ({asize / 1048576:.1f}MB • thiếu {len(failed_segs)}/{total_segs} đoạn)"
            else:
                msg = f"Hoàn thành ({asize / 1048576:.1f}MB, 0% chồng chéo)"

            job[f"tts_{target_lang}"] = {
                "status": tts_status,
                "progress": 100,
                "message": msg,
                "path": tts_out,
                "filename": f"tts_{target_lang}.mp3",
                "size": asize,
                "duration": res_tts.get("audio_duration_s", 0),
                "failed_segments": failed_segs,
                "total_segments": total_segs,
            }
            job["artifacts"][f"tts_{target_lang}"] = tts_out
            job["message"] = f"{'✅' if not failed_segs else '⚠️'} [Bước 4/5] {msg}"

    # =========================================================================
    # BƯỚC 5: IN HARDSUB ĐÃ DỊCH & XUẤT VIDEO THÀNH PHẨM
    # =========================================================================
    if steps.get("burn_sub", True) and video_path and target_srt_content:
        _check_cancel(job)
        job.update({"active_step": "step_5_burn", "progress": 85, "message": "🎬 [Bước 5/5] Đang in phụ đề đã dịch và xuất video thành phẩm..."})
        from services.burn_sub import burn_sub_video

        # Nếu Bước 3 đã chạy -> dùng video sạch làm nguồn in sub!
        input_vid_for_burn = clean_video_path if (clean_video_path and os.path.exists(clean_video_path)) else video_path
        already_cleaned = bool(clean_video_path and os.path.exists(clean_video_path))

        audio_mode = burn_opts.get("audio_mode", "keep_original")
        keep_original_audio = (audio_mode == "keep_original") or not steps.get("tts", False)
        bgm_mode = "keep_original" if keep_original_audio else ("duck" if audio_mode == "tts_ducking" else ("none" if audio_mode == 'tts_only' else "ai"))

        # NẾU ĐÃ XÓA CHỮ Ở BƯỚC 3 -> PURE BURN (In sub mới thẳng lên video sạch, không inpaint lại lần 2!)
        # NẾU CHƯA XÓA Ở BƯỚC 3 -> BLUR (Tự động che mờ viền mềm để không bị lẫn vào chữ cũ)
        r_mode = "pure_burn" if already_cleaned else burn_opts.get('render_mode', "blur")
        if r_mode not in {'pure_burn','blur','inpaint_burn'}:
            raise ValueError('Chế độ render không hợp lệ')
        clean_hardsub = not already_cleaned

        sub_reg = None
        if burn_opts.get("region_mode") == "manual" and burn_opts.get("sub_region"):
            sub_reg = burn_opts["sub_region"]
        elif already_cleaned:
            # Original glyphs are gone: do not OCR the cleaned video and fall
            # back to a different default position for the translated subtitle.
            sub_reg = job.get('source_sub_region')

        burn_info = burn_sub_video(
            job_id=job_id,
            lang=target_lang,
            srt_content=target_srt_content,
            clean_timing_srt=job.get('source_visual_srt'),
            timing_guard=clean_opts.get('visual_guard', True),
            sub_region=sub_reg,
            extra_regions=burn_opts.get("extra_regions"),
            render_mode=r_mode,
            inpaint_engine=clean_opts.get("engine", "opencv"),
            trim_intro="off",
            translate_title=False,
            title_lang=target_lang,
            brand_name="",
            bgm_mode=bgm_mode,
            bgm_volume=float(burn_opts.get("bgm_volume", 0.8)),
            clean_hardsub=clean_hardsub,
            clean_logo=False,
            clean_title=False,
            burn_new_sub=True,
            keep_original_audio=keep_original_audio,
            video_path=input_vid_for_burn,
        )

        final_burned_path = burn_info.get("path")
        _check_cancel(job)
        if not final_burned_path or not os.path.isfile(final_burned_path) or os.path.getsize(final_burned_path) == 0:
            raise RuntimeError('Render chưa tạo được video đầu ra hợp lệ')
        # Commit parent artifacts and terminal status together, instead of
        # leaving an intermediate 85% snapshot after a completed child render.
        bsize = os.path.getsize(final_burned_path)
        finished_steps = dict(job.get('step_results') or {})
        finished_steps['burn'] = burn_info
        finished_artifacts = dict(job.get('artifacts') or {})
        finished_artifacts['final_video'] = str(final_burned_path)
        final_fields = {
            'step_results': finished_steps,
            'artifacts': finished_artifacts,
            f'burn_{target_lang}': {
                'status': 'done', 'path': str(final_burned_path),
                'filename': Path(final_burned_path).name, 'size': bsize,
            },
        }
    else:
        final_fields = {}

    # HOÀN TẤT QUY TRÌNH (DONE)
    # =========================================================================
    _check_cancel(job)
    job.update({
        **final_fields,
        "status": "done",
        "progress": 100,
        "active_step": "completed",
        "message": "🎉 Tất cả các công đoạn đã hoàn thành xuất sắc!",
    })
    print(f"🎉 [Pipeline {job_id}] Toàn bộ quy trình đã hoàn tất thành công!")
