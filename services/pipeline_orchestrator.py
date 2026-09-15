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


def start_pipeline_job(job_id: str, config: Dict[str, Any]) -> dict:
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

    thread = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, config),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued", "message": "Pipeline đã bắt đầu chạy ngầm."}


def resume_pipeline_job(job_id: str, updated_srt: Optional[str] = None) -> dict:
    """Tiếp tục pipeline sau khi người dùng đã duyệt/sửa phụ đề."""
    job = get_job(job_id)
    if not job:
        raise RuntimeError(f"Không tìm thấy job: {job_id}")

    pending = job.get("pending_pipeline")
    if not pending:
        raise RuntimeError(f"Job {job_id} không ở trạng thái chờ duyệt (awaiting_review).")

    # Cập nhật lại nội dung SRT đã được người dùng chỉnh sửa
    target_lang = pending.get("target_lang", "vi")
    if updated_srt and updated_srt.strip():
        valid_srt, srt_errs = validate_srt(updated_srt.strip())
        if not valid_srt:
            raise ValueError(f"Phụ đề chỉnh sửa không hợp lệ: {'; '.join(srt_errs[:2])}")
        output_dir = OUTPUT_FOLDER / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        srt_path = output_dir / f"hardsub_{target_lang}.srt"
        srt_path.write_text(updated_srt.strip(), encoding="utf-8")
        if "srt_files" not in job:
            job["srt_files"] = {}
        meta = LANGUAGES.get(target_lang, {"name": target_lang, "flag": "🏳️"})
        job["srt_files"][target_lang] = {
            "filename": srt_path.name,
            "path": str(srt_path),
            "preview": updated_srt[:500],
            "lang_name": meta.get("name", target_lang),
            "flag": meta.get("flag", "🏳️"),
        }
        pending["target_srt_content"] = updated_srt.strip()
        pending["target_srt_path"] = str(srt_path)

    job["status"] = "processing"
    job["message"] = "Đã xác nhận phụ đề! Đang tiếp tục các công đoạn tiếp theo..."
    job["pending_pipeline"] = None

    thread = threading.Thread(
        target=_resume_worker,
        args=(job_id, pending),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "processing", "message": "Đang tiếp tục các bước tiếp theo."}


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
        # =========================================================================
        # BƯỚC 1: XỬ LÝ NGUỒN VIDEO & TẢI DOUYIN
        # =========================================================================
        video_path = job.get("video_path")
        source_type = config.get("source_type", "upload")
        url = config.get("url", "").strip()

        if source_type == "url" and url:
            job.update({"active_step": "step_1_download", "progress": 5, "message": "⬇️ [Bước 1/5] Đang phân tích link video..."})
            from services.downloader import download_douyin_master, download_from_url

            is_douyin = bool(re.search(r"(douyin\.com|iesdouyin\.com|v\.douyin)", url, re.IGNORECASE))
            if is_douyin:
                job["message"] = "⬇️ [Bước 1/5] Đang tải video gốc Master Douyin (không nén, không watermark)..."
                video_path = download_douyin_master(job_id, url)
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
            sub_engine = sub_opts.get("engine", "gemini").lower()
            style = sub_opts.get("style", "movie")
            translate_method = sub_opts.get("translate_method", "ai")

            if sub_engine == "gemini":
                # Option A: Gemini 3.8 Flash High (Vision OCR)
                gemini_model = sub_opts.get("model", "gemini-3.8-flash-high")
                job["message"] = f"🔍 [Bước 2/5] Đang nhận diện hardsub bằng Gemini Vision ({gemini_model})..."
                job["progress"] = 35

                from services.hardsub_gemini import (
                    create_gemini_proxy_video,
                    extract_hardsub_via_local_gateway,
                    parse_srt_from_text,
                    HARDSUB_PROMPT,
                )

                api_key = get_gemini_api_key()
                use_local_gw = bool(AI_TRANSLATE_CONFIG.get("api_key"))

                raw_text = ""
                if use_local_gw or gemini_model in ("gemini-3.8-flash-high", "gemini-3.7-flash-high"):
                    raw_text = extract_hardsub_via_local_gateway(video_path, job_id, model_name=gemini_model)
                else:
                    from google import genai
                    upload_path, is_proxy = create_gemini_proxy_video(video_path, job_id)
                    client = genai.Client(api_key=api_key)
                    uploaded_f = client.files.upload(file=upload_path)
                    while uploaded_f.state.name == "PROCESSING":
                        time.sleep(4)
                        uploaded_f = client.files.get(name=uploaded_f.name)
                    res = client.models.generate_content(model=gemini_model, contents=[uploaded_f, HARDSUB_PROMPT])
                    raw_text = res.text or ""

                zh_srt_content = parse_srt_from_text(raw_text)
                valid, errors = validate_srt(zh_srt_content)
                if not valid:
                    raise RuntimeError("Gemini không trích xuất được phụ đề tiếng Trung hợp lệ: " + "; ".join(errors[:2]))

                zh_path = output_dir / "hardsub_zh.srt"
                zh_path.write_text(zh_srt_content, encoding="utf-8")
                job["srt_files"]["zh"] = {
                    "filename": zh_path.name,
                    "path": str(zh_path),
                    "preview": zh_srt_content[:500],
                    "lang_name": "中文 (原文)",
                    "flag": "🇨🇳",
                }

                # Dịch sang ngôn ngữ đích (target_lang)
                job["message"] = f"🌐 [Bước 2/5] Đang dịch phụ đề sang {LANGUAGES.get(target_lang, {}).get('name', target_lang)} ({style})..."
                job["progress"] = 55

                from services.translation import translate_srt, translate_srt_ai
                if translate_method == "ai":
                    target_srt_content = translate_srt_ai(zh_srt_content, target_lang, job_id, translation_mode=style)
                else:
                    target_srt_content = translate_srt(zh_srt_content, target_lang, job_id)

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
                job["artifacts"][f"srt_{target_lang}"] = target_srt_path

            else:
                # Option B: Whisper ASR (Giọng nói)
                whisper_model = sub_opts.get("whisper_model", "large-v3-turbo")
                job["message"] = f"🎙️ [Bước 2/5] Đang nhận diện giọng nói bằng Faster-Whisper ({whisper_model})..."
                job["progress"] = 35

                from services.whisper_engine import process_video
                process_video(
                    job_id=job_id,
                    video_path=video_path,
                    model_size=whisper_model,
                    translate_langs=[target_lang],
                    translate_method=translate_method,
                    translation_mode=style,
                )

                # Đọc kết quả từ job
                current_j = get_job(job_id)
                srt_info = (current_j.get("srt_files") or {}).get(target_lang) or {}
                target_srt_path = srt_info.get("path", "")
                if target_srt_path and os.path.exists(target_srt_path):
                    target_srt_content = Path(target_srt_path).read_text(encoding="utf-8")

            # ĐIỂM DỪNG DUYỆT PHỤ ĐỀ (HUMAN-IN-THE-LOOP)
            pause_for_review = sub_opts.get("pause_for_review", False)
            if pause_for_review and (steps.get("tts") or steps.get("burn_sub")):
                job.update({
                    "status": "awaiting_review",
                    "progress": 60,
                    "message": "⏳ Đã nhận diện & dịch xong! Vui lòng xem trước/chỉnh sửa phụ đề bên dưới rồi nhấn Tiếp tục.",
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
        job["status"] = "error"
        job["message"] = f"Lỗi xử lý: {exc}"


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
            job["status"] = "error"
            job["message"] = f"Lỗi tiếp tục quy trình: {exc}"


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
            sub_reg = detect_hardsub_region(video_path, job_id=job_id, srt_content=target_srt_content)

        clean_out = str(output_dir / "video_clean_plate.mp4")
        engine = clean_opts.get("engine", "opencv")
        clean_video_pipeline(
            video_path=video_path,
            sub_region=sub_reg,
            srt_content=target_srt_content or "",
            output_path=clean_out,
            job_id=job_id,
            burn_key="clean_plate",
            engine=engine,
            re_burn_ass_path=None,
            tts_audio_path=None,
            extra_regions=clean_opts.get("extra_regions"),
        )
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
        from services.srt_to_tts import srt_to_mp3

        tts_engine = tts_opts.get("engine", "edge")
        voice = tts_opts.get("voice") or TTS_VOICES.get(target_lang, "vi-VN-NamMinhNeural")
        align_mode = tts_opts.get("align_mode", "smart_sync")
        margin = int(tts_opts.get("margin", 60))
        max_speed = float(tts_opts.get("max_speed", 1.45))

        tts_out = str(output_dir / f"tts_{target_lang}.mp3")
        res_tts = srt_to_mp3(
            srt_input=target_srt_path or target_srt_content,
            output_path=tts_out,
            lang=target_lang,
            engine=tts_engine,
            voice=voice,
            align_mode=align_mode,
            safety_margin_ms=margin,
            max_speed_ratio=max_speed,
        )

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
        job.update({"active_step": "step_5_burn", "progress": 85, "message": "🎬 [Bước 5/5] Đang in phụ đề đã dịch và xuất video thành phẩm..."})
        from services.burn_sub import burn_sub_video

        # Nếu Bước 3 đã chạy -> dùng video sạch làm nguồn in sub!
        input_vid_for_burn = clean_video_path if (clean_video_path and os.path.exists(clean_video_path)) else video_path
        already_cleaned = bool(clean_video_path and os.path.exists(clean_video_path))

        audio_mode = burn_opts.get("audio_mode", "keep_original")
        keep_original_audio = (audio_mode == "keep_original") or not steps.get("tts", False)
        bgm_mode = "keep_original" if keep_original_audio else ("ducking" if audio_mode == "tts_ducking" else "ai")

        # NẾU ĐÃ XÓA CHỮ Ở BƯỚC 3 -> PURE BURN (In sub mới thẳng lên video sạch, không inpaint lại lần 2!)
        # NẾU CHƯA XÓA Ở BƯỚC 3 -> BLUR (Tự động che mờ viền mềm để không bị lẫn vào chữ cũ)
        r_mode = "pure_burn" if already_cleaned else "blur"
        clean_hardsub = not already_cleaned

        sub_reg = None
        if not already_cleaned:
            if burn_opts.get("region_mode") == "manual" and burn_opts.get("sub_region"):
                sub_reg = burn_opts["sub_region"]

        burn_info = burn_sub_video(
            job_id=job_id,
            lang=target_lang,
            srt_content=target_srt_content,
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
        if final_burned_path and os.path.exists(final_burned_path):
            bsize = os.path.getsize(final_burned_path)
            job[f"burn_{target_lang}"] = {
                "status": "done",
                "path": str(final_burned_path),
                "filename": Path(final_burned_path).name,
                "size": bsize,
            }
            job["artifacts"]["final_video"] = str(final_burned_path)

    # HOÀN TẤT QUY TRÌNH (DONE)
    # =========================================================================
    job.update({
        "status": "done",
        "progress": 100,
        "active_step": "completed",
        "message": "🎉 Tất cả các công đoạn đã hoàn thành xuất sắc!",
    })
    print(f"🎉 [Pipeline {job_id}] Toàn bộ quy trình đã hoàn tất thành công!")
