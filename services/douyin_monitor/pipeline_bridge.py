"""
pipeline_bridge.py — Automatic Studio Pipeline Execution & Telegram Delivery.
Bridges newly downloaded Douyin videos into the complete translation & dubbing workflow.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

from config import OUTPUT_FOLDER, UPLOAD_FOLDER, create_job, get_job, jobs
from services.douyin_monitor.channel_manager import (
    get_notify_chat_id,
    mark_as_downloaded,
)

from config import TELEGRAM_BOT_TOKEN as BOT_TOKEN


def escape_markdown(text: str) -> str:
    """Safely escape Markdown special characters for Telegram."""
    if not text:
        return ""
    # In Telegram standard Markdown, escape _, *, [, ]
    for ch in ["_", "*", "[", "]", "`"]:
        text = text.replace(ch, f"\\{ch}")
    return text


def send_telegram_message(text: str, chat_id: str | None = None, parse_mode: str | None = "Markdown") -> bool:
    """Send a text message via Telegram Bot API with fallback to plain text."""
    target_chat = chat_id or get_notify_chat_id()
    if not BOT_TOKEN or not target_chat:
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": text,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception as exc:
        if parse_mode:
            # Fallback without Markdown
            return send_telegram_message(text, chat_id=target_chat, parse_mode=None)
        print(f"[Bridge] Telegram sendMessage error: {exc}")
        return False


def send_telegram_document(file_path: str, caption: str = "", chat_id: str | None = None, parse_mode: str | None = "Markdown") -> bool:
    """Send a file document via Telegram Bot API using multipart/form-data."""
    target_chat = chat_id or get_notify_chat_id()
    if not BOT_TOKEN or not target_chat or not os.path.exists(file_path):
        return False

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 49.5:
        send_telegram_message(
            f"⚠️ Video thành phẩm ({file_size_mb:.1f}MB) vượt quá giới hạn 50MB của Telegram Bot.\n"
            f"📁 File đã được lưu sẵn trên máy tại: {file_path}",
            chat_id=target_chat,
            parse_mode=None,
        )
        return False

    import mimetypes
    import uuid

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    filename = Path(file_path).name
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{target_chat}\r\n".encode("utf-8"),
    ]
    if caption:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode("utf-8"))
        if parse_mode:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\n{parse_mode}\r\n".encode("utf-8"))

    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"{filename}\"\r\nContent-Type: {mime_type}\r\n\r\n".encode("utf-8")
        + file_bytes
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )
    body = b"".join(parts)

    try:
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status == 200
    except Exception as exc:
        if parse_mode:
            # Retry without markdown
            return send_telegram_document(file_path, caption=caption, chat_id=target_chat, parse_mode=None)
        print(f"[Bridge] Telegram sendDocument error: {exc}")
        return False


def execute_auto_pipeline(work: dict, channel_cfg: dict, video_path: str) -> None:
    """Execute the end-to-end studio pipeline for a newly discovered Douyin video."""
    aweme_id = str(work.get("aweme_id", ""))
    title = work.get("desc", "Video không tiêu đề")
    nickname = channel_cfg.get("nickname", "Kênh Douyin")
    target_lang = channel_cfg.get("target_lang", "vi")
    style = channel_cfg.get("style", "driving")
    bgm_mode = channel_cfg.get("bgm_mode", "ai")
    clean_hardsub = channel_cfg.get("clean_hardsub", True)
    clean_logo = channel_cfg.get("clean_logo", True)
    translate_title = channel_cfg.get("translate_title", True)
    auto_burn = channel_cfg.get("auto_burn", True)

    print(f"\n{'='*60}")
    print(f"🚀 [AUTO STUDIO] Starting pipeline for [{nickname}] (ID: {aweme_id})")
    print(f"   Tiêu đề: {title[:70]}")
    print(f"{'='*60}")

    # 1. Alert Telegram about the new video
    send_telegram_message(
        f"🔥 *Phát hiện video mới từ kênh [{nickname}]!*\n\n"
        f"📝 *Tiêu đề:* {title}\n"
        f"🎬 *ID:* `{aweme_id}`\n\n"
        f"⚙️ *Đang tự động xử lý*: Dịch `{target_lang.upper()}` ({style}), Tách nhạc nền `{bgm_mode.upper()}`, Inpaint chữ cũ...",
        parse_mode="Markdown",
    )

    job_id = f"dy_{aweme_id}"
    job_upload_dir = UPLOAD_FOLDER / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    target_video = str(job_upload_dir / "source.mp4")
    shutil.copy2(video_path, target_video)

    # Gửi video gốc Master vừa tải về qua Telegram
    size_mb = os.path.getsize(target_video) / (1024 * 1024)
    sent_master = send_telegram_document(
        target_video,
        caption=(
            f"🎬 *Đã tải xong Video Gốc Master từ [{nickname}]*\n\n"
            f"📝 *Tiêu đề:* {title[:120]}\n"
            f"📦 *Dung lượng:* {size_mb:.1f} MB\n"
            f"🎬 *ID:* `{aweme_id}`"
        )
    )
    if not sent_master:
        send_telegram_message(
            f"🎬 *Đã tải xong Video Gốc Master từ [{nickname}]!*\n\n"
            f"📝 *Tiêu đề:* {title[:120]}\n"
            f"📦 *Dung lượng:* {size_mb:.1f} MB" + (" (Vượt quá 50MB giới hạn của Telegram Bot)\n" if size_mb > 49.5 else "\n") +
            f"📁 *Đã lưu an toàn tại:* `{target_video}`\n\n"
            f"⚙️ Đang tiếp tục xử lý dịch `{target_lang.upper()}` ({style}) và in sub mới..."
        )

    if not auto_burn:
        mark_as_downloaded(aweme_id)
        print(f"🎉 [AUTO STUDIO] Video downloaded and delivered for {aweme_id} (auto_burn=False)!")
        return

    try:
        from services.burn_sub import burn_sub_video
        from services.tts import generate_tts_audio
        from services.video.trimmer import preprocess_trim_video
        from services.whisper_engine import process_video

        # Create job entry with explicit video_file
        create_job(
            job_id,
            original_name=title[:80],
            video_path=target_video,
            translate_langs=[target_lang],
            translation_mode=style,
            trim_intro="auto",
        )
        jobs[job_id]["video_file"] = {
            "path": target_video,
            "filename": Path(target_video).name,
            "size": Path(target_video).stat().st_size if Path(target_video).exists() else 0,
        }

        # Step 1: Pre-process intro cover trim
        clean_video, trim_sec = preprocess_trim_video(
            target_video,
            output_path=str(OUTPUT_FOLDER / job_id / "video_clean_cover.mp4"),
            trim_mode="auto",
        )
        if trim_sec > 0.05:
            target_video = clean_video
            jobs[job_id]["video_path"] = target_video
            jobs[job_id]["trimmed_seconds"] = trim_sec
            print(f"✂️ [AUTO] Trimmed {trim_sec:.2f}s intro cover")

        # Step 2: Whisper ASR & AI translation
        process_video(
            job_id=job_id,
            video_path=target_video,
            model_size="large-v3-turbo",
            translate_langs=[target_lang],
            translate_method="ai",
            translation_mode=style,
        )

        job_state = get_job(job_id)
        if not job_state or job_state.get("status") == "error":
            err_msg = job_state.get("message", "Lỗi không xác định") if job_state else "Job lost"
            send_telegram_message(f"❌ Xử lý video `{aweme_id}` thất bại tại bước nhận dạng/dịch: {err_msg}")
            return

        srt_files = job_state.get("srt_files") or {}
        srt_info = srt_files.get(target_lang) or {}
        srt_path = srt_info.get("path") if isinstance(srt_info, dict) else ""
        srt_content = Path(srt_path).read_text(encoding="utf-8") if srt_path and os.path.exists(srt_path) else ""

        # Step 3: Generate TTS Voiceover
        tts_audio_path = None
        if srt_content:
            try:
                tts_res = generate_tts_audio(
                    job_id=job_id,
                    lang=target_lang,
                    srt_content=srt_content,
                    engine="edge",
                )
                tts_audio_path = tts_res.get("path") if isinstance(tts_res, dict) else None
            except Exception as tts_err:
                print(f"[AUTO STUDIO] TTS error (will continue with original audio): {tts_err}")

        # Step 4: Burn sub & Inpaint & BGM Mix
        if auto_burn and srt_content:
            burn_info = burn_sub_video(
                job_id=job_id,
                lang=target_lang,
                srt_content=srt_content,
                sub_region=None,
                extra_regions=None,
                render_mode="inpaint_burn",
                inpaint_engine="opencv",
                trim_intro="off",  # already trimmed at step 1
                translate_title=translate_title,
                title_lang=target_lang,
                brand_name="",
                bgm_mode=bgm_mode,
                bgm_volume=0.80,
                clean_hardsub=clean_hardsub,
                clean_logo=clean_logo,
                clean_title=translate_title,
                burn_new_sub=True,
            )

            final_video_path = burn_info.get("path") if isinstance(burn_info, dict) else None
            if final_video_path and os.path.exists(final_video_path):
                size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
                print(f"✅ [AUTO STUDIO] Video completed: {final_video_path} ({size_mb:.1f}MB)")

                # Deliver video to Telegram
                caption = (
                    f"🎬 *Video Hoàn Chỉnh Mới [{nickname}]*\n\n"
                    f"📝 *Tiêu đề:* {title[:120]}\n"
                    f"🌍 *Ngôn ngữ:* `{target_lang.upper()}` ({style})\n"
                    f"🎵 *Nhạc nền BGM:* `{bgm_mode.upper()}`\n"
                    f"📦 *Dung lượng:* {size_mb:.1f}MB"
                )
                sent = send_telegram_document(final_video_path, caption=caption, parse_mode="Markdown")
                if not sent:
                    send_telegram_message(
                        f"✅ Đã xử lý xong video mới từ [{nickname}]!\n\n"
                        f"📝 Tiêu đề: {title[:120]}\n"
                        f"📦 Dung lượng: {size_mb:.1f}MB\n"
                        f"📁 Lưu tại máy: {final_video_path}",
                        parse_mode=None,
                    )

        # Also deliver SRT
        if srt_path and os.path.exists(srt_path):
            send_telegram_document(srt_path, caption=f"📄 Phụ đề {target_lang.upper()}: {title[:80]}", parse_mode="Markdown")

        # Mark as successfully processed
        mark_as_downloaded(aweme_id)
        print(f"🎉 [AUTO STUDIO] Pipeline finished and delivered for {aweme_id}!")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        send_telegram_message(f"❌ Xử lý tự động video `{aweme_id}` gặp lỗi: {exc}", parse_mode=None)