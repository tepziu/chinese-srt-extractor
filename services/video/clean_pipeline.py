"""
clean_pipeline.py — End-to-end AI Clean Plate Video Inpainting Pipeline.
Completely removes hardcoded Chinese subtitles by cropping only active subtitle frames,
running inpainting (OpenCV / LaMa), feather-blending, and re-encoding via FFmpeg NVENC.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from config import DEVICE, OUTPUT_FOLDER, jobs
from services.burn_sub import extract_subtitle_intervals
from services.video.inpainters.lama_inpaint import LamaInpainter
from services.video.inpainters.opencv_inpaint import OpenCVInpainter
from services.video.mask_generator import feather_blend, generate_text_mask


def clean_video_pipeline(
    video_path: str,
    sub_region: dict,
    srt_content: str,
    output_path: str,
    job_id: str,
    burn_key: str = "burn_vi",
    engine: str = "opencv",
    re_burn_ass_path: str | None = None,
    tts_audio_path: str | None = None,
) -> dict:
    """Execute AI Clean Plate inpainting pipeline.

    Args:
        video_path: source video file.
        sub_region: dict with x_ratio, y_ratio, w_ratio, h_ratio.
        srt_content: SRT subtitle string to derive active frame intervals.
        output_path: destination MP4 file.
        job_id: job identifier for progress reporting.
        burn_key: progress key in jobs[job_id].
        engine: 'opencv' (fast 80+ fps) or 'lama' (deep learning).
        re_burn_ass_path: optional ASS subtitle path to burn on top of the clean video.
        tts_audio_path: optional TTS audio to replace original audio.

    Returns:
        dict with output video metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    # Calculate pixel bounding box from sub_region
    # Y-bounds from detected subtitle region
    sub_y = int(height * sub_region.get("y_ratio", 0.86))
    sub_h = int(height * sub_region.get("h_ratio", 0.09))
    sub_y = max(0, min(sub_y, height - 4))
    sub_h = max(4, min(sub_h, height - sub_y))
    sub_y -= sub_y % 2
    sub_h -= sub_h % 2

    # X-bounds: Wide strip (90% width, 5% margin) so long sentences (14-20 chars) are never cut off
    sub_x = max(0, int(width * 0.05))
    sub_w = min(width - sub_x, int(width * 0.90))
    sub_x -= sub_x % 2
    sub_w -= sub_w % 2

    # Extract subtitle intervals from SRT
    intervals = extract_subtitle_intervals(srt_content, min_gap=0.5, pad_start=0.10, pad_end=0.15)

    # Initialize inpainter engine
    if engine == "lama":
        inpainter = LamaInpainter()
    else:
        inpainter = OpenCVInpainter(method="telea")

    temp_clean_video = str(OUTPUT_FOLDER / f"{job_id}_raw_clean.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_clean_video, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Không thể khởi tạo VideoWriter")

    if job_id and jobs.get(job_id):
        jobs[job_id][burn_key]["message"] = f"🧹 Đang xóa chữ AI ({engine})..."
        jobs[job_id][burn_key]["progress"] = 25

    print(f"🎬 Starting Clean Plate inpainting [{engine}]: {total_frames} frames, {width}x{height}, sub_box={sub_w}x{sub_h} at ({sub_x},{sub_y})")

    frame_idx = 0
    inpainted_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if job_id and jobs.get(job_id, {}).get("cancel"):
                raise RuntimeError("Đã hủy (Stop)")

            current_time = frame_idx / fps if fps > 0 else 0
            has_sub = any(s <= current_time <= e for s, e in intervals)

            if has_sub:
                crop_strip = frame[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w]
                # Generate accurate text mask directly on current frame with full outline dilation
                mask = generate_text_mask(crop_strip, dilation_radius=14)
                if mask.max() > 0:
                    inpainted_strip = inpainter.inpaint(crop_strip, mask)
                    blended_strip = feather_blend(crop_strip, inpainted_strip, mask, blur_ksize=9)
                    frame[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w] = blended_strip
                    inpainted_count += 1

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0 and job_id and jobs.get(job_id):
                pct = 25 + int((frame_idx / max(1, total_frames)) * 55)
                jobs[job_id][burn_key]["progress"] = min(pct, 80)
                fps_rate = frame_idx / max(0.1, time.time() - t_start)
                jobs[job_id][burn_key]["message"] = (
                    f"🧹 Đang xóa chữ ({engine}): {pct}% ({frame_idx}/{total_frames}f, {fps_rate:.1f} fps)"
                )
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t_start
    print(f"✅ Inpainting loop finished: {inpainted_count}/{frame_idx} frames cleaned in {elapsed:.1f}s ({frame_idx/max(0.1, elapsed):.1f} fps)")

    if job_id and jobs.get(job_id):
        jobs[job_id][burn_key]["progress"] = 85
        jobs[job_id][burn_key]["message"] = "🎬 Đang đóng gói video & âm thanh..."

    # FFmpeg final encoding with audio and optional new subtitle burn
    has_tts = tts_audio_path and os.path.exists(tts_audio_path)
    audio_inputs = ["-i", tts_audio_path] if has_tts else []
    audio_map = ["-map", "1:a"] if has_tts else ["-map", "0:a?", "-c:a", "copy"]

    filter_complex = []
    if re_burn_ass_path and os.path.exists(re_burn_ass_path):
        ass_esc = str(re_burn_ass_path).replace("\\", "/").replace(":", "\\:")
        filter_complex = ["-filter_complex", f"[0:v]ass='{ass_esc}'[vout]", "-map", "[vout]"]
    else:
        filter_complex = ["-map", "0:v"]

    if DEVICE == "cuda":
        video_codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22", "-b:v", "0"]
    else:
        video_codec = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_clean_video,
        *(["-i", video_path] if not has_tts else audio_inputs),
        *filter_complex,
        *audio_map,
        *video_codec,
        "-shortest",
        output_path,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    Path(temp_clean_video).unlink(missing_ok=True)

    if res.returncode != 0:
        err = res.stderr[-500:] if res.stderr else "Lỗi FFmpeg không xác định"
        raise RuntimeError(f"Lỗi đóng gói video: {err}")

    if not os.path.exists(output_path):
        raise RuntimeError("File video đầu ra không được tạo")

    file_size = os.path.getsize(output_path)
    mode_label = "Xóa sạch + Sub mới" if re_burn_ass_path else "Xóa sạch chữ (Clean Plate)"
    audio_label = "TTS" if has_tts else "gốc"

    result = {
        "status": "done",
        "progress": 100,
        "message": f"Hoàn thành ({file_size / 1048576:.1f}MB) • {mode_label} ({engine}) • Audio: {audio_label}",
        "path": output_path,
        "filename": Path(output_path).name,
        "size": file_size,
        "duration": round(duration, 1),
        "method": f"clean_{engine}",
        "audio_replaced": has_tts,
        "inpainted_frames": inpainted_count,
    }

    if job_id and jobs.get(job_id):
        jobs[job_id][burn_key] = result

    return result
