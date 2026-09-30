"""
clean_pipeline.py — End-to-end AI Clean Plate Video Inpainting Pipeline.
Processes hardcoded subtitles using source intervals and bounded visual fallback,
running inpainting (OpenCV / LaMa), feather-blending, and re-encoding via FFmpeg NVENC.
"""

from __future__ import annotations

import json
import os
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from config import DEVICE, OUTPUT_FOLDER, jobs
from services.burn_sub import extract_subtitle_intervals
from services.video.inpainters.lama_inpaint import LamaInpainter
from services.video.inpainters.opencv_inpaint import OpenCVInpainter
from services.media_process import FrameEncoder, encoder_args
from services.video.mask_generator import feather_blend, generate_text_mask
from services.srt_utils import parse_srt_timing
from services.video.timing_guard import SubtitleTimingGuard


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
    extra_regions: list | None = None,
    timing_guard: bool = True,
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
    if width < 4 or height < 4 or not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError('Thông số video không hợp lệ hoặc không xác định được số frame')

    # Calculate pixel bounding box from sub_region
    # Y-bounds from detected subtitle region
    sub_y = int(height * sub_region.get("y_ratio", 0.86))
    sub_h = int(height * sub_region.get("h_ratio", 0.09))
    sub_y = max(0, min(sub_y, height - 4))
    sub_h = max(4, min(sub_h, height - sub_y))
    sub_y -= sub_y % 2
    sub_h -= sub_h % 2

    # Honor the full manually selected rectangle, including horizontal bounds.
    sub_x = max(0, min(int(width * sub_region.get("x_ratio", 0.05)), width - 4))
    sub_w = max(4, min(int(width * sub_region.get("w_ratio", 0.90)), width - sub_x))

    # Extract subtitle intervals from SRT
    intervals = extract_subtitle_intervals(srt_content, min_gap=0.5, pad_start=0.10, pad_end=0.15)
    guard = None
    extra_presence_intervals = []
    detected_outside_srt_frames = 0
    guard_error = None
    guard_started = time.monotonic()
    if timing_guard and intervals:
        try:
            guard = SubtitleTimingGuard.from_video(
                video_path, parse_srt_timing(srt_content), (sub_x, sub_y, sub_w, sub_h),
                cancelled=lambda: bool(job_id and jobs.get(job_id, {}).get('cancel')),
            )
            extra_presence_intervals, detected_outside_srt_frames = guard.scan_extra_intervals(
                video_path, intervals, (sub_x, sub_y, sub_w, sub_h),
                cancelled=lambda: bool(job_id and jobs.get(job_id, {}).get('cancel')),
            )
        except Exception as exc:
            if job_id and jobs.get(job_id, {}).get('cancel'):
                cap.release()
                raise
            guard_error = str(exc)[:200]
    guard_prepare_seconds = time.monotonic() - guard_started

    from contextlib import nullcontext
    from config import acquire_gpu_slot

    gpu_ctx = acquire_gpu_slot() if engine == "lama" else nullcontext()
    with gpu_ctx:
        # Initialize inpainter engine
        if engine == "lama":
            inpainter = LamaInpainter()
        else:
            inpainter = OpenCVInpainter(method="telea")

        actual_engine = 'opencv' if engine == 'lama' and inpainter.session is None else engine
        has_tts = bool(tts_audio_path and os.path.isfile(tts_audio_path))
        audio_source = tts_audio_path if has_tts else video_path
        filters = []
        if re_burn_ass_path:
            if not os.path.isfile(re_burn_ass_path):
                raise ValueError('Không tìm thấy file ASS')
            ass_esc = str(re_burn_ass_path).replace('\\', '/').replace(':', '\\:').replace("'", "'\\''")
            filters = ['-vf', f"ass='{ass_esc}'"]
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
               '-i', str(audio_source), '-map', '0:v', '-map', '1:a?', *filters,
               *encoder_args(), '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
               *(['-af', 'apad'] if has_tts else []), '-t', str(duration),
               '-movflags', '+faststart', str(output_path)]
        writer = FrameEncoder(cmd, job_id)
        interval_idx = 0
        presence_idx = 0
        succeeded = False

        if job_id and jobs.get(job_id):
            jobs[job_id].setdefault(burn_key, {})["message"] = f"🧹 Đang xóa chữ AI ({engine})..."
            jobs[job_id].setdefault(burn_key, {})["progress"] = 25

        print(f"🎬 Starting Clean Plate inpainting [{engine}]: {total_frames} frames, {width}x{height}, sub_box={sub_w}x{sub_h} at ({sub_x},{sub_y})")

        frame_idx = 0
        inpainted_count = 0
        recovered_boundary_frames = 0
        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if job_id and jobs.get(job_id, {}).get("cancel"):
                    raise RuntimeError("Đã hủy (Stop)")

                current_time = frame_idx / fps if fps > 0 else 0
                while interval_idx < len(intervals) and intervals[interval_idx][1] < current_time:
                    interval_idx += 1
                has_sub = not intervals or (interval_idx < len(intervals) and intervals[interval_idx][0] <= current_time)
                while presence_idx < len(extra_presence_intervals) and extra_presence_intervals[presence_idx][1] < current_time:
                    presence_idx += 1
                extra_present = (presence_idx < len(extra_presence_intervals) and
                                 extra_presence_intervals[presence_idx][0] <= current_time)

                mask = None
                crop_strip = frame[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w]
                if has_sub:
                    mask = generate_text_mask(crop_strip, dilation_radius=14)
                elif guard is not None:
                    # An approximate/unverified SRT is not enough to skip visible
                    # source glyphs. Extra erasing requires a visual reference
                    # from this video; brightness alone does not trigger it.
                    mask = guard.mask_if_visible(crop_strip, current_time, allow_style=extra_present)
                    if mask is not None:
                        recovered_boundary_frames += 1
                if mask is not None and mask.max() > 0:
                    inpainted_strip = inpainter.inpaint(crop_strip, mask)
                    blended_strip = feather_blend(crop_strip, inpainted_strip, mask, blur_ksize=9)
                    frame[sub_y : sub_y + sub_h, sub_x : sub_x + sub_w] = blended_strip
                    inpainted_count += 1

                # Inpaint extra regions (e.g. top title card, logos)
                if extra_regions:
                    for er in extra_regions:
                        ey = int(height * er.get("y_ratio", 0))
                        eh = int(height * er.get("h_ratio", 0.05))
                        ex = int(width * er.get("x_ratio", 0))
                        ew = int(width * er.get("w_ratio", 0.3))
                        ey = max(0, min(ey, height - 4))
                        eh = max(4, min(eh, height - ey))
                        ex = max(0, min(ex, width - 4))
                        ew = max(4, min(ew, width - ex))
                        e_crop = frame[ey : ey + eh, ex : ex + ew]
                        e_mask = generate_text_mask(e_crop, dilation_radius=10)
                        if e_mask.max() > 0:
                            e_inp = inpainter.inpaint(e_crop, e_mask)
                            frame[ey : ey + eh, ex : ex + ew] = feather_blend(e_crop, e_inp, e_mask, blur_ksize=7)

                writer.write(frame)
                frame_idx += 1

                if frame_idx % 30 == 0 and job_id and jobs.get(job_id):
                    pct = 25 + int((frame_idx / max(1, total_frames)) * 55)
                    jobs[job_id].setdefault(burn_key, {})["progress"] = min(pct, 80)
                    fps_rate = frame_idx / max(0.1, time.time() - t_start)
                    jobs[job_id].setdefault(burn_key, {})["message"] = (
                        f"🧹 Đang xóa chữ ({engine}): {pct}% ({frame_idx}/{total_frames}f, {fps_rate:.1f} fps)"
                    )
            if total_frames > 0 and frame_idx < total_frames - 2:
                raise RuntimeError('Video bị cắt ngắn khi giải mã')
            succeeded = True
        finally:
            cap.release()
            try:
                writer.close(abort=not succeeded)
            except Exception:
                Path(output_path).unlink(missing_ok=True)
                raise
            if not succeeded:
                Path(output_path).unlink(missing_ok=True)

    elapsed = time.time() - t_start
    print(f"✅ Inpainting loop finished: {inpainted_count}/{frame_idx} frames cleaned in {elapsed:.1f}s ({frame_idx/max(0.1, elapsed):.1f} fps)")

    if job_id and jobs.get(job_id):
        jobs[job_id].setdefault(burn_key, {})["progress"] = 85
        jobs[job_id].setdefault(burn_key, {})["message"] = "🎬 Đang đóng gói video & âm thanh..."

    if not os.path.exists(output_path):
        raise RuntimeError("File video đầu ra không được tạo")

    file_size = os.path.getsize(output_path)
    fallback_frames = getattr(inpainter, 'fallback_count', 0)
    if engine == 'lama' and inpainter.session is not None and fallback_frames:
        actual_engine = 'lama_opencv_fallback'
    mode_label = "Xóa sạch + Sub mới" if re_burn_ass_path else "Xóa sạch chữ (Clean Plate)"
    audio_label = "TTS" if has_tts else "gốc"

    result = {
        "status": "done",
        "progress": 100,
        "message": f"Hoàn thành ({file_size / 1048576:.1f}MB) • {mode_label} ({actual_engine}) • Audio: {audio_label}",
        "path": output_path,
        "filename": Path(output_path).name,
        "size": file_size,
        "duration": round(duration, 1),
        "method": f"clean_{actual_engine}",
        "requested_engine": engine,
        "fallback_calls": fallback_frames,
        "encode_passes": 1,
        "timing_mode": "cfr",
        "audio_replaced": has_tts,
        "inpainted_frames": inpainted_count,
        "timing_strategy": "source_srt_visual_guard" if guard is not None else "srt_intervals" if intervals else "all_frames",
        "timing_guard": {
            "enabled": bool(timing_guard and intervals),
            "templates": guard.template_count if guard is not None else 0,
            "missing_templates": guard.missing_templates if guard is not None else len(parse_srt_timing(srt_content)),
            "recovered_frames": recovered_boundary_frames,
            "detected_outside_srt_frames": detected_outside_srt_frames,
            "presence_gap_intervals": guard.presence_gap_intervals if guard is not None else [],
            "prepare_seconds": round(guard_prepare_seconds, 3),
            "window_seconds": guard.window if guard is not None else 0,
            "error": guard_error,
        },
        "qc_status": "not_checked",
        "sub_region": {**sub_region, "x_ratio": sub_x/width, "y_ratio": sub_y/height,
                       "w_ratio": sub_w/width, "h_ratio": sub_h/height},
    }

    if job_id and jobs.get(job_id):
        jobs[job_id][burn_key] = result

    return result
