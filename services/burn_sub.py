"""
burn_sub.py — Burn translated subtitles into video with multi-region blur and replacement
Supports multiple blur regions (subtitle area + extra areas like logos/watermarks)
"""

import os
import re
import json
import time
import subprocess

from config import DEVICE, OUTPUT_FOLDER, MAX_BLUR_REGIONS, jobs


def detect_hardsub_region(video_path: str, job_id: str = None) -> dict:
    """Detect hardcoded subtitle region using EasyOCR on sample frames.
    Returns region in new format: {x_ratio, y_ratio, w_ratio, h_ratio}"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"x_ratio": 0, "y_ratio": 0.78, "w_ratio": 1.0, "h_ratio": 0.18, "method": "fixed_fallback"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    if total_frames < 10 or height < 100:
        cap.release()
        return {"x_ratio": 0, "y_ratio": 0.78, "w_ratio": 1.0, "h_ratio": 0.18, "method": "fixed_fallback"}

    start_time = min(2.0, duration * 0.1)
    end_time = duration * 0.95
    num_samples = 15

    sample_times = [start_time + (end_time - start_time) * i / (num_samples - 1) for i in range(num_samples)]
    sample_frames = [min(int(t * fps), total_frames - 1) for t in sample_times]

    try:
        import easyocr
        reader = easyocr.Reader(['ch_sim'], gpu=True, verbose=False)
    except Exception:
        cap.release()
        return {"x_ratio": 0, "y_ratio": 0.78, "w_ratio": 1.0, "h_ratio": 0.18, "method": "fixed_fallback"}

    if job_id:
        burn_key = [k for k in jobs[job_id] if k.startswith("burn_") and isinstance(jobs[job_id][k], dict)]
        if burn_key:
            jobs[job_id][burn_key[0]]["message"] = f"🔍 Đang quét {num_samples} frame tìm hardsub..."

    all_y_tops = []
    all_y_bottoms = []
    frames_with_text = 0

    for i, frame_pos in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            continue

        crop_start = int(height * 0.80)
        bottom_crop = frame[crop_start:, :]

        try:
            results = reader.readtext(bottom_crop, paragraph=False)
            frame_has_text = False
            for (bbox, text, conf) in results:
                if conf > 0.4 and len(text) >= 2:
                    x_center = sum(p[0] for p in bbox) / 4
                    crop_w = bottom_crop.shape[1]
                    if x_center < crop_w * 0.1 or x_center > crop_w * 0.9:
                        continue

                    y_min = min(p[1] for p in bbox) + crop_start
                    y_max = max(p[1] for p in bbox) + crop_start
                    all_y_tops.append(y_min)
                    all_y_bottoms.append(y_max)
                    frame_has_text = True
            if frame_has_text:
                frames_with_text += 1
        except Exception:
            continue

        if job_id and burn_key:
            jobs[job_id][burn_key[0]]["message"] = f"🔍 Quét frame {i + 1}/{num_samples}..."

    cap.release()

    print(f"  🔍 OCR: scanned {num_samples} frames, text in {frames_with_text}, {len(all_y_tops)} detections")

    if not all_y_tops or frames_with_text < 2:
        print(f"  ⚠️ OCR: not enough text, using fixed region")
        return {"x_ratio": 0, "y_ratio": 0.78, "w_ratio": 1.0, "h_ratio": 0.18, "method": "fixed_fallback"}

    all_y_tops.sort()
    all_y_bottoms.sort()

    n = len(all_y_tops)
    idx_25 = max(0, int(n * 0.25))
    idx_75 = min(n - 1, int(n * 0.75))

    tight_y_top = all_y_tops[idx_25]
    tight_y_bottom = all_y_bottoms[idx_75]

    pad = 10
    y_start = max(0, tight_y_top - pad)
    y_end = min(height, tight_y_bottom + pad)

    if y_end - y_start < 30:
        y_start = max(0, y_start - 15)
        y_end = min(height, y_end + 15)

    y_ratio = y_start / height
    h_ratio = (y_end - y_start) / height

    print(f"  🔍 OCR detected hardsub: y={y_start}-{y_end} ({y_ratio:.2f}-{y_ratio + h_ratio:.2f})")

    return {
        "x_ratio": 0,
        "y_ratio": y_ratio,
        "w_ratio": 1.0,
        "h_ratio": h_ratio,
        "method": "ocr_detected",
        "samples": len(all_y_tops),
        "frames_with_text": frames_with_text,
    }


def _srt_to_ass(srt_content: str, play_res_x: int, play_res_y: int,
                sub_region: dict) -> str:
    """Convert SRT to ASS with style positioned inside the sub_region.
    sub_region: {x_ratio, y_ratio, w_ratio, h_ratio} in video coordinates."""
    from services.srt_utils import parse_srt

    entries = parse_srt(srt_content)

    # Convert ratios to pixels
    sub_x = int(play_res_x * sub_region.get("x_ratio", 0))
    sub_y = int(play_res_y * sub_region["y_ratio"])
    sub_w = int(play_res_x * sub_region.get("w_ratio", 1.0))
    sub_h = int(play_res_y * sub_region["h_ratio"])

    # Font size proportional to sub region height
    font_size = max(22, min(72, int(sub_h * 0.45)))

    # Alignment=8 (top-center), MarginV pushes down from top
    margin_top = sub_y + int(sub_h * 0.15)

    # Left/right margins to keep text within sub region
    margin_l = max(sub_x, int(play_res_x * 0.03))
    margin_r = max(play_res_x - sub_x - sub_w, int(play_res_x * 0.03))

    ass_header = f"""[Script Info]
Title: Burned Subtitle
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BurnSub,Arial,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0.5,0,1,3,1.5,8,{margin_l},{margin_r},{margin_top},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    ass_lines = [ass_header.strip()]

    for srt_idx, timestamp, text in entries:
        parts = timestamp.split(' --> ')
        if len(parts) != 2:
            continue
        start_t = parts[0].strip().replace(',', '.')
        end_t = parts[1].strip().replace(',', '.')

        # ASS uses centiseconds: H:MM:SS.CC
        start_ass = start_t[:-1] if len(start_t) > 10 else start_t
        end_ass = end_t[:-1] if len(end_t) > 10 else end_t

        clean_text = text.replace('\n', '\\N')
        ass_lines.append(
            f"Dialogue: 0,{start_ass},{end_ass},BurnSub,,0,0,0,,{clean_text}"
        )

    return '\n'.join(ass_lines) + '\n'


def _build_blur_filter(vw: int, vh: int, blur_regions: list,
                        ass_escaped: str) -> str:
    """Build FFmpeg filter_complex for multi-region blur + ASS subtitle.

    Args:
        vw, vh: video dimensions
        blur_regions: list of {x_ratio, y_ratio, w_ratio, h_ratio}
        ass_escaped: escaped path to ASS subtitle file

    Returns:
        filter_complex string for FFmpeg
    """
    n = len(blur_regions)

    if n == 0:
        # No blur, just subtitle
        return f"[0:v]ass='{ass_escaped}'[vout]"

    # Need n+1 splits: 1 main + n blur sources
    split_count = n + 1
    split_labels = "[main]" + "".join(f"[b{i}]" for i in range(n))

    parts = [f"[0:v]split={split_count}{split_labels};"]

    for i, region in enumerate(blur_regions):
        rx = int(vw * region.get("x_ratio", 0))
        ry = int(vh * region["y_ratio"])
        rw = int(vw * region.get("w_ratio", 1.0))
        rh = int(vh * region["h_ratio"])

        # Clamp to valid dimensions
        rx = max(0, min(rx, vw - 2))
        ry = max(0, min(ry, vh - 2))
        rw = max(2, min(rw, vw - rx))
        rh = max(2, min(rh, vh - ry))

        max_r = max(1, min(rw, rh) // 2 - 1)
        lr = min(25, max_r)
        cr = max(1, lr // 2)
        parts.append(
            f"[b{i}]crop={rw}:{rh}:{rx}:{ry},"
            f"boxblur=luma_radius={lr}:luma_power=3:chroma_radius={cr}:chroma_power=3"
            f"[blur{i}];"
        )
    prev = "[main]"
    for i in range(n):
        rx = int(vw * blur_regions[i].get("x_ratio", 0))
        ry = int(vh * blur_regions[i]["y_ratio"])
        rx = max(0, min(rx, vw - 2))
        ry = max(0, min(ry, vh - 2))

        if i == n - 1:
            # Last overlay: also apply ASS, output [vout]
            parts.append(
                f"{prev}[blur{i}]overlay={rx}:{ry},"
                f"ass='{ass_escaped}'[vout]"
            )
        else:
            out_label = f"[tmp{i}]"
            parts.append(f"{prev}[blur{i}]overlay={rx}:{ry}{out_label};")
            prev = out_label

    return "".join(parts)


def burn_sub_video(job_id: str, lang: str, srt_content: str,
                    sub_region: dict = None, extra_regions: list = None):
    """Burn translated subtitle into video with multi-region Gaussian blur.

    Args:
        job_id: job identifier
        lang: target language code
        srt_content: SRT subtitle content
        sub_region: subtitle region {x_ratio, y_ratio, w_ratio, h_ratio} or None for OCR
        extra_regions: list of additional blur regions (logos, watermarks, etc.)
    """
    burn_key = f"burn_{lang}"

    jobs[job_id][burn_key] = {
        "status": "processing",
        "progress": 10,
        "message": "Đang chuẩn bị...",
    }

    video_file = jobs[job_id].get("video_file", {})
    video_path = video_file.get("path", "")

    if not video_path or not os.path.exists(video_path):
        raise Exception("Không tìm thấy video gốc")

    # Determine subtitle region
    if sub_region:
        method = "manual"
        y_r = sub_region["y_ratio"]
        h_r = sub_region["h_ratio"]
        jobs[job_id][burn_key]["message"] = f"📐 Vùng sub: thủ công ({y_r:.0%}-{y_r + h_r:.0%})"
        jobs[job_id][burn_key]["progress"] = 40
    else:
        jobs[job_id][burn_key]["message"] = "🔍 Đang phát hiện vùng hardsub cũ..."
        jobs[job_id][burn_key]["progress"] = 20

        detected = detect_hardsub_region(video_path, job_id)
        sub_region = {
            "x_ratio": detected.get("x_ratio", 0),
            "y_ratio": detected["y_ratio"],
            "w_ratio": detected.get("w_ratio", 1.0),
            "h_ratio": detected["h_ratio"],
        }
        method = detected.get("method", "ocr")

        y_r = sub_region["y_ratio"]
        h_r = sub_region["h_ratio"]
        jobs[job_id][burn_key]["message"] = f"📐 Vùng sub: {method} ({y_r:.0%}-{y_r + h_r:.0%})"
        jobs[job_id][burn_key]["progress"] = 40

    if not extra_regions:
        extra_regions = []

    # Build combined blur regions list (sub_region first, then extras)
    all_blur_regions = [sub_region] + extra_regions[:MAX_BLUR_REGIONS - 1]

    # Get video dimensions and duration
    probe_cmd = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height:format=duration',
         '-print_format', 'json', video_path],
        capture_output=True, text=True, timeout=30
    )
    probe_data = json.loads(probe_cmd.stdout)
    stream = probe_data.get("streams", [{}])[0] if probe_data.get("streams") else {}
    vw = int(stream.get("width", 1080))
    vh = int(stream.get("height", 1920))
    total_duration = float(probe_data.get("format", {}).get("duration", 0))

    output_path = str(OUTPUT_FOLDER / f"{job_id}_burned_{lang}.mp4")

    jobs[job_id][burn_key]["message"] = "🎬 Đang tạo phụ đề ASS..."
    jobs[job_id][burn_key]["progress"] = 45

    # Generate ASS subtitle file
    ass_content = _srt_to_ass(srt_content, vw, vh, sub_region)
    ass_path = OUTPUT_FOLDER / f"{job_id}_burn_{lang}.ass"
    ass_path.write_text(ass_content, encoding='utf-8-sig')

    region_count = len(all_blur_regions)
    print(f"🎬 ASS created: {vw}x{vh}, {region_count} blur region(s), method={method}")

    jobs[job_id][burn_key]["message"] = f"🎬 Đang burn ({region_count} vùng blur)..."
    jobs[job_id][burn_key]["progress"] = 50

    # Escape path for FFmpeg
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")

    # Check for TTS audio
    tts_key = f"tts_{lang}"
    tts_info = jobs[job_id].get(tts_key, {})
    tts_path = tts_info.get("path", "") if tts_info.get("status") == "done" else ""
    has_tts = tts_path and os.path.exists(tts_path)

    # Build filter complex
    filter_complex = _build_blur_filter(vw, vh, all_blur_regions, ass_escaped)

    # Encoder
    if DEVICE == "cuda":
        video_codec = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '22', '-b:v', '0']
    else:
        video_codec = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20']

    # Audio mapping
    if has_tts:
        audio_map = ['-map', '[vout]', '-map', '1:a']
    else:
        audio_map = ['-map', '[vout]', '-map', '0:a', '-c:a', 'copy']

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        *(['-i', tts_path] if has_tts else []),
        '-filter_complex', filter_complex,
        *audio_map,
        *video_codec,
        '-shortest',
        output_path,
    ]

    print(f"🎬 FFmpeg: {region_count} blur regions, video={vw}x{vh}, tts={has_tts}")

    # Redirect stderr to log file to avoid PIPE deadlock
    stderr_log = str(OUTPUT_FOLDER / f"{job_id}_burn_{lang}_ffmpeg.log")
    with open(stderr_log, 'w', encoding='utf-8') as log_f:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
        jobs[job_id]["_ffmpeg_process"] = process

        while process.poll() is None:
            if jobs.get(job_id, {}).get("cancel"):
                process.terminate()
                raise Exception("Đã hủy (Stop)")

            try:
                with open(stderr_log, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 2000))
                    last_output = f.read()

                    matches = re.findall(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", last_output)
                    if matches and total_duration > 0:
                        h, m, s = matches[-1]
                        current_sec = int(h) * 3600 + int(m) * 60 + float(s)

                        pct = int((current_sec / total_duration) * 100)
                        prog = 50 + int((current_sec / total_duration) * 40)
                        jobs[job_id][burn_key]["progress"] = min(prog, 90)

                        speed_match = re.findall(r"speed=\s*([\d\.]+)x", last_output)
                        speed_str = f" ({speed_match[-1]}x)" if speed_match else ""

                        jobs[job_id][burn_key]["message"] = f"🎬 Đang burn: {pct}%{speed_str}"
            except Exception:
                pass

            time.sleep(1)

    if process.returncode != 0:
        err = ""
        try:
            with open(stderr_log, 'r', encoding='utf-8', errors='replace') as f:
                err = f.read()[-500:]
        except:
            pass
        raise Exception(f"FFmpeg error: {err}")

    jobs[job_id][burn_key]["progress"] = 90

    if not os.path.exists(output_path):
        raise Exception("Output video not created")

    file_size = os.path.getsize(output_path)

    dur_cmd = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-print_format', 'json', output_path],
        capture_output=True, text=True, timeout=30
    )
    duration = 0
    try:
        duration = float(json.loads(dur_cmd.stdout)["format"]["duration"])
    except:
        pass

    audio_label = "TTS" if has_tts else "gốc"
    region_label = f"{region_count} vùng blur"
    print(f"✅ Burn [{lang}]: {file_size / 1048576:.1f}MB, {duration:.0f}s, {region_label}, audio={audio_label}")

    jobs[job_id][burn_key] = {
        "status": "done",
        "progress": 100,
        "message": f"Hoàn thành ({file_size / 1048576:.1f}MB) • {region_label} • Audio: {audio_label}",
        "path": output_path,
        "filename": f"{jobs[job_id].get('original_name', 'video')[:30]}_{lang}_sub.mp4",
        "size": file_size,
        "duration": round(duration, 1),
        "method": method,
        "audio_replaced": has_tts,
        "blur_regions": len(all_blur_regions),
    }


def burnsub_worker(job_id: str, lang: str, srt_content: str,
                    sub_region: dict = None, extra_regions: list = None):
    """Background worker for burn subtitle"""
    burn_key = f"burn_{lang}"
    try:
        burn_sub_video(job_id, lang, srt_content, sub_region, extra_regions)
    except Exception as e:
        if jobs.get(job_id):
            jobs[job_id][burn_key] = {
                "status": "error",
                "progress": 0,
                "message": f"Lỗi burn sub: {str(e)}",
            }
        import traceback
        print(f"❌ Burn sub [{lang}] failed: {e}")
        traceback.print_exc()
