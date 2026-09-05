"""
burn_sub.py — Burn translated subtitles into video with tight bounding box and feathered edge blur.
Supports multiple blur regions (subtitle area + extra areas like logos/watermarks).
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

from config import DEVICE, MAX_BLUR_REGIONS, OUTPUT_FOLDER, jobs


def create_feather_mask(
    mask_path: str,
    width: int,
    height: int,
    fade_x: int = 24,
    fade_y: int = 14,
) -> str:
    """Generate a smooth alpha gradient mask PNG for seamless feathered blending."""
    import cv2
    import numpy as np

    width = max(4, int(width))
    height = max(4, int(height))
    mask = np.ones((height, width), dtype=np.float32)

    fade_x = max(1, min(int(fade_x), width // 4))
    fade_y = max(1, min(int(fade_y), height // 4))

    # Cosine smoothstep curve for natural feathered edges
    for y in range(fade_y):
        val = 0.5 * (1.0 - np.cos(np.pi * (y / float(fade_y))))
        mask[y, :] *= val
        mask[height - 1 - y, :] *= val

    for x in range(fade_x):
        val = 0.5 * (1.0 - np.cos(np.pi * (x / float(fade_x))))
        mask[:, x] *= val
        mask[:, width - 1 - x] *= val

    mask_uint8 = (mask * 255.0).astype(np.uint8)
    cv2.imwrite(mask_path, mask_uint8)
    return mask_path


def detect_hardsub_region(video_path: str, job_id: str = None, srt_content: str = None) -> dict:
    """Detect hardcoded subtitle region using SRT-guided sampling & OCR geometry.
    Returns region: {x_ratio, y_ratio, w_ratio, h_ratio, method}"""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"x_ratio": 0.08, "y_ratio": 0.86, "w_ratio": 0.84, "h_ratio": 0.09, "method": "fixed_fallback"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    if total_frames < 10 or height < 100:
        cap.release()
        return {"x_ratio": 0.08, "y_ratio": 0.86, "w_ratio": 0.84, "h_ratio": 0.09, "method": "fixed_fallback"}

    # 1. Smart sampling: If SRT content is provided, sample directly from dialogue timestamps!
    sample_frames = []
    if srt_content:
        from services.srt_utils import parse_srt_timing
        timings = parse_srt_timing(srt_content)
        if timings:
            step = max(1, len(timings) // 12)
            picked = timings[::step][:15]
            for s_ms, e_ms, _ in picked:
                mid_s = (s_ms + e_ms) / 2000.0
                f_no = min(int(mid_s * fps), total_frames - 1)
                if f_no not in sample_frames:
                    sample_frames.append(f_no)

    # If no SRT, sample evenly across the video duration
    if not sample_frames:
        num_samples = 15
        sample_times = [min(2.0, duration * 0.1) + duration * 0.8 * i / (num_samples - 1) for i in range(num_samples)]
        sample_frames = [min(int(t * fps), total_frames - 1) for t in sample_times]

    try:
        import easyocr
        reader = easyocr.Reader(['ch_sim'], gpu=True, verbose=False)
    except Exception:
        cap.release()
        return {"x_ratio": 0.08, "y_ratio": 0.86, "w_ratio": 0.84, "h_ratio": 0.09, "method": "fixed_fallback"}

    if job_id:
        burn_key = [k for k in jobs[job_id] if k.startswith("burn_") and isinstance(jobs[job_id][k], dict)]
        if burn_key:
            jobs[job_id][burn_key[0]]["message"] = f"🔍 Đang quét {len(sample_frames)} frame tìm hardsub..."

    all_y_mins = []
    all_y_maxs = []
    all_x_mins = []
    all_x_maxs = []
    all_heights = []
    frames_with_text = 0

    crop_start = int(height * 0.60)
    crop_end = int(height * 0.98)

    for i, frame_pos in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            continue

        bottom_crop = frame[crop_start:crop_end, :]

        try:
            results = reader.readtext(bottom_crop, paragraph=False)
            frame_has_text = False
            for (bbox, text, conf) in results:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys) + crop_start, max(ys) + crop_start
                h = y_max - y_min
                w = x_max - x_min

                # Subtitle criteria: reasonable line height, wide enough, in lower portion
                # Note: We do NOT filter by conf > 0.40 because styled Chinese text often has low recognition score but perfect geometry!
                if h >= 14 and w >= width * 0.06 and y_min >= height * 0.60:
                    all_y_mins.append(y_min)
                    all_y_maxs.append(y_max)
                    all_x_mins.append(x_min)
                    all_x_maxs.append(x_max)
                    all_heights.append(h)
                    frame_has_text = True
            if frame_has_text:
                frames_with_text += 1
        except Exception:
            continue

        if job_id and burn_key:
            jobs[job_id][burn_key[0]]["message"] = f"🔍 Quét frame {i + 1}/{len(sample_frames)}..."

    cap.release()

    print(f"  🔍 OCR: scanned {len(sample_frames)} frames, text in {frames_with_text}, {len(all_y_mins)} detections")

    if not all_y_mins or frames_with_text < 2:
        print("  ⚠️ OCR: not enough text detected, using fallback region")
        return {
            "x_ratio": 0.08,
            "y_ratio": 0.86,
            "w_ratio": 0.84,
            "h_ratio": 0.09,
            "method": "fixed_fallback",
        }

    # Vertical (Y) robust median bounding
    med_ymin = int(np.median(all_y_mins))
    med_ymax = int(np.median(all_y_maxs))
    med_h = med_ymax - med_ymin

    pad_y = max(8, int(med_h * 0.22))
    y_start = max(0, med_ymin - pad_y)
    y_end = min(height, med_ymax + pad_y)

    sub_h = y_end - y_start
    sub_h = max(int(height * 0.05), min(sub_h, int(height * 0.12)))
    y_ratio = round(y_start / height, 3)
    h_ratio = round(sub_h / height, 3)

    # Horizontal (X) robust centering
    detected_w = np.percentile(all_x_maxs, 85) - np.percentile(all_x_mins, 15)
    target_w = max(int(width * 0.60), min(int(detected_w + width * 0.08), int(width * 0.88)))
    target_w -= (target_w % 2)
    x_start = max(0, int((width - target_w) / 2))
    x_ratio = round(x_start / width, 3)
    w_ratio = round(target_w / width, 3)

    print(f"  🔍 Smart OCR detected hardsub: x={x_ratio}-{x_ratio+w_ratio:.2f}, y={y_ratio}-{y_ratio+h_ratio:.2f} (w={w_ratio:.2f}, h={h_ratio:.2f}, lines={med_ymin}-{med_ymax})")

    return {
        "x_ratio": x_ratio,
        "y_ratio": y_ratio,
        "w_ratio": w_ratio,
        "h_ratio": h_ratio,
        "method": "smart_ocr_detected",
        "samples": len(all_y_mins),
        "frames_with_text": frames_with_text,
    }


def _srt_to_ass(
    srt_content: str,
    play_res_x: int,
    play_res_y: int,
    sub_region: dict,
    extra_ass_styles: list[str] | None = None,
    extra_ass_events: list[str] | None = None,
) -> str:
    """Convert SRT to ASS with styling fitted cleanly inside the tight sub_region."""
    from services.srt_utils import parse_srt

    entries = parse_srt(srt_content)

    sub_x = int(play_res_x * sub_region.get("x_ratio", 0.08))
    sub_y = int(play_res_y * sub_region.get("y_ratio", 0.86))
    sub_w = int(play_res_x * sub_region.get("w_ratio", 0.84))
    sub_h = int(play_res_y * sub_region.get("h_ratio", 0.09))

    # Standard readable font size (proportional to resolution)
    font_size = max(20, min(56, int(play_res_y * 0.038)))

    # Top margin vertically centered inside sub_region
    margin_top = sub_y + max(2, int((sub_h - font_size) * 0.45))

    # Left and right margins
    margin_l = max(int(play_res_x * 0.04), sub_x)
    margin_r = max(int(play_res_x * 0.04), play_res_x - sub_x - sub_w)

    styles_list = [
        f"Style: BurnSub,Arial,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0.5,0,1,3.0,1.2,8,{margin_l},{margin_r},{margin_top},1"
    ]
    if extra_ass_styles:
        styles_list.extend(extra_ass_styles)
    styles_block = "\n".join(styles_list)

    ass_header = f"""[Script Info]
Title: Burned Subtitle
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_block}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    ass_lines = [ass_header.strip()]
    if extra_ass_events:
        ass_lines.extend(extra_ass_events)

    for _srt_idx, timestamp, text in entries:
        parts = timestamp.split(' --> ')
        if len(parts) != 2:
            continue
        start_t = parts[0].strip().replace(',', '.')
        end_t = parts[1].strip().replace(',', '.')

        start_ass = start_t[:-1] if len(start_t) > 10 else start_t
        end_ass = end_t[:-1] if len(end_t) > 10 else end_t

        clean = text.replace("\r", "").strip()
        # Smart line balancing: wrap into 2 lines if long
        if "\n" not in clean and len(clean) > 36:
            words = clean.split()
            if len(words) >= 3:
                mid = len(clean) // 2
                best_i = 0
                min_dist = 999
                curr = 0
                for w_idx in range(len(words) - 1):
                    curr += len(words[w_idx]) + 1
                    if abs(curr - mid) < min_dist:
                        min_dist = abs(curr - mid)
                        best_i = w_idx
                clean = " ".join(words[:best_i + 1]) + r"\N" + " ".join(words[best_i + 1:])
            else:
                clean = clean.replace("\n", r"\N")
        else:
            clean = clean.replace("\n", r"\N")

        ass_lines.append(
            f"Dialogue: 0,{start_ass},{end_ass},BurnSub,,0,0,0,,{clean}"
        )

    return '\n'.join(ass_lines) + '\n\n\n'
def extract_subtitle_intervals(
    srt_content: str,
    min_gap: float = 1.20,
    pad_start: float = 0.15,
    pad_end: float = 0.18,
) -> list[tuple[float, float]]:
    """Extract and merge subtitle time intervals in seconds from SRT content."""
    from services.srt_utils import parse_srt_timing

    raw_timings = parse_srt_timing(srt_content)
    if not raw_timings:
        return []

    intervals = []
    for start_ms, end_ms, _ in raw_timings:
        s = max(0.0, (start_ms / 1000.0) - pad_start)
        e = max(s + 0.1, (end_ms / 1000.0) + pad_end)
        intervals.append((s, e))

    intervals.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if not merged:
            merged.append((s, e))
        else:
            prev_s, prev_e = merged[-1]
            if s <= prev_e + min_gap:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
    return [(round(s, 3), round(e, 3)) for s, e in merged]


def build_timeline_enable_expression(intervals: list[tuple[float, float]]) -> str:
    """Build FFmpeg enable='between(t,s,e)+...' expression."""
    if not intervals:
        return ""
    clauses = [f"between(t,{s:.3f},{e:.3f})" for s, e in intervals]
    return "+".join(clauses)


def _build_blur_filter(
    vw: int,
    vh: int,
    blur_regions: list,
    ass_escaped: str,
    mask_input_start: int = 1,
    timeline_enable: str = "",
) -> str:
    """Build FFmpeg filter_complex with feathered alpha masks + ASS subtitle.

    Args:
        vw, vh: video dimensions
        blur_regions: list of {x_ratio, y_ratio, w_ratio, h_ratio}
        ass_escaped: escaped path to ASS subtitle file
        mask_input_start: index of first mask input in FFmpeg cmd
    """
    n = len(blur_regions)
    if n == 0:
        return f"[0:v]ass='{ass_escaped}'[vout]"

    split_count = n + 1
    split_labels = "[main]" + "".join(f"[b{i}]" for i in range(n))
    parts = [f"[0:v]split={split_count}{split_labels};"]

    for i, region in enumerate(blur_regions):
        rx = int(vw * region.get("x_ratio", 0.08))
        ry = int(vh * region["y_ratio"])
        rw = int(vw * region.get("w_ratio", 0.84))
        rh = int(vh * region["h_ratio"])

        # Clamp and make even
        rx = max(0, min(rx, vw - 4))
        ry = max(0, min(ry, vh - 4))
        rw = max(4, min(rw, vw - rx))
        rh = max(4, min(rh, vh - ry))
        rw -= (rw % 2)
        rh -= (rh % 2)
        rx -= (rx % 2)
        ry -= (ry % 2)

        max_r = max(1, min(rw, rh) // 2 - 1)
        lr = min(20, max_r)
        cr = max(1, lr // 2)
        mask_idx = mask_input_start + i

        parts.append(
            f"[b{i}]crop={rw}:{rh}:{rx}:{ry},"
            f"boxblur=luma_radius={lr}:luma_power=2:chroma_radius={cr}:chroma_power=2[b_raw{i}];"
            f"[b_raw{i}]format=rgba[b_rgba{i}];"
            f"[{mask_idx}:v]format=gray[m_gray{i}];"
            f"[b_rgba{i}][m_gray{i}]alphamerge[feathered{i}];"
        )

    prev = "[main]"
    for i in range(n):
        rx = int(vw * blur_regions[i].get("x_ratio", 0.08))
        ry = int(vh * blur_regions[i]["y_ratio"])
        rx = max(0, min(rx, vw - 4))
        ry = max(0, min(ry, vh - 4))
        rx -= (rx % 2)
        ry -= (ry % 2)

        # Region 0 (main subtitle) only activates during speech intervals
        enable_clause = f":enable='{timeline_enable}'" if (i == 0 and timeline_enable) else ""

        if i == n - 1:
            parts.append(f"{prev}[feathered{i}]overlay={rx}:{ry}{enable_clause}:shortest=1,ass='{ass_escaped}'[vout]")
        else:
            out_label = f"[tmp{i}]"
            parts.append(f"{prev}[feathered{i}]overlay={rx}:{ry}{enable_clause}:shortest=1{out_label};")
            prev = out_label

    return "".join(parts)


def burn_sub_video(
    job_id: str,
    lang: str,
    srt_content: str,
    sub_region: dict = None,
    extra_regions: list = None,
    render_mode: str = "blur",
    inpaint_engine: str = "opencv",
    trim_intro: str = "off",
    translate_title: bool = False,
    title_lang: str = "vi",
    brand_name: str = "",
    bgm_mode: str = "auto",
    bgm_volume: float = 0.8,
    clean_hardsub: bool = True,
    clean_logo: bool = False,
    clean_title: bool = False,
    burn_new_sub: bool = True,
):
    """Burn translated subtitle into video with tight bounding box and feathered edge blur."""
    burn_key = f"burn_{lang}"

    jobs[job_id][burn_key] = {
        "status": "processing",
        "progress": 10,
        "message": "Đang chuẩn bị...",
    }

    video_file = jobs[job_id].get("video_file")
    video_path = video_file.get("path", "") if isinstance(video_file, dict) else ""
    if not video_path:
        video_path = str(jobs[job_id].get("video_path", ""))

    if not video_path or not os.path.exists(video_path):
        raise RuntimeError("Không tìm thấy video gốc")

    # Get video dimensions & duration
    probe_cmd = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height:format=duration',
            '-print_format', 'json',
            video_path,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    vw, vh = 720, 1280
    total_duration = 0
    try:
        probe_data = json.loads(probe_cmd.stdout)
        streams = probe_data.get("streams", [{}])
        if streams:
            vw = int(streams[0].get("width", 720))
            vh = int(streams[0].get("height", 1280))
        fmt = probe_data.get("format", {})
        total_duration = float(fmt.get("duration", 0))
    except Exception as exc:
        print(f"ffprobe warning: {exc}, using fallback size {vw}x{vh}")

    if total_duration == 0:
        total_duration = float(jobs[job_id].get("duration", 60) or 60)

    # Step 1: Intro cover trimming if requested (skip if already trimmed at ingest stage)
    trimmed_video_path = None
    already_trimmed = float(jobs.get(job_id, {}).get("trimmed_seconds", 0) or 0) > 0.05
    if not already_trimmed and trim_intro and trim_intro != "off":
        from services.video.trimmer import detect_intro_cover, trim_video_and_shift_srt
        trim_sec = 0.0
        if trim_intro == "auto":
            jobs[job_id][burn_key]["message"] = "✂️ Đang phát hiện bìa/intro đầu video..."
            trim_sec = detect_intro_cover(video_path, srt_content=srt_content)
        else:
            try:
                trim_sec = float(trim_intro)
            except ValueError:
                trim_sec = 0.0

        if trim_sec > 0.05:
            jobs[job_id][burn_key]["message"] = f"✂️ Đang cắt bỏ {trim_sec:.2f}s bìa đầu video..."
            trim_out = str(OUTPUT_FOLDER / f"{job_id}_trimmed.mp4")
            video_path, shifted_srt = trim_video_and_shift_srt(video_path, trim_sec, trim_out, srt_content)
            trimmed_video_path = trim_out
            if shifted_srt:
                srt_content = shifted_srt

    # Build modular regions based on user choices:
    # 1. Hardsub region (bottom dialogue subtitles)
    method = "manual"
    if not (sub_region and "y_ratio" in sub_region):
        jobs[job_id][burn_key]["message"] = "🔍 Đang quét vị trí hardsub..."
        jobs[job_id][burn_key]["progress"] = 20
        sub_region = detect_hardsub_region(video_path, job_id, srt_content=srt_content)
        method = sub_region.get("method", "ocr_detected")

    all_blur_regions = []
    if clean_hardsub:
        all_blur_regions.append(sub_region)

    # 2. Logo / Watermark regions (corners)
    if clean_logo:
        if extra_regions:
            for r in extra_regions[:MAX_BLUR_REGIONS - 1]:
                if isinstance(r, dict) and "y_ratio" in r:
                    all_blur_regions.append(r)
        else:
            all_blur_regions.append({"x_ratio": 0.65, "y_ratio": 0.01, "w_ratio": 0.34, "h_ratio": 0.055})

    extra_ass_styles = []
    extra_ass_events = []

    # 3. Top Title Banner (detection, inpainting, and translation)
    if clean_title or translate_title:
        from services.video.title_detector import detect_top_title, translate_title, generate_title_ass_style_and_event
        jobs[job_id][burn_key]["message"] = "🏷️ Đang quét tiêu đề trên video..."
        top_title = detect_top_title(video_path)
        if top_title and top_title.get("text"):
            all_blur_regions.append({
                "x_ratio": top_title["x_ratio"],
                "y_ratio": top_title["y_ratio"],
                "w_ratio": top_title["w_ratio"],
                "h_ratio": top_title["h_ratio"],
            })
            if translate_title:
                translated_title = translate_title(top_title["text"], target_lang=title_lang)
                if translated_title:
                    t_style, t_event = generate_title_ass_style_and_event(
                        translated_title, vw, vh, y_ratio=top_title["y_ratio"], duration_sec=min(8.0, total_duration)
                    )
                    extra_ass_styles.append(t_style)
                    extra_ass_events.append(t_event)

    # 4. Brand Watermark
    if brand_name and str(brand_name).strip():
        brand_clean = str(brand_name).strip().replace("\n", "").replace("\\", "")
        b_font_size = max(18, min(36, int(vh * 0.026)))
        b_style = f"Style: BrandWatermark,Arial,{b_font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H70000000,-1,0,0,0,100,100,0,0,1,2.0,1.0,7,30,30,25,1"
        b_event = f"Dialogue: 2,0:00:00.00,9:59:59.00,BrandWatermark,,0,0,0,,{brand_clean}"
        extra_ass_styles.append(b_style)
        extra_ass_events.append(b_event)

    # Check TTS voiceover & BGM mixing
    tts_key = f"tts_{lang}"
    tts_info = jobs[job_id].get(tts_key, {})
    tts_path = tts_info.get("path", "") if tts_info.get("status") == "done" else ""
    has_tts = tts_path and os.path.exists(tts_path)

    audio_to_mux = tts_path
    if has_tts and bgm_mode != "none":
        from services.audio_mixer import mix_voiceover_with_bgm
        jobs[job_id][burn_key]["message"] = "🎵 Đang xử lý và trộn nhạc nền (BGM)..."
        mixed_audio_file = str(OUTPUT_FOLDER / job_id / f"final_mixed_{lang}.m4a")
        try:
            audio_to_mux = mix_voiceover_with_bgm(
                video_path=video_path,
                tts_audio_path=tts_path,
                output_audio_path=mixed_audio_file,
                job_id=job_id,
                bgm_mode=bgm_mode,
                bgm_volume=bgm_volume,
            )
        except Exception as exc:
            print(f"Warning: BGM mix failed ({exc}), using raw voiceover")
            audio_to_mux = tts_path

    # Route to AI Inpainting if requested (Clean Plate or Inpaint & Re-burn)
    if render_mode in ("clean", "inpaint_burn"):

        output_dir = OUTPUT_FOLDER / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_tag = "clean" if render_mode == "clean" else f"{lang}_sub"
        out_file = str(output_dir / f"{jobs[job_id].get('original_name', 'video')[:30]}_{clean_tag}.mp4")

        re_burn_ass = None
        if render_mode == "inpaint_burn":
            ass_content = _srt_to_ass(
                srt_content, vw, vh, sub_region,
                extra_ass_styles=extra_ass_styles,
                extra_ass_events=extra_ass_events,
            )
            ass_path = OUTPUT_FOLDER / f"{job_id}_burn_{lang}.ass"
            ass_path.write_text(ass_content, encoding="utf-8-sig")
            re_burn_ass = str(ass_path)

        extra_inpaint = all_blur_regions[1:] if len(all_blur_regions) > 1 else None

        from services.video.clean_pipeline import clean_video_pipeline
        return clean_video_pipeline(
            video_path=video_path,
            sub_region=sub_region,
            srt_content=srt_content,
            output_path=out_file,
            job_id=job_id,
            burn_key=burn_key,
            engine=inpaint_engine,
            re_burn_ass_path=re_burn_ass,
            tts_audio_path=audio_to_mux if has_tts else None,
            extra_regions=extra_inpaint,
        )

    # Output path
    output_dir = OUTPUT_FOLDER / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"burned_{lang}.mp4")

    jobs[job_id][burn_key]["message"] = "🎬 Đang tạo phụ đề ASS..."
    jobs[job_id][burn_key]["progress"] = 45

    ass_content = _srt_to_ass(
        srt_content, vw, vh, sub_region,
        extra_ass_styles=extra_ass_styles,
        extra_ass_events=extra_ass_events,
    )
    ass_path = OUTPUT_FOLDER / f"{job_id}_burn_{lang}.ass"
    ass_path.write_text(ass_content, encoding='utf-8-sig')

    region_count = len(all_blur_regions)
    print(f"🎬 ASS created: {vw}x{vh}, {region_count} blur region(s), method={method}")

    # Handle Pure Burn mode (without blur background)
    if render_mode == "pure_burn":
        ass_escaped = str(ass_path).replace(chr(92), "/").replace(":", r"\:")
        video_codec = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '22', '-b:v', '0'] if DEVICE == "cuda" else ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20']
        if has_tts:
            fc = f"[0:v]ass='{ass_escaped}'[vout];[1:a]apad[aout]"
            amap = ['-map', '[vout]', '-map', '[aout]', '-c:a', 'aac', '-b:a', '192k']
            cmd_pure = ['ffmpeg', '-y', '-i', video_path, '-i', audio_to_mux, '-filter_complex', fc, *amap, *video_codec, '-shortest', output_path]
        else:
            fc = f"[0:v]ass='{ass_escaped}'[vout]"
            amap = ['-map', '[vout]', '-map', '0:a?', '-c:a', 'copy']
            cmd_pure = ['ffmpeg', '-y', '-i', video_path, '-filter_complex', fc, *amap, *video_codec, output_path]

        jobs[job_id][burn_key]["message"] = "🎬 Đang in phụ đề trực tiếp lên video..."
        jobs[job_id][burn_key]["progress"] = 50

        stderr_log = str(OUTPUT_FOLDER / f"{job_id}_pure_burn_{lang}_ffmpeg.log")
        try:
            with open(stderr_log, 'w', encoding='utf-8') as log_f:
                process = subprocess.Popen(cmd_pure, stdout=subprocess.DEVNULL, stderr=log_f)
                jobs[job_id]["_ffmpeg_process"] = process

                while process.poll() is None:
                    if jobs.get(job_id, {}).get("cancel"):
                        process.terminate()
                        raise RuntimeError("Đã hủy (Stop)")

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
                                prog = 50 + int((current_sec / total_duration) * 45)
                                jobs[job_id][burn_key]["progress"] = min(prog, 95)

                                speed_match = re.findall(r"speed=\s*([\d\.]+)x", last_output)
                                speed_str = f" ({speed_match[-1]}x)" if speed_match else ""
                                jobs[job_id][burn_key]["message"] = (
                                    f"🎬 Đang in sub: {pct}% ({h}:{m}:{int(float(s)):02d}/{int(total_duration//60)}:{int(total_duration%60):02d}){speed_str}"
                                )
                    except Exception:
                        pass
                    time.sleep(1)

            if process.returncode != 0:
                with open(stderr_log, 'r', encoding='utf-8', errors='replace') as f:
                    err = f.read()[-500:]
                raise RuntimeError(f"FFmpeg error: {err or 'Lỗi burn sub trực tiếp'}")
        finally:
            jobs.get(job_id, {}).pop("_ffmpeg_process", None)
            try: ass_path.unlink(missing_ok=True)
            except OSError: pass
            try: Path(stderr_log).unlink(missing_ok=True)
            except OSError: pass

        if not os.path.exists(output_path):
            raise RuntimeError("Output video not created")
        file_size = os.path.getsize(output_path)
        jobs[job_id][burn_key] = {
            "status": "done",
            "progress": 100,
            "message": f"Hoàn thành ({file_size / 1048576:.1f}MB) • In sub mới trực tiếp",
            "path": output_path,
            "filename": f"{jobs[job_id].get('original_name', 'video')[:30]}_{lang}_sub.mp4",
            "size": file_size,
            "duration": round(total_duration, 1),
            "method": "pure_burn",
            "audio_replaced": has_tts,
        }
        return jobs[job_id][burn_key]

    jobs[job_id][burn_key]["message"] = f"🎬 Đang burn ({region_count} vùng blur viền mềm)..."
    jobs[job_id][burn_key]["progress"] = 50

    ass_escaped = str(ass_path).replace(chr(92), "/").replace(":", r"\:")

    tts_key = f"tts_{lang}"
    tts_info = jobs[job_id].get(tts_key, {})
    tts_path = tts_info.get("path", "") if tts_info.get("status") == "done" else ""
    has_tts = tts_path and os.path.exists(tts_path)

    # Generate feathered alpha masks for each region
    mask_files = []
    mask_args = []
    for i, region in enumerate(all_blur_regions):
        rw = max(4, int(vw * region.get("w_ratio", 0.84)))
        rh = max(4, int(vh * region.get("h_ratio", 0.085)))
        rw -= (rw % 2)
        rh -= (rh % 2)
        mask_path = str(OUTPUT_FOLDER / f"{job_id}_mask_{lang}_{i}.png")
        create_feather_mask(mask_path, rw, rh, fade_x=min(24, rw // 6), fade_y=min(14, rh // 4))
        mask_files.append(mask_path)
        mask_args.extend(["-loop", "1", "-i", mask_path])

    mask_start_idx = 2 if has_tts else 1
    intervals = extract_subtitle_intervals(srt_content)
    timeline_enable = build_timeline_enable_expression(intervals)
    filter_complex = _build_blur_filter(
        vw, vh, all_blur_regions, ass_escaped,
        mask_input_start=mask_start_idx,
        timeline_enable=timeline_enable,
    )
    if has_tts:
        filter_complex += ";[1:a]apad[aout]" 

    filter_script_path = OUTPUT_FOLDER / f"{job_id}_filter_{lang}.txt"
    filter_script_path.write_text(filter_complex, encoding='utf-8')

    if DEVICE == "cuda":
        video_codec = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '22', '-b:v', '0']
    else:
        video_codec = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20']

    if has_tts:
        audio_map = ['-map', '[vout]', '-map', '[aout]', '-c:a', 'aac', '-b:a', '192k']
    else:
        audio_map = ['-map', '[vout]', '-map', '0:a?', '-c:a', 'copy']

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        *(['-i', audio_to_mux] if has_tts else []),
        *mask_args,
        '-filter_complex_script', str(filter_script_path),
        *audio_map,
        *video_codec,
        '-shortest',
        output_path,
    ]

    print(f"🎬 FFmpeg: {region_count} feathered blur regions, video={vw}x{vh}, tts={has_tts}")

    stderr_log = str(OUTPUT_FOLDER / f"{job_id}_burn_{lang}_ffmpeg.log")
    try:
        with open(stderr_log, 'w', encoding='utf-8') as log_f:
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
            jobs[job_id]["_ffmpeg_process"] = process

            while process.poll() is None:
                if jobs.get(job_id, {}).get("cancel"):
                    process.terminate()
                    raise RuntimeError("Đã hủy (Stop)")

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
            except Exception:
                pass
            raise RuntimeError(f"FFmpeg error: {err}")

        jobs[job_id][burn_key]["progress"] = 90

        if not os.path.exists(output_path):
            raise RuntimeError("Output video not created")

        file_size = os.path.getsize(output_path)

        dur_cmd = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-print_format', 'json', output_path
            ],
            capture_output=True, text=True, timeout=30,
        )
        duration = 0
        try:
            duration = float(json.loads(dur_cmd.stdout)["format"]["duration"])
        except Exception:
            pass

        audio_label = "TTS" if has_tts else "gốc"
        region_label = f"{region_count} vùng blur viền mềm"
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
    finally:
        for mf in mask_files:
            try:
                Path(mf).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            ass_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            if 'filter_script_path' in locals():
                filter_script_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            if 'trimmed_video_path' in locals() and trimmed_video_path:
                Path(trimmed_video_path).unlink(missing_ok=True)
        except OSError:
            pass


def burnsub_worker(
    job_id: str,
    lang: str,
    srt_content: str,
    sub_region: dict = None,
    extra_regions: list = None,
    render_mode: str = "blur",
    inpaint_engine: str = "opencv",
    trim_intro: str = "off",
    translate_title: bool = False,
    title_lang: str = "vi",
    brand_name: str = "",
    bgm_mode: str = "auto",
    bgm_volume: float = 0.8,
    clean_hardsub: bool = True,
    clean_logo: bool = False,
    clean_title: bool = False,
    burn_new_sub: bool = True,
):
    """Background worker for burn subtitle"""
    burn_key = f"burn_{lang}"
    try:
        burn_sub_video(
            job_id, lang, srt_content, sub_region, extra_regions,
            render_mode, inpaint_engine, trim_intro, translate_title, title_lang, brand_name,
            bgm_mode, bgm_volume, clean_hardsub, clean_logo, clean_title, burn_new_sub
        )
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
