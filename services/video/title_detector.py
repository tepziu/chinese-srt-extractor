"""
title_detector.py — Detects, translates, and renders stylized Top Title Cards / Banners.
Supports translation to Vietnamese (vi) and English (en).
"""

from __future__ import annotations

import re
import cv2
import numpy as np

from config import AI_DEFAULT_MODEL, AI_TRANSLATE_CONFIG, LANGUAGES


def detect_top_title(
    video_path: str,
    max_check_sec: float = 6.0,
    num_samples: int = 6,
) -> dict | None:
    """Scan the top 1/3 of the video in the opening seconds to detect major Chinese titles or banners."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    if duration < 2.0:
        cap.release()
        return None

    sample_times = [min(0.5 + i * 1.0, duration * 0.9) for i in range(num_samples)]
    sample_frames = [min(int(t * fps), total_frames - 1) for t in sample_times]

    try:
        import easyocr
        from services.video.trimmer import get_ocr_reader
        reader = get_ocr_reader()
        if reader is None:
            reader = easyocr.Reader(['ch_sim'], gpu=False)
    except Exception:
        cap.release()
        return None

    # Focus on upper region: y = 4% to 35% of frame height
    crop_top = int(height * 0.04)
    crop_bottom = int(height * 0.35)

    candidates = []

    for f_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        top_crop = frame[crop_top:crop_bottom, :]
        results = reader.readtext(top_crop)

        for bbox, text, conf in results:
            clean = text.strip()
            # Title criteria: at least 3 Chinese characters, reasonable text height and width
            if len(clean) >= 3:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys) + crop_top, max(ys) + crop_top
                w = x_max - x_min
                h = y_max - y_min

                if w >= width * 0.15 and h >= 18:
                    candidates.append({
                        "text": clean,
                        "x_min": x_min, "x_max": x_max,
                        "y_min": y_min, "y_max": y_max,
                        "w": w, "h": h,
                    })

    cap.release()

    if not candidates:
        return None

    # Select candidate with largest text area (likely the primary hook title)
    best = max(candidates, key=lambda c: c["w"] * c["h"])

    pad_x = int(width * 0.04)
    pad_y = int(best["h"] * 0.25)

    x_start = max(0, best["x_min"] - pad_x)
    x_end = min(width, best["x_max"] + pad_x)
    y_start = max(0, best["y_min"] - pad_y)
    y_end = min(height, best["y_max"] + pad_y)

    sub_w = x_end - x_start
    sub_h = y_end - y_start

    return {
        "text": best["text"],
        "x_ratio": round(x_start / width, 3),
        "y_ratio": round(y_start / height, 3),
        "w_ratio": round(sub_w / width, 3),
        "h_ratio": round(sub_h / height, 3),
    }


def translate_title(
    title_text: str,
    target_lang: str = "vi",
    ai_model: str | None = None,
) -> str:
    """Translate Chinese video title into punchy uppercase title in target language."""
    if not title_text or not title_text.strip():
        return ""

    target_lang = target_lang if target_lang in ("vi", "en") else "vi"
    lang_name = "tiếng Việt" if target_lang == "vi" else "English"

    if AI_TRANSLATE_CONFIG.get("api_key"):
        try:
            from openai import OpenAI
            model = ai_model or AI_TRANSLATE_CONFIG.get("model") or AI_DEFAULT_MODEL
            client = OpenAI(base_url=AI_TRANSLATE_CONFIG["base_url"], api_key=AI_TRANSLATE_CONFIG["api_key"])

            if target_lang == "en":
                sys_prompt = (
                    "You are an expert viral short video title and hook copywriter (Chinese to English). "
                    "Translate the Chinese video title into a concise, punchy, high-converting UPPERCASE English hook title. "
                    "No quotes, no explanation, UPPERCASE only."
                )
                user_msg = f"Translate this Chinese title into a punchy UPPERCASE English title:\n{title_text}"
            else:
                sys_prompt = (
                    f"Bạn là chuyên gia biên tập giật tít tiêu đề video ngắn (Shorts/TikTok/Reels) từ tiếng Trung sang {lang_name}. "
                    "Hãy dịch tiêu đề video súc tích, hấp dẫn, viết hoa toàn bộ (UPPERCASE), không thêm dấu ngoặc kép, không giải thích."
                )
                user_msg = f"Dịch tiêu đề video sau sang {lang_name}:\n{title_text}"

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            result = resp.choices[0].message.content.strip().strip('"').strip("'")
            if result:
                return result.upper()
        except Exception as exc:
            print(f"Warning: AI title translation failed ({exc}), trying Google fallback")

    # Fallback to Google Translator
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="zh-CN", target=target_lang)
        res = translator.translate(title_text) or title_text
        return res.strip().upper()
    except Exception as exc:
        print(f"Google title translation error: {exc}")
        return title_text


def generate_title_ass_style_and_event(
    translated_title: str,
    play_res_x: int,
    play_res_y: int,
    y_ratio: float = 0.10,
    duration_sec: float = 7.0,
) -> tuple[str, str]:
    """Generate ASS style header and event line for the translated top title banner."""
    font_size = max(24, min(64, int(play_res_y * 0.045)))
    margin_top = max(10, int(play_res_y * y_ratio))

    style_line = (
        f"Style: TopTitle,Arial Black,{font_size},&H0000FFFF,&H0000FFFF,&H00000000,&H90000000,"
        f"-1,0,0,0,100,100,0.5,0,1,3.5,2.0,8,40,40,{margin_top},1"
    )

    clean_text = translated_title.replace("\n", "\\N")
    end_cents = int(duration_sec * 100)
    mins, remainder = divmod(end_cents, 6000)
    secs, cents = divmod(remainder, 100)
    hours, mins = divmod(mins, 60)
    end_ass = f"{hours}:{mins:02d}:{secs:02d}.{cents:02d}"

    event_line = f"Dialogue: 1,0:00:00.00,{end_ass},TopTitle,,0,0,0,,{clean_text}"
    return style_line, event_line
