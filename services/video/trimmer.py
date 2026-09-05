"""
trimmer.py — Auto-detection and trimming of intro covers, freeze-frames, and title cards.
Detects Chinese cover slides, verifies Chinese characters, synchronizes with speech onset,
and handles frame-accurate video trimming and precise SRT timestamp shifting.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from config import DEVICE
from services.srt_utils import format_timestamp, parse_srt_timing

_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_ocr_reader = None
_ocr_lock = threading.Lock()


def get_ocr_reader():
    """Thread-safe singleton loader for EasyOCR Chinese reader."""
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                try:
                    import easyocr
                    use_gpu = (DEVICE == "cuda")
                    _ocr_reader = easyocr.Reader(['ch_sim'], gpu=use_gpu, verbose=False)
                except Exception as exc:
                    print(f"Warning: EasyOCR reader initialization failed ({exc})")
                    _ocr_reader = False
    return _ocr_reader if _ocr_reader is not False else None


def inspect_chinese_text(frame: np.ndarray) -> tuple[bool, int, str]:
    """Check if the frame contains Chinese characters (especially in the upper region)."""
    if frame is None or frame.size == 0:
        return False, 0, ""
    reader = get_ocr_reader()
    if reader is None:
        return False, 0, ""
    try:
        h, w = frame.shape[:2]
        crop_h = int(h * 0.70)
        crop = frame[:crop_h, :]
        results = reader.readtext(crop, paragraph=False)
        all_text = " ".join(item[1] for item in results)
        chars = _CHINESE_CHAR_RE.findall(all_text)
        return len(chars) >= 2, len(chars), all_text
    except Exception as exc:
        print(f"Cover OCR inspection warning: {exc}")
        return False, 0, ""


def detect_audio_speech_onset(video_path: str, max_check_sec: float = 4.0) -> float | None:
    """Quickly measure audio energy in the opening seconds to detect speech onset."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-t", f"{max_check_sec:.2f}",
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            return None
        samples = np.frombuffer(proc.stdout, dtype=np.int16)
        if len(samples) < 1600:
            return None

        # 50ms windows = 800 samples at 16kHz
        win_size = 800
        n_wins = len(samples) // win_size
        energies = []
        for w_idx in range(n_wins):
            win = samples[w_idx * win_size : (w_idx + 1) * win_size].astype(np.float32)
            rms = float(np.sqrt(np.mean(win ** 2)))
            energies.append(rms)

        if not energies:
            return None

        max_e = max(energies)
        if max_e < 350:
            return None

        baseline = float(np.median(energies[: min(8, len(energies))]))
        threshold = max(600.0, baseline * 2.2)

        for w_idx, e in enumerate(energies):
            if e > threshold and w_idx >= 4:
                onset_sec = (w_idx * win_size) / 16000.0
                return round(onset_sec, 2)
    except Exception:
        pass
    return None


def shift_srt_timestamps(srt_content: str, shift_seconds: float) -> str:
    """Shift all SRT timestamps earlier by shift_seconds.
    Omits any entries that ended before shift_seconds."""
    if not srt_content or shift_seconds <= 0.02:
        return srt_content

    shift_ms = int(round(shift_seconds * 1000))
    raw_timings = parse_srt_timing(srt_content)
    if not raw_timings:
        return srt_content

    new_lines = []
    counter = 1

    for start_ms, end_ms, text in raw_timings:
        new_end_ms = end_ms - shift_ms
        if new_end_ms <= 80:
            continue

        new_start_ms = max(0, start_ms - shift_ms)
        new_lines.extend([
            str(counter),
            f"{format_timestamp(new_start_ms / 1000.0)} --> {format_timestamp(new_end_ms / 1000.0)}",
            text,
            "",
        ])
        counter += 1

    return "\n".join(new_lines)


def detect_intro_cover(
    video_path: str,
    srt_content: str | None = None,
    max_check_sec: float = 4.0,
    require_chinese: bool = True,
) -> float:
    """Detect if the beginning of the video contains an intro cover, title slate, or freeze-frame with Chinese text.
    Returns the detected cut point in seconds (0.0 if no intro cover is detected).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration < 3.0:
        cap.release()
        return 0.0

    first_speech_time = None
    if srt_content:
        timings = parse_srt_timing(srt_content)
        if timings:
            first_speech_time = timings[0][0] / 1000.0
    if first_speech_time is None:
        first_speech_time = detect_audio_speech_onset(video_path, max_check_sec=max_check_sec)

    max_frames = min(int(max_check_sec * fps), total_frames - 10)
    if max_frames < 10:
        cap.release()
        return 0.0

    prev_gray = None
    prev_hist = None
    frame_diffs: list[tuple[float, float, float]] = []

    frames_cache: dict[int, np.ndarray] = {}

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i in (int(0.2 * fps), int(0.5 * fps), int(1.0 * fps), int(1.5 * fps)):
            frames_cache[i] = frame.copy()

        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        if prev_gray is not None and prev_hist is not None:
            d = float(np.mean(cv2.absdiff(gray, prev_gray)))
            corr = float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL))
            t = i / fps
            frame_diffs.append((t, d, corr))

        prev_gray = gray
        prev_hist = hist

    cap.release()

    if not frame_diffs:
        return 0.0

    diff_values = [d for _, d, _ in frame_diffs]
    avg_diff = float(np.mean(diff_values))
    std_diff = float(np.std(diff_values))

    cut_candidates = []
    static_run = 0

    for idx, (t, d, corr) in enumerate(frame_diffs):
        is_static = (d < 1.4 and corr > 0.95)
        if is_static:
            static_run += 1
        else:
            is_sharp_cut = (d > max(14.0, avg_diff + 2.0 * std_diff)) or (corr < 0.72)
            if static_run >= 6 and is_sharp_cut:
                cut_candidates.append(t)
            elif is_sharp_cut and 0.4 <= t <= 3.5:
                cut_candidates.append(t)
            static_run = 0

    if first_speech_time and first_speech_time >= 1.0:
        for t, d, corr in frame_diffs:
            if 0.4 <= t <= first_speech_time - 0.08:
                if (d > max(12.0, avg_diff + 1.5 * std_diff)) or (corr < 0.75):
                    cut_candidates.append(t)

    if not cut_candidates:
        return 0.0

    valid_cuts = sorted([c for c in set(cut_candidates) if 0.35 <= c <= max_check_sec])
    if first_speech_time:
        valid_cuts = [c for c in valid_cuts if c <= max(0.4, first_speech_time - 0.05)]

    if not valid_cuts:
        return 0.0

    best_cut = min(valid_cuts)

    if require_chinese:
        check_fidx = min(frames_cache.keys(), key=lambda k: abs(k / fps - (best_cut / 2))) if frames_cache else None
        sample_frame = frames_cache.get(check_fidx) if check_fidx is not None else None

        if sample_frame is None:
            c2 = cv2.VideoCapture(video_path)
            c2.set(cv2.CAP_PROP_POS_FRAMES, max(1, int((best_cut / 2) * fps)))
            ret, sample_frame = c2.read()
            c2.release()

        has_zh, zh_count, zh_text = inspect_chinese_text(sample_frame)
        if not has_zh:
            print(f"[INFO] Intro segment ({best_cut:.2f}s) has no Chinese cover text -> keeping video intact.")
            return 0.0

        print(f"[INFO] Detected Chinese intro cover [{zh_count} chars: '{zh_text[:30]}'], cut point: {best_cut:.2f}s")
        return round(best_cut, 2)

    print(f"[INFO] Detected intro cover cut point: {best_cut:.2f}s")
    return round(best_cut, 2)


def trim_video_file(
    video_path: str,
    trim_seconds: float,
    output_video_path: str,
) -> str:
    """Trim video starting from trim_seconds using frame-accurate FFmpeg encoding."""
    if trim_seconds <= 0.05:
        return video_path

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[TRIM] Trimming intro cover: cutting first {trim_seconds:.2f}s of video -> {output_video_path}")

    codecs_to_try = []
    if DEVICE == "cuda":
        codecs_to_try.append(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"])
    codecs_to_try.append(["-c:v", "libx264", "-preset", "fast", "-crf", "20"])

    last_err = ""
    for video_codec in codecs_to_try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", f"{trim_seconds:.3f}",
            "-map", "0:v",
            "-map", "0:a?",
            *video_codec,
            "-c:a", "aac", "-b:a", "192k",
            output_video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if res.returncode == 0 and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            return output_video_path
        last_err = res.stderr[-500:] if res.stderr else "FFmpeg trim error"

    raise RuntimeError(f"Lỗi cắt video: {last_err}")


def preprocess_trim_video(
    video_path: str,
    output_path: str | None = None,
    trim_mode: str = "auto",
    max_check_sec: float = 4.0,
    srt_content: str | None = None,
) -> tuple[str, float]:
    """Ingest-stage intro cover trimming.

    Returns:
        (processed_video_path, trimmed_seconds)
    """
    if not video_path or not os.path.exists(video_path):
        return video_path, 0.0

    trim_mode = str(trim_mode or "auto").strip().lower()
    if trim_mode in ("off", "0", "false", "none"):
        return video_path, 0.0

    trim_sec = 0.0
    if trim_mode in ("auto", "on", "true"):
        trim_sec = detect_intro_cover(video_path, srt_content=srt_content, max_check_sec=max_check_sec)
    else:
        try:
            trim_sec = float(trim_mode)
        except ValueError:
            trim_sec = 0.0

    if trim_sec <= 0.05:
        return video_path, 0.0

    if not output_path:
        p = Path(video_path)
        output_path = str(p.parent / f"{p.stem}_clean_cover{p.suffix}")

    try:
        trimmed_file = trim_video_file(video_path, trim_sec, output_path)
        return trimmed_file, trim_sec
    except Exception as exc:
        print(f"Warning: Cover trim failed ({exc}), continuing with original video")
        return video_path, 0.0


def trim_video_and_shift_srt(
    video_path: str,
    trim_seconds: float,
    output_video_path: str,
    srt_content: str | None = None,
) -> tuple[str, str | None]:
    """Trim video starting from trim_seconds and shift SRT timestamps accordingly."""
    if trim_seconds <= 0.05:
        return video_path, srt_content

    trimmed_path = trim_video_file(video_path, trim_seconds, output_video_path)
    shifted_srt = None
    if srt_content:
        shifted_srt = shift_srt_timestamps(srt_content, trim_seconds)

    return trimmed_path, shifted_srt
