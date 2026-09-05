"""
audio_mixer.py — BGM (Background Music) Separation, Audio Ducking, and Voiceover Mixing.
Supports AI stem separation via Demucs (htdemucs) and native FFmpeg sidechain compression.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import DEVICE, OUTPUT_FOLDER, jobs

SR = 48000


def has_audio_stream(video_or_audio_path: str) -> bool:
    """Check whether the input media file contains an audio stream."""
    if not video_or_audio_path or not os.path.exists(video_or_audio_path):
        return False
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_name",
        "-print_format", "json",
        video_or_audio_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            return False
        data = json.loads(proc.stdout or "{}")
        return len(data.get("streams", [])) > 0
    except Exception:
        return False


def separate_bgm_and_vocals(
    video_or_audio_path: str,
    output_dir: Path,
) -> tuple[str | None, str | None]:
    """Separate input media into vocals and instrumental (BGM + SFX) using Demucs.

    Returns:
        (bgm_path, vocal_path) or (None, None) if separation failed/unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    venv_demucs = Path(sys.executable).parent / "demucs.exe"
    demucs_exe = str(venv_demucs) if venv_demucs.exists() else shutil.which("demucs")

    if not demucs_exe:
        print("[BGM] Demucs executable not found in PATH or venv")
        return None, None

    # Pre-extract a clean 44.1kHz stereo WAV for fast, reliable Demucs ingestion
    temp_wav = output_dir / "audio_source.wav"
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", str(video_or_audio_path),
        "-vn", "-ac", "2", "-ar", "44100",
        "-c:a", "pcm_s16le",
        str(temp_wav),
    ]
    extract_res = subprocess.run(cmd_extract, capture_output=True, text=True, timeout=600)
    input_to_demucs = str(temp_wav) if (extract_res.returncode == 0 and temp_wav.exists()) else str(video_or_audio_path)

    cmd = [
        demucs_exe,
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-d", "cpu",
        "-o", str(output_dir),
        input_to_demucs,
    ]

    try:
        print(f"[BGM] Running Demucs separation on {Path(input_to_demucs).name}...")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if res.returncode == 0:
            vocal_file = next(output_dir.rglob("vocals.wav"), None)
            bgm_file = next(output_dir.rglob("no_vocals.wav"), None)
            if bgm_file and vocal_file and bgm_file.exists() and vocal_file.exists():
                print(f"[BGM] Demucs separation successful -> {bgm_file.name}")
                return str(bgm_file), str(vocal_file)
        else:
            err = res.stderr[-400:] if res.stderr else "Unknown demucs error"
            print(f"[BGM] Demucs exited with code {res.returncode}: {err}")
    except Exception as exc:
        print(f"[BGM] Demucs separation exception: {exc}")
    finally:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except OSError:
                pass

    return None, None


def mix_voiceover_with_bgm(
    video_path: str,
    tts_audio_path: str,
    output_audio_path: str,
    job_id: str,
    bgm_mode: str = "ai",
    bgm_volume: float = 0.80,
    original_voice_bleed: float = 0.0,
) -> str:
    """Blend the new TTS voiceover with the background music of the original video.

    Args:
        video_path: path to the source video (or audio).
        tts_audio_path: path to the new TTS voiceover audio file.
        output_audio_path: destination path for the mixed audio file.
        job_id: job identifier.
        bgm_mode: 'ai' (Demucs separation), 'duck' (FFmpeg sidechain compression), 'none' (voiceover only), or 'auto'.
        bgm_volume: volume multiplier for the BGM track (default 0.80).
        original_voice_bleed: ratio of original vocals to bleed back in.

    Returns:
        Path to the resulting audio file.
    """
    if not tts_audio_path or not os.path.exists(tts_audio_path):
        return tts_audio_path

    if not has_audio_stream(video_path):
        print("[BGM] Source video has no audio stream -> using voiceover directly")
        return tts_audio_path

    bgm_mode = str(bgm_mode or "ai").strip().lower()
    if bgm_mode in ("none", "off", "0", "mute"):
        print("[BGM] BGM mode is disabled -> using voiceover only")
        return tts_audio_path

    job_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)

    clamped_vol = max(0.05, min(1.5, float(bgm_volume)))

    # Attempt 1: AI separation with Demucs if mode is 'ai' or 'auto'
    if bgm_mode in ("ai", "auto"):
        if job_id and jobs.get(job_id):
            burn_keys = [k for k in jobs[job_id] if k.startswith("burn_") and isinstance(jobs[job_id][k], dict)]
            if burn_keys:
                jobs[job_id][burn_keys[0]]["message"] = "🎵 Đang tách nhạc nền AI (Demucs - xóa sạch tiếng Trung)..."
                jobs[job_id][burn_keys[0]]["progress"] = 35

        sep_dir = job_dir / "separated"
        bgm_file, _ = separate_bgm_and_vocals(video_path, sep_dir)
        if bgm_file and os.path.exists(bgm_file):
            print(f"[BGM] Mixing separated BGM ({clamped_vol*100:.0f}% volume) with TTS...")
            if job_id and jobs.get(job_id):
                burn_keys = [k for k in jobs[job_id] if k.startswith("burn_") and isinstance(jobs[job_id][k], dict)]
                if burn_keys:
                    jobs[job_id][burn_keys[0]]["message"] = "🎧 Đang trộn nhạc nền đã tách với giọng đọc mới..."
                    jobs[job_id][burn_keys[0]]["progress"] = 65

            filter_mix = (
                f"[0:a]volume={clamped_vol:g},aformat=sample_rates=48000:channel_layouts=stereo[bgm];"
                "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[sc_vo][vo];"
                "[bgm][sc_vo]sidechaincompress=threshold=0.04:ratio=6:attack=30:release=400[ducked_bgm];"
                "[ducked_bgm][vo]amix=inputs=2:duration=first:normalize=0[aout]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", bgm_file,
                "-i", tts_audio_path,
                "-filter_complex", filter_mix,
                "-map", "[aout]",
                "-c:a", "aac", "-b:a", "192k",
                output_audio_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                print(f"[BGM] AI mixing completed successfully -> {output_audio_path}")
                return output_audio_path
            print("[BGM] AI mix filter failed, falling back to FFmpeg sidechain ducking")

    # Attempt 2: Fast Native FFmpeg Sidechain Compression on original audio track
    print(f"[BGM] Using FFmpeg Sidechain Ducking (bgm_vol={clamped_vol*100:.0f}%)...")
    filter_duck = (
        f"[0:a]volume={clamped_vol:g},aformat=sample_rates=48000:channel_layouts=stereo[orig];"
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[sc_vo][vo];"
        "[orig][sc_vo]sidechaincompress=threshold=0.025:ratio=12:attack=15:release=350[ducked_orig];"
        "[ducked_orig][vo]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", tts_audio_path,
        "-filter_complex", filter_duck,
        "-map", "[aout]",
        "-c:a", "aac", "-b:a", "192k",
        output_audio_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
        print(f"[BGM] FFmpeg ducking completed successfully -> {output_audio_path}")
        return output_audio_path

    print("[BGM] All mixing methods failed -> falling back to original voiceover")
    return tts_audio_path
