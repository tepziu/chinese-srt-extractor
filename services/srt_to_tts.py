"""
services/srt_to_tts.py — Modular Subtitle-to-Speech (SRT to TTS) Engine.

Cung cấp tính năng chuyển đổi độc lập từ file phụ đề .srt sang file âm thanh TTS
với các nguyên tắc chuẩn:
1. Độ dài file audio TTS chính bằng thời lượng của phụ đề (timeline-accurate).
2. Thuyết minh TTS và SRT đồng nhất và đủ ý: đọc trọn vẹn ngữ nghĩa, không nuốt chữ.
3. Chống chồng chéo tuyệt đối (Zero Overlap Guarantee): phân bổ timeline thông minh,
   cắt tỉa khoảng lặng thừa, tăng tốc mượt mà bằng atempo nếu cần, có buffer an toàn.
4. Xuất file audio MP3 và file phụ đề SRT đồng bộ hóa (aligned SRT).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import (
    LANGUAGES,
    OUTPUT_FOLDER,
    SPEAKER_VOICE_MAPS,
    TTS_VOICES,
    UPLOAD_FOLDER,
    create_job,
    get_job,
    jobs,
)
from services.google_tts import synthesize_to_wav
from services.srt_utils import format_timestamp, parse_srt
from services.runtime_state import serial_task

_SPEAKER_RE = re.compile(r"^\s*(?:\[|\()([A-Za-z0-9_]+)(?:\]|\))\s*:?\s*", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
_TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def clean_subtitle_line(raw_text: str) -> tuple[str, str | None]:
    """
    Làm sạch các tag định dạng (HTML, ASS) và trích xuất mã nhân vật nếu có.
    Ví dụ: '[M1] Xin chào các bạn' -> ('Xin chào các bạn', 'M1')
    """
    text = str(raw_text or "").strip()
    text = _HTML_TAG_RE.sub("", text)
    speaker = None
    spk_match = _SPEAKER_RE.match(text)
    if spk_match:
        speaker = spk_match.group(1).upper()
        text = text[spk_match.end():].strip()

    # Dọn dẹp khoảng trắng dư thừa
    text = " ".join(text.split())
    return text, speaker


def parse_srt_data(srt_content: str) -> list[dict[str, Any]]:
    """
    Phân tích chuỗi SRT thành danh sách các segment có cấu trúc đầy đủ.
    Trả về danh sách các dict chứa:
    - index: số thứ tự (1, 2, 3...)
    - start_ms: thời điểm bắt đầu (ms)
    - end_ms: thời điểm kết thúc (ms)
    - duration_ms: thời lượng hiển thị (ms)
    - text: lời thoại đã làm sạch để đọc TTS
    - raw_text: nội dung gốc
    - speaker: mã speaker nếu có (M1, F1...)
    """
    entries = parse_srt(srt_content)
    segments = []
    for pos, (idx_str, timestamp, text) in enumerate(entries, start=1):
        parts = timestamp.split("-->", 1)
        if len(parts) != 2:
            continue
        m_start = _TIMESTAMP_RE.match(parts[0].strip())
        m_end = _TIMESTAMP_RE.match(parts[1].strip().split()[0])
        if not m_start or not m_end:
            continue
        h1, m1, s1, ms1 = map(int, m_start.groups())
        h2, m2, s2, ms2 = map(int, m_end.groups())
        start_ms = h1 * 3600000 + m1 * 60000 + s1 * 1000 + ms1
        end_ms = h2 * 3600000 + m2 * 60000 + s2 * 1000 + ms2
        if end_ms <= start_ms:
            continue

        clean_text, speaker = clean_subtitle_line(text)
        if not clean_text:
            continue

        segments.append({
            "index": pos,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "text": clean_text,
            "raw_text": text,
            "speaker": speaker,
        })

    segments.sort(key=lambda x: x["start_ms"])
    return segments


def trim_silence_padding(
    audio: AudioSegment,
    silence_threshold: int = -40,
    padding_ms: int = 25,
) -> AudioSegment:
    """
    Cắt tỉa khoảng lặng thừa đầu và cuối của audio TTS (Edge/Gemini TTS thường dư 150-350ms),
    giữ lại một khoảng đệm mỏng để giọng nói phát âm mượt mà, không bị cụt âm đầu/cuối.
    """
    orig_len = len(audio)
    if orig_len <= 80:
        return audio
    start_trim = detect_leading_silence(audio, silence_threshold=silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=silence_threshold)
    start_pos = max(0, start_trim - padding_ms)
    end_pos = min(orig_len, orig_len - end_trim + padding_ms)
    if start_pos < end_pos and (end_pos - start_pos) >= 50:
        return audio[start_pos:end_pos]
    return audio


def speed_adjust_audio(
    audio: AudioSegment,
    target_duration_ms: int,
    temp_dir: Path | None = None,
    max_speed: float = 1.85,
) -> tuple[AudioSegment, float]:
    """
    Điều chỉnh tốc độ âm thanh bằng FFmpeg filter atempo.
    Bảo toàn cao độ âm thanh (pitch), tránh hiện tượng giọng chipmunk hoặc vỡ tiếng.
    """
    curr_len = len(audio)
    if target_duration_ms <= 0 or curr_len <= target_duration_ms:
        return audio, 1.0

    if curr_len - target_duration_ms <= 15:
        return audio, 1.0

    speed_ratio = curr_len / target_duration_ms
    speed_factor = min(speed_ratio, max_speed)

    filters = []
    remaining = speed_factor
    while remaining > 2.0:
        filters.append("atempo=2.0000")
        remaining /= 2.0
    if remaining > 1.0:
        filters.append(f"atempo={remaining:.4f}")

    if not filters:
        return audio, 1.0

    import tempfile
    if temp_dir is None or not temp_dir.exists():
        temp_dir = Path(tempfile.gettempdir())

    unique_id = uuid.uuid4().hex[:8]
    in_path = temp_dir / f"atempo_in_{unique_id}.wav"
    out_path = temp_dir / f"atempo_out_{unique_id}.wav"

    try:
        with open(in_path, "wb") as f_in:
            audio.export(f_in, format="wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-filter:a", ",".join(filters),
            "-vn", str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=15)
        if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            with open(out_path, "rb") as f_out:
                adjusted = AudioSegment.from_wav(f_out)
            return adjusted, speed_factor
    except Exception as exc:
        print(f"[SRT_TO_TTS] Lỗi atempo điều chỉnh tốc độ: {exc}")
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)

    return audio, 1.0


def align_srt_to_tts_timeline(segments, segment_audios, options=None, temp_dir=None):
    """Place complete speech without overlap; report overflow instead of cutting words.

    PCM is copied once per segment into a mutable master buffer, avoiding a full
    AudioSegment.overlay copy for every sentence (quadratic work on long videos).
    """
    from services.pipeline_options import validate_tts_options
    opts = validate_tts_options(options)
    margin = opts['safety_margin_ms']
    mode = opts.get('align_mode', 'smart_sync')
    sub_end = max((s['end_ms'] for s in segments), default=0)
    pcm = bytearray()
    last_end = 0
    placed = False
    results = []
    aligned = []
    for i, seg in enumerate(segments):
        if opts.get('cancelled_cb') and opts['cancelled_cb']():
            raise RuntimeError('Đã hủy quá trình căn chỉnh TTS')
        raw = segment_audios[i] if i < len(segment_audios) else None
        row = dict(index=seg['index'], text=seg['text'], speaker=seg.get('speaker'),
                   orig_start_ms=seg['start_ms'], orig_end_ms=seg['end_ms'],
                   orig_duration_ms=seg['duration_ms'], raw_duration_ms=0,
                   actual_start_ms=seg['start_ms'], actual_end_ms=seg['end_ms'],
                   actual_duration_ms=0, speed_factor=1.0, overlap_ms=0,
                   timing_overflow_ms=0, status='failed')
        if raw is None or len(raw) == 0:
            results.append(row)
            continue
        audio = trim_silence_padding(raw).set_frame_rate(24000).set_channels(1).set_sample_width(2)
        if len(audio) == 0:
            results.append(row)
            continue
        raw_duration = len(audio)
        safe_start = last_end + margin if placed else 0
        start = max(seg['start_ms'], safe_start)
        if mode == 'smart_sync' and opts.get('allow_borrow_preceding', True) and len(audio) > seg['duration_ms']:
            start -= min(max(0, start-safe_start), 350, len(audio)-seg['duration_ms'])
        end_limit = (segments[i+1]['start_ms']-margin) if i+1 < len(segments) else seg['end_ms']
        if mode == 'strict_sub':
            end_limit = min(end_limit, seg['end_ms'])
        available = max(1, end_limit-start)
        speed = 1.0
        if len(audio) > available:
            audio, speed = speed_adjust_audio(audio, available, temp_dir=temp_dir, max_speed=opts['max_speed_ratio'])
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        end = start + math.ceil(len(audio.raw_data) / 48)
        overflow = max(0, end-end_limit)
        # At most codec rounding tolerance; any larger spill must be reviewed.
        status = 'timing_overflow' if overflow > 25 else ('speed_adjusted' if speed > 1 else 'natural_fit')
        offset = start * 48  # 24 kHz, mono, signed 16-bit PCM
        needed = offset + len(audio.raw_data)
        if len(pcm) < needed:
            pcm.extend(b'\0' * (needed-len(pcm)))
        pcm[offset:needed] = audio.raw_data
        row.update(actual_start_ms=start, actual_end_ms=end, actual_duration_ms=len(audio),
                   raw_duration_ms=raw_duration, speed_factor=round(speed, 3), status=status,
                   overlap_ms=max(0, last_end-start) if placed else 0, timing_overflow_ms=overflow)
        results.append(row)
        aligned.extend([str(len(aligned)//4+1),
                        f"{format_timestamp(start/1000)} --> {format_timestamp(end/1000)}",
                        seg['text'], ''])
        last_end = end
        placed = True
    duration = max(sub_end, last_end, int(opts.get('target_duration_ms') or 0))
    needed = duration * 48
    if len(pcm) < needed:
        pcm.extend(b'\0' * (needed-len(pcm)))
    final = AudioSegment(data=bytes(pcm), sample_width=2, frame_rate=24000, channels=1)
    return final, results, '\n'.join(aligned), len(final)


def _resolve_voice(engine: str, lang: str, speaker_id: str | None, custom_voice: str | None = None, custom_map: dict | None = None) -> tuple[str, str]:
    """Xác định giọng đọc và cao độ/emotion cho một speaker."""
    if custom_voice:
        return custom_voice, "+0Hz" if engine == "edge" else "warm"

    if custom_map and speaker_id and speaker_id in custom_map:
        val = custom_map[speaker_id]
        if isinstance(val, str):
            return val, "+0Hz" if engine == "edge" else "warm"
        if isinstance(val, dict):
            return val.get("voice", TTS_VOICES.get(lang, "vi-VN-NamMinhNeural")), val.get("pitch", "+0Hz")

    engine_map = SPEAKER_VOICE_MAPS.get(engine, {}).get(lang, {})
    spk_key = speaker_id or "M1"
    if spk_key in engine_map:
        info = engine_map[spk_key]
        return info["voice"], info.get("pitch", "+0Hz") if engine == "edge" else info.get("emotion", "warm")

    default_v = TTS_VOICES.get(lang, "vi-VN-NamMinhNeural")
    return default_v, "+0Hz" if engine == "edge" else "warm"


async def synthesize_edge_segment(
    text: str,
    voice: str,
    pitch: str,
    out_path: Path,
    cancelled_cb=None,
) -> bool:
    """Tạo audio cho một đoạn bằng edge-tts với cơ chế thử lại."""
    import edge_tts
    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    for attempt in range(4):
        if cancelled_cb and cancelled_cb():
            return False
        try:
            communicate = edge_tts.Communicate(text, voice, pitch=pitch)
            staging = out_path.with_suffix('.part.mp3')
            await communicate.save(str(staging))
            if staging.exists() and staging.stat().st_size > 0:
                os.replace(staging, out_path)
                return True
        except Exception as exc:
            if attempt < 3:
                await asyncio.sleep(1 + attempt * 1.5)
        finally:
            out_path.with_suffix('.part.mp3').unlink(missing_ok=True)
    return False


def synthesize_all_segments(
    segments: list[dict[str, Any]],
    lang: str,
    engine: str = "edge",
    custom_voice: str | None = None,
    speaker_voices: dict | None = None,
    temp_dir: Path | None = None,
    progress_cb=None,
    cancelled_cb=None,
    options: dict | None = None,
) -> list[AudioSegment | None]:
    """
    Sinh audio cho tất cả các câu trong danh sách segments.
    Hỗ trợ Edge TTS, Gemini TTS và OmniVoice.
    Trả về danh sách các đối tượng AudioSegment (hoặc None nếu lỗi).
    """
    if temp_dir is None:
        temp_dir = Path(UPLOAD_FOLDER) / "temp_srt_tts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total = len(segments)
    audio_segments: list[AudioSegment | None] = [None] * total

    def cache_path(seg, voice, style, suffix):
        from services.google_tts import load_google_tts_settings
        model = load_google_tts_settings().model if engine == 'gemini' else 'edge-v1'
        data = [seg['text'], lang, engine, model, voice, style, (options or {}).get('style_prompt', ''),
                seg['index'] > 1 if engine == 'gemini' else False]
        digest = hashlib.sha256(json.dumps(data, ensure_ascii=False).encode('utf-8')).hexdigest()[:24]
        return temp_dir / f'seg_{digest}.{suffix}'

    if engine == "edge":
        async def run_edge():
            batch_size = 3
            for i in range(0, total, batch_size):
                if cancelled_cb and cancelled_cb():
                    break
                batch_indices = list(range(i, min(i + batch_size, total)))
                tasks = []
                for idx in batch_indices:
                    seg = segments[idx]
                    spk = seg.get("speaker") or "M1"
                    v, p = _resolve_voice("edge", lang, spk, custom_voice, speaker_voices)
                    out_f = cache_path(seg, v, p, 'mp3')
                    tasks.append((idx, out_f, synthesize_edge_segment(seg["text"], v, p, out_f, cancelled_cb)))

                sub_results = await asyncio.gather(*[t[2] for t in tasks])
                for (idx, out_f, _), ok in zip(tasks, sub_results):
                    if ok and out_f.exists() and out_f.stat().st_size > 0:
                        try:
                            audio_segments[idx] = AudioSegment.from_mp3(str(out_f))
                        except Exception:
                            out_f.unlink(missing_ok=True)
                            audio_segments[idx] = None

                if progress_cb:
                    done_count = min(i + batch_size, total)
                    progress_cb(int(done_count / total * 70), f"Đang tạo giọng đọc: {done_count}/{total}")
                await asyncio.sleep(0.3)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_edge())
        finally:
            loop.close()

    elif engine == "gemini":
        opts = options or {}
        style_prompt = str(opts.get("style_prompt") or "").strip()[:500]
        for idx, seg in enumerate(segments):
            if cancelled_cb and cancelled_cb():
                break
            spk = seg.get("speaker") or "M1"
            v, emotion = _resolve_voice("gemini", lang, spk, custom_voice, speaker_voices)
            out_f = cache_path(seg, v, emotion, 'wav')
            if not out_f.exists() or out_f.stat().st_size == 0:
                try:
                    staging = out_f.with_suffix('.part.wav')
                    synthesize_to_wav(
                        seg["text"], staging, lang=lang, voice=v, emotion=emotion,
                        style_prompt=style_prompt, continuation=idx > 0,
                        cancelled=cancelled_cb,
                    )
                    os.replace(staging, out_f)
                except Exception as exc:
                    print(f"[Gemini TTS] Đoạn {idx} lỗi: {exc}")
                finally:
                    out_f.with_suffix('.part.wav').unlink(missing_ok=True)

            if out_f.exists() and out_f.stat().st_size > 0:
                try:
                    audio_segments[idx] = AudioSegment.from_wav(str(out_f))
                except Exception:
                    out_f.unlink(missing_ok=True)
                    audio_segments[idx] = None

            if progress_cb:
                progress_cb(int((idx + 1) / total * 70), f"Gemini đang tạo giọng: {idx + 1}/{total}")

    elif engine == 'omnivoice':
        from services.tts import generate_omnivoice_audio
        job_id = (options or {}).get('job_id')
        if not job_id:
            raise ValueError('OmniVoice cần job context')
        paths = generate_omnivoice_audio(job_id, lang,
            [(s['start_ms'], s['end_ms'], s['text']) for s in segments], total, temp_dir, f'tts_{lang}')
        for idx, path in enumerate(paths):
            if path:
                audio_segments[idx] = AudioSegment.from_wav(path)
    else:
        raise ValueError('Engine TTS không được hỗ trợ')

    return audio_segments


@serial_task
def process_srt_to_tts(
    srt_content: str,
    lang: str = "vi",
    engine: str = "edge",
    voice: str | None = None,
    options: dict[str, Any] | None = None,
    job_id: str | None = None,
    output_dir: Path | None = None,
    output_audio_path: Path | str | None = None,
    manage_status: bool = True,
) -> dict[str, Any]:
    """
    Toàn bộ luồng xử lý: Từ file SRT sang audio MP3 timeline-synced và aligned SRT.
    """
    if not str(srt_content or "").strip():
        raise ValueError("Nội dung phụ đề .srt rỗng")

    segments = parse_srt_data(srt_content)
    if not segments:
        raise ValueError("Không tìm thấy dòng phụ đề hợp lệ nào trong file .srt")

    if lang not in LANGUAGES:
        lang = "vi"

    if job_id is None:
        job_id = f"srttts_{uuid.uuid4().hex[:10]}"

    if output_dir is None:
        output_dir = OUTPUT_FOLDER / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = UPLOAD_FOLDER / job_id / "seg_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)

    from services.pipeline_options import validate_tts_options
    opts = validate_tts_options(options)
    opts['job_id'] = job_id
    speakers = opts.get('segment_speakers') or jobs.get(job_id, {}).get('segment_speakers') or []
    for idx, segment in enumerate(segments):
        if not segment.get('speaker') and idx < len(speakers):
            segment['speaker'] = speakers[idx]
    tts_key = f"tts_{lang}"

    def _mark(prog: int, msg: str):
        if job_id in jobs:
            jobs[job_id]["progress"] = prog
            jobs[job_id]["message"] = msg
            jobs[job_id].setdefault(tts_key, {})["progress"] = prog
            jobs[job_id][tts_key]["message"] = msg

    def _is_cancelled() -> bool:
        return bool(jobs.get(job_id, {}).get("cancel"))

    if _is_cancelled():
        raise RuntimeError('Đã hủy quá trình tạo thuyết minh')
    opts['cancelled_cb'] = _is_cancelled

    _mark(5, "Đang khởi tạo TTS cho phụ đề...")

    # 1. Sinh âm thanh cho từng segment
    segment_audios = synthesize_all_segments(
        segments=segments,
        lang=lang,
        engine=engine,
        custom_voice=voice,
        speaker_voices=opts.get("speaker_voices"),
        temp_dir=temp_dir,
        progress_cb=_mark,
        cancelled_cb=_is_cancelled,
        options=opts,
    )

    if _is_cancelled():
        raise RuntimeError("Đã hủy quá trình tạo thuyết minh")
    if not any(audio is not None and len(audio) for audio in segment_audios):
        raise RuntimeError('Không tạo được đoạn giọng đọc nào; không xuất audio im lặng')

    _mark(75, "Đang tính toán timeline chống chồng chéo...")

    # 2. Căn chỉnh timeline, chống chồng chéo, ghép master audio
    final_audio, seg_results, aligned_srt, total_dur_ms = align_srt_to_tts_timeline(
        segments=segments,
        segment_audios=segment_audios,
        options=opts,
        temp_dir=temp_dir,
    )

    _mark(90, "Đang xuất file audio MP3 và phụ đề đồng bộ...")

    # 3. Xuất file kết quả
    lang_name = LANGUAGES.get(lang, {}).get("name", lang)
    if output_audio_path:
        out_audio_path = Path(output_audio_path)
        out_audio_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_audio_path = output_dir / f"tts_aligned_{lang}.mp3"
    out_srt_path = output_dir / f"subtitles_aligned_{lang}.srt"

    with open(out_audio_path, "wb") as f_out:
        final_audio.export(f_out, format="mp3", bitrate="192k")
    out_srt_path.write_text(aligned_srt, encoding="utf-8")

    file_size = out_audio_path.stat().st_size
    failures = [r["index"] for r in seg_results if r["status"] == "failed"]
    needs_review = [r['index'] for r in seg_results if r['status'] == 'timing_overflow']
    speed_adjusted_count = sum(1 for r in seg_results if "speed_adjusted" in r["status"])

    sub_end_ms = segments[-1]["end_ms"]
    res_dict = {
        "status": "partial" if failures or needs_review else "done",
        "job_id": job_id,
        "lang": lang,
        "lang_name": lang_name,
        "engine": engine,
        "audio_path": str(out_audio_path),
        "audio_filename": out_audio_path.name,
        "audio_size": file_size,
        "audio_duration_ms": total_dur_ms,
        "audio_duration_s": round(total_dur_ms / 1000.0, 2),
        "sub_duration_ms": sub_end_ms,
        "sub_duration_s": round(sub_end_ms / 1000.0, 2),
        "srt_path": str(out_srt_path),
        "srt_filename": out_srt_path.name,
        "total_segments": len(segments),
        "speed_adjusted_segments": speed_adjusted_count,
        "failed_segments": failures,
        "needs_review_segments": needs_review,
        "overlap_ms": sum(r['overlap_ms'] for r in seg_results),
        "clipped_segments": [],
        "segments": seg_results,
        "aligned_srt_preview": aligned_srt[:1200],
    }

    final_status = "partial" if failures or needs_review else "done"
    status_msg = (
        f"Hoàn thành ({file_size / 1048576:.1f}MB, {len(segments)} câu, 0% chồng chéo)"
        if not failures
        else f"Hoàn thành một phần ({file_size / 1048576:.1f}MB • thiếu {len(failures)}/{len(segments)} đoạn)"
    )
    if needs_review:
        status_msg += f' • {len(needs_review)} câu cần chỉnh thời gian (đã giữ đủ lời)'
    from services.srt_utils import validate_srt
    if aligned_srt and not validate_srt(aligned_srt)[0]:
        raise RuntimeError('Phụ đề sau căn chỉnh không hợp lệ')
    _mark(100, status_msg)
    if job_id in jobs:
        if manage_status:
            jobs[job_id]["status"] = final_status
        jobs[job_id][tts_key] = {
            "status": final_status,
            "progress": 100,
            "message": status_msg,
            "path": str(out_audio_path),
            "filename": out_audio_path.name,
            "size": file_size,
            "duration": round(total_dur_ms / 1000.0, 1),
            "failed_segments": failures,
            "needs_review_segments": needs_review,
            "total_segments": len(segments),
        }
        jobs[job_id]["aligned_srt"] = {
            "path": str(out_srt_path),
            "filename": out_srt_path.name,
        }
        jobs[job_id]["srt_to_tts_result"] = res_dict
        if hasattr(jobs[job_id], 'persist'):
            jobs[job_id].persist()

    (output_dir / f'tts_manifest_{lang}.json').write_text(json.dumps(res_dict, ensure_ascii=False, indent=2), encoding='utf-8')

    return res_dict


def srt_to_mp3(
    srt_input: str | Path,
    output_path: str | Path | None = None,
    lang: str = "vi",
    engine: str = "edge",
    voice: str | None = None,
    align_mode: str = "smart_sync",
    safety_margin_ms: int = 60,
    max_speed_ratio: float = 1.65,
) -> dict[str, Any]:
    """
    Hàm độc lập gọn nhẹ: Nhận vào file .srt hoặc chuỗi SRT, chỉ xuất ra file audio TTS (.mp3).
    - srt_input: Đường dẫn tới file .srt (str/Path) hoặc nội dung chuỗi SRT.
    - output_path: Đường dẫn file mp3 đầu ra (nếu None, tự đặt tên <tên_gốc>_tts.mp3 cùng thư mục).
    """
    srt_path = None
    if isinstance(srt_input, Path) or (isinstance(srt_input, str) and os.path.exists(srt_input) and os.path.isfile(srt_input)):
        srt_path = Path(srt_input)
        srt_content = srt_path.read_text(encoding="utf-8")
        if output_path is None:
            output_path = srt_path.with_name(f"{srt_path.stem}_tts.mp3")
    else:
        srt_content = str(srt_input)
        if output_path is None:
            output_path = OUTPUT_FOLDER / f"tts_{int(time.time())}.mp3"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    options = {
        "align_mode": align_mode,
        "safety_margin_ms": safety_margin_ms,
        "max_speed_ratio": max_speed_ratio,
    }

    job_id = f"srttts_{uuid.uuid4().hex[:10]}"
    create_job(
        job_id,
        status="generating",
        progress=5,
        message="Đang tạo file audio TTS từ phụ đề .srt...",
    )

    return process_srt_to_tts(
        srt_content=srt_content,
        lang=lang,
        engine=engine,
        voice=voice,
        options=options,
        job_id=job_id,
        output_audio_path=output_path,
    )
