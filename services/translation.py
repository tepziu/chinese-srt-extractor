"""Subtitle translation with multi-speaker detection, length control, and strict SRT validation."""

from __future__ import annotations

import re
import time

from config import AI_DEFAULT_MODEL, AI_LANG_NAMES, AI_TRANSLATE_CONFIG, LANGUAGES, jobs
from services.srt_utils import parse_srt, parse_srt_timing, validate_srt

_NUMBERED_LINE = re.compile(r"^\s*(\d+)[.)]\s*(?:\[([A-Za-z0-9_]+)\]|\(([A-Za-z0-9_]+)\))?\s*(.*?)\s*$")
_SPEAKER_IN_TEXT = re.compile(r"^\s*[\[\(]([A-Za-z0-9_]+)[\]\)]\s*(.*)$")


def _rebuild_srt(entries: list, translated_map: dict[int, str]) -> str:
    lines = []
    for index, (srt_index, timestamp, text) in enumerate(entries):
        lines.extend([srt_index, timestamp, translated_map.get(index, text), ""])
    return "\n".join(lines)


def _parse_numbered_result(result_text: str, batch_size: int) -> tuple[dict[int, str], dict[int, str]]:
    """Parse numbered result, extracting clean text and optional speaker tags."""
    parsed_text = {}
    parsed_speakers = {}
    for raw_line in str(result_text or "").splitlines():
        match = _NUMBERED_LINE.match(raw_line)
        if not match:
            continue
        number = int(match.group(1)) - 1
        speaker = (match.group(2) or match.group(3) or "").upper()
        value = match.group(4).strip()

        # If speaker tag was embedded inside text, extract it
        if not speaker and value:
            sub_m = _SPEAKER_IN_TEXT.match(value)
            if sub_m:
                speaker = sub_m.group(1).upper()
                value = sub_m.group(2).strip()

        if 0 <= number < batch_size and value:
            parsed_text[number] = value
            if speaker in {"M1", "F1", "M2", "F2", "N"}:
                parsed_speakers[number] = speaker
    return parsed_text, parsed_speakers


def _fallback_google(batch: list, translated_map: dict, start_i: int, target_lang: str) -> None:
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="zh-CN", target=target_lang)
    for offset, (_srt_index, _timestamp, text) in enumerate(batch):
        index = start_i + offset
        if index in translated_map:
            continue
        try:
            translated_map[index] = translator.translate(text) or text
        except Exception:
            translated_map[index] = text


def translate_srt_ai(srt_content: str, target_lang: str, job_id: str, ai_model: str | None = None) -> str:
    """Translate SRT using AI with length-awareness and multi-speaker role detection."""
    from openai import OpenAI

    entries = parse_srt(srt_content)
    if not entries:
        return srt_content
    if target_lang not in LANGUAGES:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ: {target_lang}")
    if not AI_TRANSLATE_CONFIG["api_key"]:
        raise RuntimeError("AI translation API key chưa được cấu hình")

    model = ai_model or jobs.get(job_id, {}).get("ai_model") or AI_TRANSLATE_CONFIG.get("model") or AI_DEFAULT_MODEL
    client = OpenAI(base_url=AI_TRANSLATE_CONFIG["base_url"], api_key=AI_TRANSLATE_CONFIG["api_key"])
    lang_name = AI_LANG_NAMES.get(target_lang, target_lang)

    # Parse timing to compute slot duration for each line
    timings = parse_srt_timing(srt_content)
    durations = {}
    for i, (s_ms, e_ms, _) in enumerate(timings):
        dur_s = max(0.8, round((e_ms - s_ms) / 1000, 1))
        durations[i] = dur_s

    batch_size = 15
    batches = [(start, entries[start : start + batch_size]) for start in range(0, len(entries), batch_size)]
    translated_map: dict[int, str] = {}
    speaker_map: dict[int, str] = {}

    for batch_number, (start_i, batch) in enumerate(batches, start=1):
        if jobs.get(job_id, {}).get("cancel"):
            raise RuntimeError("Đã hủy dịch thuật (Stop)")

        numbered_lines = []
        for offset, entry in enumerate(batch):
            idx = start_i + offset
            dur_info = f"[Thời lượng: {durations.get(idx, 2.0)}s]"
            numbered_lines.append(f"{offset + 1}. {dur_info} {entry[2]}")
        numbered_text = "\n".join(numbered_lines)

        prompt = (
            f"Bạn là chuyên gia dịch thuật phụ đề và lồng tiếng phim chuyên nghiệp (Trung sang {lang_name}).\n"
            f"Hãy dịch các câu thoại tiếng Trung sau sang {lang_name} tự nhiên, đúng ngữ cảnh đối thoại.\n\n"
            "QUY TẮC:\n"
            "1. PHÂN VAI NHÂN VẬT: Gắn mã nhân vật vào đầu mỗi câu dịch:\n"
            "   - [M1] = Nam chính\n"
            "   - [F1] = Nữ chính\n"
            "   - [M2] = Nam phụ\n"
            "   - [F2] = Nữ phụ\n"
            "   - [N]  = Dẫn chuyện / Thuyết minh\n"
            "2. KHỐNG CHẾ ĐỘ DÀI: Dịch ngắn gọn, xúc tích tương ứng với thời lượng của câu (khoảng 3-4 từ/giây) để giọng đọc vừa khớp video.\n"
            "3. ĐỊNH DẠNG: Chỉ trả về các dòng đánh số tương ứng với số thứ tự, KHÔNG giải thích thêm.\n\n"
            f"{numbered_text}"
        )

        parsed_text = {}
        parsed_spk = {}
        last_error = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"Bạn là dịch giả phụ đề phim Trung - {lang_name} xuất sắc."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2500,
                )
                parsed_text, parsed_spk = _parse_numbered_result(
                    response.choices[0].message.content, len(batch)
                )
                if len(parsed_text) != len(batch):
                    missing = sorted(set(range(len(batch))) - set(parsed_text))
                    raise RuntimeError(f"AI trả thiếu dòng: {missing}")
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if len(parsed_text) != len(batch):
            print(f"AI batch {batch_number} failed ({last_error}); falling back to Google")
            _fallback_google(batch, translated_map, start_i, target_lang)
            for offset in range(len(batch)):
                speaker_map[start_i + offset] = "M1"
        else:
            for offset, text in parsed_text.items():
                translated_map[start_i + offset] = text
                speaker_map[start_i + offset] = parsed_spk.get(offset, "M1")

        progress = min(int(batch_number / len(batches) * 100), 99)
        jobs[job_id].setdefault("translate_progress", {})[target_lang] = progress
        jobs[job_id]["message"] = f"AI ({model}) đang dịch sang {LANGUAGES[target_lang]['name']}: {progress}%"

    # Save detected speakers into job
    speaker_counts = {}
    segment_speakers = []
    for idx in range(len(entries)):
        spk = speaker_map.get(idx, "M1")
        segment_speakers.append(spk)
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    jobs[job_id]["speakers"] = speaker_counts
    jobs[job_id]["segment_speakers"] = segment_speakers

    translated = _rebuild_srt(entries, translated_map)
    valid, errors = validate_srt(translated)
    if not valid:
        raise RuntimeError(f"Bản dịch SRT không hợp lệ: {'; '.join(errors[:3])}")
    return translated


def translate_srt(srt_content: str, target_lang: str, job_id: str) -> str:
    """Translate SRT with Google Translate batches and line-level fallback."""
    from deep_translator import GoogleTranslator

    entries = parse_srt(srt_content)
    if not entries:
        return srt_content
    if target_lang not in LANGUAGES:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ: {target_lang}")

    translator = GoogleTranslator(source="zh-CN", target=target_lang)
    batches = []
    current = []
    current_len = 0
    for index, entry in enumerate(entries):
        text_len = len(entry[2]) + 1
        if current and (current_len + text_len > 4500 or len(current) >= 50):
            batches.append(current)
            current = []
            current_len = 0
        current.append((index, entry))
        current_len += text_len
    if current:
        batches.append(current)

    translated_map = {}
    for batch_number, batch in enumerate(batches, start=1):
        if jobs.get(job_id, {}).get("cancel"):
            raise RuntimeError("Đã hủy dịch thuật (Stop)")
        texts = [entry[1][2] for entry in batch]
        try:
            results = translator.translate_batch(texts)
            if not isinstance(results, list) or len(results) != len(texts):
                raise RuntimeError("Google trả thiếu dòng")
            for (index, entry), result in zip(batch, results):
                translated_map[index] = result or entry[2]
        except Exception:
            for index, entry in batch:
                try:
                    translated_map[index] = translator.translate(entry[2]) or entry[2]
                except Exception:
                    translated_map[index] = entry[2]
        progress = min(int(batch_number / len(batches) * 100), 99)
        jobs[job_id].setdefault("translate_progress", {})[target_lang] = progress
        jobs[job_id]["message"] = f"Đang dịch sang {LANGUAGES[target_lang]['name']}: {progress}%"

    # Default all to M1 for Google translate
    jobs[job_id]["speakers"] = {"M1": len(entries)}
    jobs[job_id]["segment_speakers"] = ["M1"] * len(entries)

    translated = _rebuild_srt(entries, translated_map)
    valid, errors = validate_srt(translated)
    if not valid:
        raise RuntimeError(f"Bản dịch SRT không hợp lệ: {'; '.join(errors[:3])}")
    return translated
