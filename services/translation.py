"""Subtitle translation providers with strict SRT/coverage validation."""

from __future__ import annotations

import re
import time

from config import AI_LANG_NAMES, AI_TRANSLATE_CONFIG, LANGUAGES, jobs
from services.srt_utils import parse_srt, validate_srt

_NUMBERED_LINE = re.compile(r"^\s*(\d+)[.)]\s*(.*?)\s*$")


def _rebuild_srt(entries, translated_map: dict[int, str]) -> str:
    lines = []
    for index, (srt_index, timestamp, text) in enumerate(entries):
        lines.extend([srt_index, timestamp, translated_map.get(index, text), ""])
    return "\n".join(lines)


def _parse_numbered_result(result_text: str, batch_size: int) -> dict[int, str]:
    parsed = {}
    for raw_line in str(result_text or "").splitlines():
        match = _NUMBERED_LINE.match(raw_line)
        if not match:
            continue
        number = int(match.group(1)) - 1
        value = match.group(2).strip()
        if 0 <= number < batch_size and value:
            parsed[number] = value
    return parsed


def _fallback_google(batch, translated_map, start_i, target_lang):
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


def translate_srt_ai(srt_content: str, target_lang: str, job_id: str) -> str:
    """Translate numbered batches with retries and Google fallback for failed lines."""
    from openai import OpenAI

    entries = parse_srt(srt_content)
    if not entries:
        return srt_content
    if target_lang not in LANGUAGES:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ: {target_lang}")
    if not AI_TRANSLATE_CONFIG["api_key"]:
        raise RuntimeError("AI translation API key chưa được cấu hình")

    client = OpenAI(base_url=AI_TRANSLATE_CONFIG["base_url"], api_key=AI_TRANSLATE_CONFIG["api_key"])
    lang_name = AI_LANG_NAMES.get(target_lang, target_lang)
    batch_size = 15
    batches = [(start, entries[start:start + batch_size]) for start in range(0, len(entries), batch_size)]
    translated_map: dict[int, str] = {}

    for batch_number, (start_i, batch) in enumerate(batches, start=1):
        if jobs.get(job_id, {}).get("cancel"):
            raise RuntimeError("Đã hủy dịch thuật (Stop)")
        numbered = "\n".join(f"{offset + 1}. {entry[2]}" for offset, entry in enumerate(batch))
        prompt = (
            f"Translate the following Chinese subtitle lines to {lang_name}. "
            "Keep the same numbered format. Translate naturally and concisely for subtitles. "
            "Return exactly one translated line for every input number and no explanations.\n\n"
            f"{numbered}"
        )
        parsed = {}
        last_error = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=AI_TRANSLATE_CONFIG["model"],
                    messages=[
                        {"role": "system", "content": f"You are a professional Chinese-to-{lang_name} subtitle translator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )
                parsed = _parse_numbered_result(response.choices[0].message.content, len(batch))
                if len(parsed) != len(batch):
                    missing = sorted(set(range(len(batch))) - set(parsed))
                    raise RuntimeError(f"AI trả thiếu dòng: {missing}")
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if len(parsed) != len(batch):
            print(f"AI batch {batch_number} failed; falling back to Google: {last_error}")
            _fallback_google(batch, translated_map, start_i, target_lang)
        else:
            translated_map.update({start_i + offset: text for offset, text in parsed.items()})

        progress = min(int(batch_number / len(batches) * 100), 99)
        jobs[job_id].setdefault("translate_progress", {})[target_lang] = progress
        jobs[job_id]["message"] = f"AI đang dịch sang {LANGUAGES[target_lang]['name']}: {progress}%"

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

    translated = _rebuild_srt(entries, translated_map)
    valid, errors = validate_srt(translated)
    if not valid:
        raise RuntimeError(f"Bản dịch SRT không hợp lệ: {'; '.join(errors[:3])}")
    return translated
