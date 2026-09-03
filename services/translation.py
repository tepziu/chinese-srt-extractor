"""Subtitle translation with multi-speaker detection, strict word budgeting, and adaptive auto-compress."""

from __future__ import annotations

import re
import time

import json
from pathlib import Path
from config import AI_DEFAULT_MODEL, AI_LANG_NAMES, AI_TRANSLATE_CONFIG, DEFAULT_TRANSLATION_MODE, LANGUAGES, OUTPUT_FOLDER, TRANSLATION_MODES, jobs
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


def _auto_compress_long_lines(
    client,
    model: str,
    lang_name: str,
    long_items: list[tuple[int, str, str, int, float]],
    translated_map: dict[int, str],
    speaker_map: dict[int, str],
    translation_mode: str = "movie",
) -> None:
    """Adaptive Pass: Automatically compress lines that exceed the target speaking budget."""
    if not long_items:
        return

    compress_lines = []
    for offset, (idx, spk, curr_text, max_w, dur_s) in enumerate(long_items):
        compress_lines.append(
            f"{offset + 1}. [Thời lượng: {dur_s}s | BẮT BUỘC TỐI ĐA {max_w} TỪ] [{spk}] {curr_text}"
        )
    if translation_mode == "driving":
        sys_prompt = f"Bạn là chuyên gia biên tập lời thuyết minh dạy lái xe tiếng {lang_name} siêu ngắn gọn, dứt khoát và chuẩn kỹ thuật."
        compress_prompt = (
            f"Các câu hướng dẫn mẹo lái xe tiếng {lang_name} sau đây đang bị QUÁ DÀI so với thời lượng thao tác trong video.\n"
            f"Hãy viết lại từng câu thành MỘT CÂU HƯỚNG DẪN NGẮN GỌN HƠN, dứt khoát, chuẩn thuật ngữ lái xe/ô tô, giữ nguyên 100% ý chính và chỉ dẫn kỹ thuật.\n"
            "QUY TẮC:\n"
            "1. BẮT BUỘC KHÔNG VƯỢT QUÁ số từ tối đa đã ghi trong ngoặc.\n"
            "2. BẮT BUỘC giữ nguyên mã [M1] ở đầu câu (đây là video 1 người nói).\n"
            "3. Chỉ trả về danh sách đánh số, KHÔNG giải thích thêm.\n\n"
            + "\n".join(compress_lines)
        )
    else:
        sys_prompt = f"Bạn là chuyên gia biên tập rút gọn lời thoại phim {lang_name} siêu ngắn gọn và tự nhiên."
        compress_prompt = (
            f"Các câu thoại tiếng {lang_name} sau đây đang bị QUÁ DÀI so với thời lượng nhân vật nói trong video.\n"
            f"Hãy viết lại từng câu thành MỘT CÂU NGẮN GỌN HƠN, xúc tích, giữ nguyên 100% ý chính và xưng hô.\n"
            "QUY TẮC:\n"
            "1. BẮT BUỘC KHÔNG VƯỢT QUÁ số từ tối đa đã ghi trong ngoặc.\n"
            "2. Giữ nguyên mã nhân vật [M1], [F1], [M2], [F2], [N] ở đầu câu.\n"
            "3. Chỉ trả về danh sách đánh số, KHÔNG giải thích thêm.\n\n"
            + "\n".join(compress_lines)
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": compress_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        parsed_comp, parsed_spk = _parse_numbered_result(response.choices[0].message.content, len(long_items))
        for offset, (idx, orig_spk, orig_text, max_w, dur_s) in enumerate(long_items):
            if offset in parsed_comp:
                new_text = parsed_comp[offset]
                # If compressed text is actually shorter, apply it!
                if len(new_text.split()) < len(orig_text.split()):
                    print(f"  ✨ Auto-compressed line {idx+1}: '{orig_text}' ({len(orig_text.split())} words) -> '{new_text}' ({len(new_text.split())} words, max {max_w})")
                    translated_map[idx] = new_text
                    if translation_mode == "driving":
                        speaker_map[idx] = "M1"
                    elif offset in parsed_spk:
                        speaker_map[idx] = parsed_spk[offset]
    except Exception as exc:
        print(f"⚠️ Auto-compress error ({exc}), keeping original translations")


def translate_srt_ai(srt_content: str, target_lang: str, job_id: str, ai_model: str | None = None, translation_mode: str | None = None) -> str:
    """Translate SRT using AI with strict word budgeting and adaptive auto-compress."""
    from openai import OpenAI

    entries = parse_srt(srt_content)
    if not entries:
        return srt_content
    if target_lang not in LANGUAGES:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ: {target_lang}")
    if not AI_TRANSLATE_CONFIG["api_key"]:
        raise RuntimeError("AI translation API key chưa được cấu hình")

    mode = translation_mode or jobs.get(job_id, {}).get("translation_mode") or DEFAULT_TRANSLATION_MODE
    if mode not in TRANSLATION_MODES:
        mode = DEFAULT_TRANSLATION_MODE
    if jobs.get(job_id):
        jobs[job_id]["translation_mode"] = mode

    model = ai_model or jobs.get(job_id, {}).get("ai_model") or AI_TRANSLATE_CONFIG.get("model") or AI_DEFAULT_MODEL
    client = OpenAI(base_url=AI_TRANSLATE_CONFIG["base_url"], api_key=AI_TRANSLATE_CONFIG["api_key"])
    lang_name = AI_LANG_NAMES.get(target_lang, target_lang)

    # Parse timing to compute accurate available duration and target word budget
    timings = parse_srt_timing(srt_content)
    durations = {}
    word_budgets = {}
    for i in range(len(timings)):
        s_ms, e_ms, _ = timings[i]
        next_s_ms = timings[i + 1][0] if i + 1 < len(timings) else e_ms + 2000
        avail_ms = next_s_ms - s_ms
        avail_s = max(0.8, round(avail_ms / 1000, 1))
        durations[i] = avail_s
        # 3.2 words/second is standard comfortable speaking rate for Vietnamese
        word_budgets[i] = max(2, int(avail_s * 3.2))

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
            dur_s = durations.get(idx, 2.0)
            max_w = word_budgets.get(idx, 6)
            dur_info = f"[Thời lượng: {dur_s}s | TỐI ĐA: {max_w} TỪ]"
            numbered_lines.append(f"{offset + 1}. {dur_info} {entry[2]}")
        numbered_text = "\n".join(numbered_lines)

        if mode == "driving":
            sys_prompt = (
                f"Bạn là chuyên gia đào tạo lái xe ô tô và biên tập viên video mẹo lái xe thực chiến Trung - {lang_name}. "
                f"Bạn chuyên dịch lời thoại hướng dẫn kỹ thuật lái xe, mẹo lái xe an toàn, căn đường, đỗ xe ngắn gọn, "
                f"dứt khoát, trực quan và chuẩn xác thuật ngữ chuyên ngành ô tô."
            )
            prompt = (
                f"Bạn là chuyên gia dịch thuật video dạy lái xe và mẹo lái xe ô tô chuyên nghiệp (Trung sang {lang_name}).\n"
                f"Hãy dịch các câu thuyết minh tiếng Trung sau sang {lang_name} theo phong cách hướng dẫn lái xe thực tế, dứt khoát, dễ hiểu.\n\n"
                "QUY TẮC BẮT BUỘC:\n"
                "1. ĐỒNG NHẤT 1 NGƯỜI NÓI [M1]:\n"
                "   - Đây là video do 1 người hướng dẫn / thầy dạy lái xe thuyết minh duy nhất.\n"
                "   - BẮT BUỘC gắn mã [M1] ở đầu MỌI câu dịch (Ví dụ: '1. [M1] Khi lùi chuồng, nhìn gương chiếu hậu trái...').\n"
                "   - TUYỆT ĐỐI KHÔNG dùng F1, M2, F2 vì video chỉ có một người nói duy nhất.\n"
                "2. THUẬT NGỮ LÁI XE & Ô TÔ CHUẨN XÁC:\n"
                "   - Sử dụng chính xác thuật ngữ kỹ thuật lái xe và giao thông của người Việt:\n"
                "     * Thao tác: côn, ga, phanh (chân phanh, phanh tay), cần số, về số N, số D, số R, số P.\n"
                "     * Vô lăng: đánh lái, trả lái, đánh kịch lái (hết lái), đánh chết lái, giữ thẳng lái, ôm cua.\n"
                "     * Điểm chuẩn: gương chiếu hậu (gương trái/phải/giữa), góc chữ A, điểm mù, vạch kẻ đường, vỉa hè, bó vỉa, tim đường.\n"
                "     * Đỗ xe: ghép chuồng dọc (lùi chuồng), ghép chuồng ngang (đỗ song song), de xe, tiến/lùi.\n"
                "     * Tín hiệu: xi-nhan, gạt mưa, đèn pha/cốt, đèn cảnh báo nguy hiểm (hazard).\n"
                "3. VĂN PHONG HƯỚNG DẪN DỨT KHOÁT, THỰC CHIẾN:\n"
                "   - Dùng các câu ngắn, dứt khoát, nhắm thẳng vào hành động và quan sát ('Hãy quan sát...', 'Lập tức phanh...', 'Căn chuẩn vỉa hè...').\n"
                "   - Xưng hô phù hợp video hướng dẫn: dùng lối nói trực tiếp hoặc 'chúng ta', 'bạn', 'các bác', 'anh em'. Tuyệt đối TRÁNH xưng hô tình cảm sướt mướt kiểu phim truyền hình (anh/em, tiểu thư, công tử,...).\n"
                "4. NGÂN SÁCH SỐ TỪ: BẮT BUỘC KHÔNG VƯỢT QUÁ số từ tối đa trong ngoặc. Lời nói phải khớp thời lượng thao tác trên hình ảnh, KHÔNG ĐƯỢC TRÀN TIẾNG.\n"
                "5. ĐỊNH DẠNG: Chỉ trả về danh sách đánh số theo mẫu: '1. [M1] Lời dịch', KHÔNG giải thích thêm.\n\n"
                f"{numbered_text}"
            )
        else:
            sys_prompt = f"Bạn là dịch giả phụ đề phim Trung - {lang_name} xuất sắc, chuyên gia khống chế độ dài câu thoại."
            prompt = (
                f"Bạn là chuyên gia dịch thuật phụ đề và lồng tiếng phim chuyên nghiệp (Trung sang {lang_name}).\n"
                f"Hãy dịch các câu thoại tiếng Trung sau sang {lang_name} tự nhiên, đúng ngữ cảnh đối thoại.\n\n"
                "QUY TẮC BẮT BUỘC:\n"
                "1. PHÂN VAI NHÂN VẬT: Gắn mã nhân vật vào đầu mỗi câu dịch:\n"
                "   - [M1] = Nam chính\n"
                "   - [F1] = Nữ chính\n"
                "   - [M2] = Nam phụ\n"
                "   - [F2] = Nữ phụ\n"
                "   - [N]  = Dẫn chuyện / Thuyết minh\n"
                "2. NGÂN SÁCH SỐ TỪ: BẮT BUỘC KHÔNG ĐƯỢC VƯỢT QUÁ số từ tối đa ghi trong ngoặc. Dịch cô đọng, súc tích, gãy gọn để khi lồng tiếng đọc vừa khít thời lượng, KHÔNG BỊ TRÀN TIẾNG.\n"
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
                        {"role": "system", "content": sys_prompt},
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
            # Check for any line that exceeded its word budget by > 30%
            long_items_in_batch = []
            for offset, text in parsed_text.items():
                idx = start_i + offset
                spk = parsed_spk.get(offset, "M1")
                translated_map[idx] = text
                speaker_map[idx] = spk

                words_count = len(text.split())
                max_allowed = word_budgets.get(idx, 6)
                dur_s = durations.get(idx, 2.0)
                if words_count > max(max_allowed + 2, int(max_allowed * 1.35)):
                    long_items_in_batch.append((idx, spk, text, max_allowed, dur_s))

            # Trigger Adaptive Auto-Compress if needed
            if long_items_in_batch:
                _auto_compress_long_lines(client, model, lang_name, long_items_in_batch, translated_map, speaker_map, translation_mode=mode)

        progress = min(int(batch_number / len(batches) * 100), 99)
        jobs[job_id].setdefault("translate_progress", {})[target_lang] = progress
        mode_label = TRANSLATION_MODES.get(mode, {}).get("name", mode)
        jobs[job_id]["message"] = f"AI ({model} | {mode_label}) đang dịch sang {LANGUAGES[target_lang]['name']}: {progress}%"

    # Save detected speakers into job
    speaker_counts = {}
    segment_speakers = []
    for idx in range(len(entries)):
        spk = "M1" if mode == "driving" else speaker_map.get(idx, "M1")
        segment_speakers.append(spk)
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    jobs[job_id]["speakers"] = speaker_counts
    jobs[job_id]["segment_speakers"] = segment_speakers
    jobs[job_id]["translation_mode"] = mode
    try:
        speakers_dir = OUTPUT_FOLDER / job_id
        speakers_dir.mkdir(parents=True, exist_ok=True)
        speakers_file = speakers_dir / "speakers.json"
        speakers_file.write_text(json.dumps({
            "speakers": speaker_counts,
            "segment_speakers": segment_speakers,
            "translation_mode": mode,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to save speakers.json: {exc}")

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
