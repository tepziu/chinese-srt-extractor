"""Hybrid subtitle contracts: deterministic timing, AI semantic fusion, safe fallbacks."""
from __future__ import annotations

import json
import re
from typing import Any

from services.srt_utils import generate_srt, parse_srt, parse_srt_timing

_ALLOWED_SPEAKERS = {"M1", "F1", "M2", "F2", "N"}
_SENTENCE_END = set("。！？!?；;")


def cues_from_srt(content: str) -> list[dict[str, Any]]:
    timings = parse_srt_timing(content)
    if not timings:
        raise ValueError("Không có cue SRT hợp lệ")
    return [
        {"id": index, "start_ms": start, "end_ms": end, "text": text}
        for index, (start, end, text) in enumerate(timings, start=1)
    ]


def parse_fusion_response(raw: str) -> list[dict[str, Any]]:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = min([pos for pos in (text.find("{"), text.find("[")) if pos >= 0], default=-1)
    if start < 0:
        raise ValueError("AI không trả JSON")
    end = max(text.rfind("}"), text.rfind("]"))
    if end < start:
        raise ValueError("JSON AI bị thiếu phần kết thúc")
    data = json.loads(text[start:end + 1])
    groups = data.get("groups") if isinstance(data, dict) else data
    if not isinstance(groups, list) or not groups:
        raise ValueError("AI không trả danh sách groups")
    return groups


def validate_fusion_groups(groups: list[dict[str, Any]], cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {cue["id"]: cue for cue in cues}
    expected = list(by_id)
    used: list[int] = []
    normalized = []
    for position, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise ValueError(f"Group {position} không phải object")
        ids = group.get("source_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(value, int) for value in ids):
            raise ValueError(f"Group {position} thiếu source_ids hợp lệ")
        if ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError(f"Group {position} có source_ids không liên tiếp")
        if any(value not in by_id for value in ids):
            raise ValueError(f"Group {position} chứa source ID không tồn tại")
        if any(value in used for value in ids):
            raise ValueError("Mỗi source ID phải xuất hiện đúng một lần")
        gaps = [by_id[right]["start_ms"] - by_id[left]["end_ms"] for left, right in zip(ids, ids[1:])]
        if any(gap > 800 for gap in gaps):
            raise ValueError(f"Group {position} gộp qua khoảng nghỉ lớn")
        if by_id[ids[-1]]["end_ms"] - by_id[ids[0]]["start_ms"] > 10000:
            raise ValueError(f"Group {position} dài quá 10 giây")
        translation = str(group.get("translation") or "").strip()
        if not translation:
            raise ValueError(f"Group {position} thiếu bản dịch")
        used.extend(ids)
        first, last = by_id[ids[0]], by_id[ids[-1]]
        speaker = str(group.get("speaker") or "M1").upper()
        if speaker not in _ALLOWED_SPEAKERS:
            speaker = "M1"
        normalized.append({
            "source_ids": ids,
            "source_text": "".join(by_id[value]["text"] for value in ids),
            "translation": " ".join(translation.split()),
            "speaker": speaker,
            "start_ms": first["start_ms"],
            "end_ms": last["end_ms"],
        })
    if len(used) != len(set(used)):
        raise ValueError("Mỗi source ID phải xuất hiện đúng một lần")
    if used != expected:
        missing = [value for value in expected if value not in used]
        if missing:
            raise ValueError(f"Kết quả AI thiếu source ID: {missing[:12]}")
        raise ValueError("Các source ID không giữ đúng thứ tự")
    return normalized


def groups_to_srt(groups: list[dict[str, Any]]) -> str:
    return generate_srt([
        {"start": group["start_ms"] / 1000, "end": group["end_ms"] / 1000,
         "text": group["translation"]}
        for group in groups
    ])


def _join_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if left.endswith(("...", "…")):
        left = left.rstrip(". …")
    if right.startswith(("...", "…")):
        right = right.lstrip(". …")
    return (left + " " + right).strip()


_SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"'”’»」』)\]]*(?=\s|$)")


def split_text_sentences(text: str) -> list[str]:
    """Split translated text into complete sentences, preserving terminal punctuation.

    Subtitle output is intentionally sentence-oriented: one cue should carry one
    complete thought instead of two or more sentences joined together. Text with
    no sentence punctuation remains a single cue because there is no safe split.
    """
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return []
    sentences = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(normalized):
        piece = normalized[start:match.end()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    remainder = normalized[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences or [normalized]


def _split_source_ids(source_ids: list[int], sentence_count: int, weights: list[int]) -> list[list[int]]:
    """Partition source IDs where possible without inventing or reordering IDs."""
    if not source_ids:
        return [[] for _ in range(sentence_count)]
    if len(source_ids) < sentence_count:
        return [[source_ids[index]] if index < len(source_ids) else []
                for index in range(sentence_count)]
    chunks = []
    cursor = 0
    remaining_weight = sum(weights)
    for index, weight in enumerate(weights):
        remaining_sentences = sentence_count - index
        remaining_ids = len(source_ids) - cursor
        if remaining_sentences == 1:
            take = remaining_ids
        else:
            take = max(1, round(remaining_ids * weight / max(1, remaining_weight)))
            take = min(take, remaining_ids - (remaining_sentences - 1))
        chunks.append(source_ids[cursor:cursor + take])
        cursor += take
        remaining_weight -= weight
    return chunks


def _has_sentence_terminal(text: str) -> bool:
    return bool(re.search(r"[.!?。！？]+[\"'”’»」』)\]]*$", str(text or "").strip()))


def _merge_fragment_groups(groups: list[dict[str, Any]], *, max_duration_ms: int = 9000,
                           max_gap_ms: int = 400, max_words: int = 26) -> list[dict[str, Any]]:
    """Join adjacent fragments when they form one complete sentence."""
    if not groups:
        return []
    result = [dict(groups[0])]
    for group in groups[1:]:
        previous = result[-1]
        gap = group["start_ms"] - previous["end_ms"]
        combined = _join_text(previous["translation"], group["translation"])
        safe = (
            previous.get("speaker") == group.get("speaker")
            and 0 <= gap <= max_gap_ms
            and group["end_ms"] - previous["start_ms"] <= max_duration_ms
            and len(combined.split()) <= max_words
            and not _has_sentence_terminal(previous["translation"])
            and len(split_text_sentences(combined)) == 1
        )
        if safe:
            previous["source_ids"] = list(previous.get("source_ids") or []) + list(group.get("source_ids") or [])
            previous["source_text"] = previous.get("source_text", "") + group.get("source_text", "")
            previous["translation"] = combined
            previous["end_ms"] = group["end_ms"]
        else:
            result.append(dict(group))
    return result


def split_groups_to_single_sentences(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make each final subtitle cue one complete sentence, merging source fragments first."""
    split_groups = []
    for group in groups:
        sentences = split_text_sentences(group.get("translation", ""))
        if len(sentences) <= 1:
            split_groups.append(dict(group))
            continue
        weights = [max(1, len(sentence.split())) for sentence in sentences]
        total_weight = sum(weights)
        start_ms = int(group["start_ms"])
        end_ms = int(group["end_ms"])
        duration = max(1, end_ms - start_ms)
        ids_chunks = _split_source_ids(list(group.get("source_ids") or []), len(sentences), weights)
        cursor = start_ms
        for index, (sentence, weight) in enumerate(zip(sentences, weights)):
            child_end = end_ms if index == len(sentences) - 1 else cursor + max(1, round(duration * weight / total_weight))
            child = dict(group)
            child["source_ids"] = ids_chunks[index]
            child["source_text"] = group.get("source_text", "") if index == 0 else ""
            child["translation"] = sentence
            child["start_ms"] = cursor
            child["end_ms"] = max(cursor + 1, child_end)
            split_groups.append(child)
            cursor = child["end_ms"]
    return _merge_fragment_groups(split_groups)


def build_tts_groups(display_groups: list[dict[str, Any]], *, max_duration_ms: int = 9000,
                     max_gap_ms: int = 400, max_words: int = 26) -> list[dict[str, Any]]:
    if not display_groups:
        return []
    result = [dict(display_groups[0])]
    for group in display_groups[1:]:
        previous = result[-1]
        gap = group["start_ms"] - previous["end_ms"]
        combined_text = _join_text(previous["translation"], group["translation"])
        if (previous.get("semantic_parent_translation")
                and previous.get("semantic_parent_translation") == group.get("semantic_parent_translation")):
            combined_text = previous["semantic_parent_translation"]
        duration = group["end_ms"] - previous["start_ms"]
        safe = (previous.get("speaker") == group.get("speaker") and 0 <= gap <= max_gap_ms
                and duration <= max_duration_ms and len(combined_text.split()) <= max_words
                and len(split_text_sentences(combined_text)) <= 1)
        if safe:
            previous["source_ids"] = previous["source_ids"] + group["source_ids"]
            previous["source_text"] += group.get("source_text", "")
            previous["translation"] = combined_text
            previous["end_ms"] = group["end_ms"]
        else:
            result.append(dict(group))
    return result



def display_srt_to_tts_srt(display_srt: str, speakers: list[str] | None = None) -> tuple[str, list[str]]:
    cues = cues_from_srt(display_srt)
    groups = []
    for index, cue in enumerate(cues):
        speaker = (speakers[index] if speakers and index < len(speakers) else "M1").upper()
        if speaker not in _ALLOWED_SPEAKERS:
            speaker = "M1"
        groups.append({
            "source_ids": [cue["id"]], "source_text": cue["text"],
            "translation": cue["text"], "speaker": speaker,
            "start_ms": cue["start_ms"], "end_ms": cue["end_ms"],
        })
    groups = split_groups_to_single_sentences(groups)
    tts_groups = build_tts_groups(groups)
    return groups_to_srt(tts_groups), [group["speaker"] for group in tts_groups]

def heuristic_source_groups(cues: list[dict[str, Any]], *, max_duration_ms: int = 6000,
                            max_gap_ms: int = 350, max_chars: int = 52) -> list[dict[str, Any]]:
    groups = []
    current = None
    for cue in cues:
        if current is None:
            current = {"source_ids": [cue["id"]], "text": cue["text"],
                       "start_ms": cue["start_ms"], "end_ms": cue["end_ms"]}
            continue
        gap = cue["start_ms"] - current["end_ms"]
        duration = cue["end_ms"] - current["start_ms"]
        previous_ended = bool(current["text"] and current["text"][-1] in _SENTENCE_END)
        can_merge = (0 <= gap <= max_gap_ms and duration <= max_duration_ms
                     and len(current["text"] + cue["text"]) <= max_chars and not previous_ended)
        if can_merge:
            current["source_ids"].append(cue["id"])
            current["text"] += cue["text"]
            current["end_ms"] = cue["end_ms"]
        else:
            groups.append(current)
            current = {"source_ids": [cue["id"]], "text": cue["text"],
                       "start_ms": cue["start_ms"], "end_ms": cue["end_ms"]}
    if current:
        groups.append(current)
    return groups


def _grouped_source_srt(cues: list[dict[str, Any]], source_groups: list[dict[str, Any]]) -> str:
    return generate_srt([{"start": group["start_ms"] / 1000, "end": group["end_ms"] / 1000,
                          "text": group["text"]} for group in source_groups])


def _translate_grouped_fallback(cues, target_lang, job_id, translation_mode):
    from services.translation import translate_srt_ai
    source_groups = heuristic_source_groups(cues)
    grouped_srt = _grouped_source_srt(cues, source_groups)
    return translate_srt_ai(grouped_srt, target_lang, job_id, translation_mode=translation_mode)



def evaluate_fusion_quality(groups: list[dict[str, Any]], target_lang: str) -> dict[str, Any]:
    warnings = []
    severe = []
    fragment_endings = {"the", "a", "an", "and", "or", "to", "of", "if", "with", "for"}
    for index, group in enumerate(groups, start=1):
        text = group["translation"].strip()
        duration_s = max(0.001, (group["end_ms"] - group["start_ms"]) / 1000)
        cps = len(text) / duration_s
        if cps > 25:
            warnings.append({"group": index, "type": "high_cps", "value": round(cps, 1)})
        if duration_s < 0.65:
            warnings.append({"group": index, "type": "short_duration", "value": round(duration_s, 2)})
        last_word = re.sub(r"[^A-Za-z]", "", text.split()[-1]).lower() if text.split() else ""
        if text.endswith(("...", "…")) or last_word in fragment_endings:
            severe.append({"group": index, "type": "sentence_fragment"})
        if target_lang in {"en", "vi", "id"} and re.search(r"[\u4e00-\u9fff]", text):
            severe.append({"group": index, "type": "han_residue"})
    return {"warnings": warnings, "severe": severe, "needs_review": bool(severe)}

def _prompt_payload(cues, target_lang, translation_mode, whisper_context=None):
    compact = [{"id": cue["id"], "start_ms": cue["start_ms"], "end_ms": cue["end_ms"],
                "text": cue["text"]} for cue in cues]
    return f"""You are a professional subtitle editor and translator.
Target language: {target_lang}. Style: {translation_mode}.
Merge adjacent source cues into complete natural thoughts, correct obvious OCR/ASR mistakes from context, and translate them.
Return STRICT JSON only: {{"groups":[{{"source_ids":[1,2],"translation":"...","speaker":"F1"}}]}}.
Rules:
- Cover every source id exactly once, in original order.
- Each group's source_ids must be contiguous.
- Do not output timestamps. Python owns timing.
- Each group must contain exactly one complete sentence or one short sentence-like utterance; never put two sentences in one translation.
- Prefer complete thoughts lasting about 2-6 seconds; never merge across a pause over 800 ms or scene/topic changes.
- Keep English concise at roughly 2.6 spoken words/second (Vietnamese/Indonesian roughly 3.4 words/second) using each group's start/end budget.
- If a long sentence contains two meaningful clauses, split it at a natural semantic boundary instead of forcing both clauses into one cue.
- Preserve names, facts and intent. Do not invent dialogue.
- Speaker must be one of M1,F1,M2,F2,N. Use M1 when uncertain.
- Avoid sentence fragments that end with function words or ellipses when a safe merge is possible.
Source cues JSON:
{json.dumps(compact, ensure_ascii=False)}
Audio transcript for cross-check only (may contain recognition errors):
{whisper_context or 'not provided'}"""


_MAX_WORDS_BY_LANG = {"en": 18, "vi": 22, "id": 20}


def _max_subtitle_words(target_lang: str) -> int:
    return _MAX_WORDS_BY_LANG.get(target_lang, 18)


def _needs_semantic_segmentation(group: dict[str, Any], target_lang: str) -> bool:
    text = str(group.get("translation") or "").strip()
    words = len(text.split())
    if words > _max_subtitle_words(target_lang):
        return True
    if words <= 14:
        return False
    return bool(re.search(r",|;|:|\b(and|but|because|if|when|while|so)\b", text, re.IGNORECASE))


def _call_segmentation_model(groups: list[dict[str, Any]], target_lang: str,
                             translation_mode: str, model: str | None = None) -> str:
    """Ask the selected translation model to split long cues at semantic boundaries."""
    from openai import OpenAI
    from config import AI_TRANSLATE_CONFIG, AI_DEFAULT_MODEL
    if not AI_TRANSLATE_CONFIG.get("api_key"):
        raise RuntimeError("Chưa cấu hình AI translation gateway để phân tích cue")
    payload = [{
        "index": index,
        "duration_ms": group["end_ms"] - group["start_ms"],
        "source_text": group.get("source_text", ""),
        "translation": group["translation"],
    } for index, group in enumerate(groups, start=1)]
    prompt = f"""You are a senior subtitle segmentation editor.
Target language: {target_lang}. Style: {translation_mode}.
Analyze each translated subtitle and split it only at a natural semantic boundary when it contains two meaningful ideas or clauses.
The goal is readable display subtitles and accurate narration timing, not a mechanical word limit.
Rules:
- Return STRICT JSON only: {{"items":[{{"index":1,"segments":[{{"text":"...","weight":1}}]}}]}}.
- Each segment must express one meaningful idea and be understandable in context; a clause without final punctuation is allowed when it is a natural unit.
- Preserve names, facts, numbers, warnings and technical meaning. Do not add or remove information.
- Keep the original wording as much as possible; only make a tiny grammatical adjustment when needed after splitting.
- Return one segment unchanged when splitting would make the subtitle less natural.
- Do not put two independent ideas back into one segment.
Example: split `If you often struggle to time your lane changes, this video will clearly explain the core rules of safe lane changing.` into `If you often struggle to time your lane changes` and `this video will clearly explain the core rules of safe lane changing.`.
Items:
{json.dumps(payload, ensure_ascii=False)}"""
    client = OpenAI(base_url=AI_TRANSLATE_CONFIG["base_url"], api_key=AI_TRANSLATE_CONFIG["api_key"])
    response = client.chat.completions.create(
        model=model or AI_TRANSLATE_CONFIG.get("model") or AI_DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=3500,
    )
    return response.choices[0].message.content or ""


def _parse_segmentation_response(raw: str) -> dict[int, list[str]]:
    text = re.sub(r"^```(?:json)?\s*", "", str(raw or "").strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI phân tích cue không trả JSON hợp lệ")
    data = json.loads(text[start:end + 1])
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("AI phân tích cue thiếu items")
    result: dict[int, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        raw_segments = item.get("segments")
        if not isinstance(raw_segments, list):
            continue
        segments = []
        for segment in raw_segments:
            value = segment.get("text") if isinstance(segment, dict) else segment
            value = " ".join(str(value or "").split())
            if value:
                segments.append(value)
        if index > 0 and segments:
            result[index] = segments
    return result


def _allocate_segment_groups(group: dict[str, Any], segments: list[str]) -> list[dict[str, Any]]:
    if len(segments) <= 1:
        return [dict(group)]
    weights = [max(1, len(segment.split())) for segment in segments]
    total_weight = sum(weights)
    start_ms = int(group["start_ms"])
    end_ms = int(group["end_ms"])
    duration = max(1, end_ms - start_ms)
    ids_chunks = _split_source_ids(list(group.get("source_ids") or []), len(segments), weights)
    result = []
    cursor = start_ms
    for index, (segment, weight) in enumerate(zip(segments, weights)):
        child_end = end_ms if index == len(segments) - 1 else cursor + max(1, round(duration * weight / total_weight))
        child = dict(group)
        child["source_ids"] = ids_chunks[index]
        child["source_text"] = group.get("source_text", "") if index == 0 else ""
        child["translation"] = segment
        child["start_ms"] = cursor
        child["end_ms"] = max(cursor + 1, child_end)
        child["semantic_split"] = True
        child["semantic_parent_translation"] = group["translation"]
        result.append(child)
        cursor = child["end_ms"]
    return result


def segment_overlong_groups(groups: list[dict[str, Any]], target_lang: str,
                            translation_mode: str, model: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the selected AI model to split long cues into meaningful semantic units."""
    candidates = [group for group in groups if _needs_semantic_segmentation(group, target_lang)]
    state = {
        "model": model,
        "attempted": len(candidates),
        "changed": 0,
        "segments_created": 0,
        "warning": None,
    }
    if not candidates:
        return groups, state
    try:
        segmented = _parse_segmentation_response(
            _call_segmentation_model(candidates, target_lang, translation_mode, model)
        )
    except Exception as exc:
        state["warning"] = str(exc)
        return groups, state
    result = []
    candidate_cursor = 0
    for group in groups:
        if not _needs_semantic_segmentation(group, target_lang):
            result.append(dict(group))
            continue
        candidate_cursor += 1
        segments = segmented.get(candidate_cursor) or [group["translation"]]
        valid = all(
            segment and not (target_lang in {"en", "vi", "id"} and re.search(r"[\u4e00-\u9fff]", segment))
            and len(split_text_sentences(segment)) <= 1
            for segment in segments
        )
        if not valid:
            result.append(dict(group))
            continue
        if len(segments) > 1:
            children = _allocate_segment_groups(group, segments)
            result.extend(children)
            state["changed"] += 1
            state["segments_created"] += len(children)
        else:
            child = dict(group)
            if segments[0] != group["translation"]:
                child["translation"] = segments[0]
                state["changed"] += 1
            result.append(child)
    return result, state


# Backward-compatible name for callers from the first compaction implementation.
def compact_overlong_groups(groups: list[dict[str, Any]], target_lang: str,
                            translation_mode: str, model: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return segment_overlong_groups(groups, target_lang, translation_mode, model)


def _call_fusion_model(cues, target_lang, translation_mode, whisper_context=None, model=None):
    from openai import OpenAI
    from config import AI_TRANSLATE_CONFIG, AI_DEFAULT_MODEL
    if not AI_TRANSLATE_CONFIG.get("api_key"):
        raise RuntimeError("Chưa cấu hình AI translation gateway")
    client = OpenAI(base_url=AI_TRANSLATE_CONFIG["base_url"], api_key=AI_TRANSLATE_CONFIG["api_key"])
    response = client.chat.completions.create(
        model=model or AI_TRANSLATE_CONFIG.get("model") or AI_DEFAULT_MODEL,
        messages=[{"role": "user", "content": _prompt_payload(cues, target_lang, translation_mode, whisper_context)}],
        temperature=0.2,
        max_tokens=8000,
    )
    return response.choices[0].message.content or ""


def semantic_fuse_translate(source_srt: str, target_lang: str, job_id: str, *,
                            translation_mode: str = "movie", whisper_context: str | None = None,
                            model: str | None = None) -> dict[str, Any]:
    from config import jobs
    cues = cues_from_srt(source_srt)
    used_fallback = False
    warning = None
    try:
        raw = _call_fusion_model(cues, target_lang, translation_mode, whisper_context, model)
        display_groups = validate_fusion_groups(parse_fusion_response(raw), cues)
    except Exception as exc:
        used_fallback = True
        warning = str(exc)
        source_groups = heuristic_source_groups(cues)
        translated_srt = _translate_grouped_fallback(cues, target_lang, job_id, translation_mode)
        translated_entries = parse_srt_timing(translated_srt)
        if len(translated_entries) != len(source_groups):
            raise RuntimeError(f"Fallback semantic translation sai số nhóm: {exc}") from exc
        fallback_groups = []
        for source_group, (_start, _end, text) in zip(source_groups, translated_entries):
            fallback_groups.append({"source_ids": source_group["source_ids"], "translation": text, "speaker": "M1"})
        display_groups = validate_fusion_groups(fallback_groups, cues)
    display_groups = split_groups_to_single_sentences(display_groups)
    display_groups, compaction = segment_overlong_groups(display_groups, target_lang, translation_mode, model)
    quality = evaluate_fusion_quality(display_groups, target_lang)
    tts_groups = build_tts_groups(display_groups)
    display_srt = groups_to_srt(display_groups)
    tts_srt = groups_to_srt(tts_groups)
    from config import OUTPUT_FOLDER
    manifest_dir = OUTPUT_FOLDER / job_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"semantic_fusion_{target_lang}.json"
    manifest_path.write_text(json.dumps({
        "version": "semantic-fusion.v1", "target_lang": target_lang,
        "translation_mode": translation_mode, "source_cues": cues,
        "display_groups": display_groups, "tts_groups": tts_groups,
        "quality": quality, "compaction": compaction,
        "used_fallback": used_fallback, "warning": warning,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    state = {
        "version": "semantic-fusion.v1",
        "source_segments": len(cues),
        "display_segments": len(display_groups),
        "tts_segments": len(tts_groups),
        "used_fallback": used_fallback,
        "warning": warning,
        "quality": quality,
        "compaction": compaction,
        "needs_review": quality["needs_review"],
        "manifest_path": str(manifest_path),
    }
    job = jobs.get(job_id)
    if job:
        job["semantic_fusion"] = state
        job["segment_speakers"] = [group["speaker"] for group in display_groups]
        job["tts_segment_speakers"] = [group["speaker"] for group in tts_groups]
        if warning:
            job.setdefault("translation_provider_warnings", {})[target_lang] = warning[:500]
    return {"display_srt": display_srt, "tts_srt": tts_srt, "groups": display_groups,
            "tts_groups": tts_groups, "used_fallback": used_fallback, "warning": warning}
