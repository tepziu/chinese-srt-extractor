"""SRT parsing, validation, generation and timestamp helpers."""

from __future__ import annotations

import re

_TIMESTAMP_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})(?:\s+.*)?$"
)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format with millisecond carry handling."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments) -> str:
    """Generate SRT from Whisper objects or {start,end,text} dictionaries."""
    srt_lines = []
    index = 0
    for segment in segments:
        start = segment["start"] if isinstance(segment, dict) else segment.start
        end = segment["end"] if isinstance(segment, dict) else segment.end
        text = (segment["text"] if isinstance(segment, dict) else segment.text).strip()
        if not text:
            continue
        start = max(0.0, float(start))
        end = max(start, float(end))
        index += 1
        srt_lines.extend([str(index), f"{format_timestamp(start)} --> {format_timestamp(end)}", text, ""])
    return "\n".join(srt_lines)


def _timestamp_to_ms(value: str) -> int | None:
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$", value.strip())
    if not match:
        return None
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3_600_000 + minutes * 60_000 + seconds * 1000 + millis


def parse_srt(srt_content: str) -> list[tuple[str, str, str]]:
    """Parse SRT blocks, preserving index/timestamp/text for translation."""
    blocks = re.split(r"\r?\n\s*\r?\n+", str(srt_content or "").strip())
    entries = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines()]
        if len(lines) < 3:
            continue
        if not lines[0].strip().isdigit() or "-->" not in lines[1]:
            continue
        text = "\n".join(lines[2:]).strip()
        if text:
            entries.append((lines[0].strip(), lines[1].strip(), text))
    return entries


def validate_srt(srt_content: str, *, duration_ms: int | None = None) -> tuple[bool, list[str]]:
    """Validate ordering, timestamps and text without modifying subtitle content."""
    errors: list[str] = []
    previous_end = -1
    entries = parse_srt(srt_content)
    if not entries:
        return False, ["Không có entry SRT hợp lệ"]
    for position, (_index, timestamp, text) in enumerate(entries, start=1):
        match = _TIMESTAMP_RE.match(timestamp)
        if not match:
            errors.append(f"Entry {position}: timestamp không hợp lệ")
            continue
        start_ms = _timestamp_to_ms(match.group(0).split("-->")[0])
        end_ms = _timestamp_to_ms(match.group(0).split("-->")[1].split()[0])
        if start_ms is None or end_ms is None:
            errors.append(f"Entry {position}: timestamp không hợp lệ")
            continue
        if end_ms <= start_ms:
            errors.append(f"Entry {position}: end phải lớn hơn start")
        if start_ms < previous_end:
            errors.append(f"Entry {position}: timestamp bị lùi hoặc overlap")
        if duration_ms is not None and end_ms > duration_ms + 1000:
            errors.append(f"Entry {position}: vượt quá duration video")
        if not text.strip():
            errors.append(f"Entry {position}: text rỗng")
        previous_end = max(previous_end, end_ms)
    return not errors, errors


def parse_srt_timing(srt_content: str) -> list[tuple[int, int, str]]:
    """Parse SRT into (start_ms, end_ms, text) tuples for TTS timeline work."""
    segments = []
    for _index, timestamp, text in parse_srt(srt_content):
        parts = timestamp.split("-->", 1)
        if len(parts) != 2:
            continue
        start_ms = _timestamp_to_ms(parts[0])
        end_ms = _timestamp_to_ms(parts[1].split()[0])
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            continue
        segments.append((start_ms, end_ms, " ".join(text.splitlines()).strip()))
    return [(start, end, text) for start, end, text in segments if text]
