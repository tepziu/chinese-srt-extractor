"""Gemini 3.1 Flash TTS client with shared, non-leaking credentials."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


BASE_DIR = Path(__file__).resolve().parent.parent
GOOGLE_MEDIA_CONFIG = BASE_DIR / "config" / "google_media.json"
GOOGLE_TTS_CONFIG = BASE_DIR / "config" / "google_tts.json"

SAFE_MARKUP_TAGS = {
    "sigh", "laughing", "laughs", "uhm", "sarcasm", "robotic",
    "shouting", "whispering", "extremely fast", "short pause",
    "medium pause", "long pause",
}
PERFORMANCE_CUES = {
    "mystery", "sensory", "tension", "suspense", "wonder", "urgent",
    "melancholic", "scared", "curious", "bored",
}
EMOTION_PROMPTS = {
    "neutral": "Use a natural, balanced adult narration with clear articulation.",
    "warm": "Use a warm, sincere, emotionally present adult narration.",
    "formal": "Use a polished, confident and professional adult narration.",
    "reassuring": "Use a calm, reassuring and trustworthy adult narration.",
    "energetic": "Use a bright, energetic and engaging adult narration without rushing.",
    "solemn": "Use a restrained, solemn and reflective adult narration with meaningful pauses.",
    "intimate": "Use an intimate, gentle and conversational adult narration.",
    "dramatic": "Use a cinematic, dramatic adult performance with controlled intensity.",
    "mystery": "Use a mysterious, restrained delivery that invites curiosity.",
    "sensory": "Use vivid, immersive sensory storytelling and natural pauses.",
    "tension": "Build controlled tension with focused emphasis and deliberate pauses.",
    "suspense": "Use suspenseful timing and carefully delayed reveals.",
    "wonder": "Use an open, quietly amazed delivery with warmth.",
    "urgent": "Use clear urgency while preserving intelligibility.",
    "melancholic": "Use a reflective, melancholic delivery with emotional restraint.",
    "scared": "Use a believable frightened delivery without becoming theatrical.",
    "curious": "Use an inquisitive delivery with light upward inflection where natural.",
    "bored": "Use a deliberately flat, disengaged delivery with reduced variation.",
}

LANGUAGE_CODES = {"vi": "vi-VN", "en": "en-US", "id": "id-ID", "zh": "zh-CN"}
DEFAULT_LANGUAGE_PROFILES = {
    "vi-VN": "Đọc bằng tiếng Việt tự nhiên, phát âm rõ dấu và phụ âm cuối, nhịp điệu điện ảnh nhưng gần gũi. Giữ giọng người trưởng thành và tránh ngữ điệu máy móc.",
    "en-US": "Speak in natural American English with clear articulation, human rhythm, and cinematic restraint. Avoid an announcer-like cadence.",
    "id-ID": "Speak in natural Indonesian with clear articulation, an adult voice, and a conversational cinematic rhythm.",
    "zh-CN": "Speak in natural Mandarin Chinese with clear tones, an adult voice, and a conversational cinematic rhythm.",
}


@dataclass(frozen=True)
class GoogleTTSSettings:
    endpoint: str
    model: str
    voice: str
    sample_rate: int
    timeout_seconds: float
    default_emotion: str
    style_prompt: str
    language_profiles: dict[str, str]
    keys: tuple[str, ...]
    credential_source: str

    @property
    def configured(self) -> bool:
        return bool(self.keys)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"File cấu hình JSON không hợp lệ: {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"File cấu hình phải là JSON object: {path.name}")
    return data


def normalize_api_keys(value: Any) -> tuple[str, ...]:
    values = value.split(",") if isinstance(value, str) else value if isinstance(value, (list, tuple)) else []
    result: list[str] = []
    for item in values:
        key = str(item or "").strip()
        if key and key not in result:
            result.append(key)
    return tuple(result[:16])


def load_google_media_credentials() -> tuple[tuple[str, ...], str]:
    data = _read_json(GOOGLE_MEDIA_CONFIG)
    candidates = (
        ("GOOGLE_MEDIA_API_KEYS", normalize_api_keys(os.environ.get("GOOGLE_MEDIA_API_KEYS", ""))),
        ("GOOGLE_MEDIA_API_KEY", normalize_api_keys(os.environ.get("GOOGLE_MEDIA_API_KEY", ""))),
        ("GEMINI_API_KEY", normalize_api_keys(os.environ.get("GEMINI_API_KEY", ""))),
        (str(GOOGLE_MEDIA_CONFIG), normalize_api_keys(data.get("api_keys")) or normalize_api_keys(data.get("api_key"))),
    )
    for source, keys in candidates:
        if keys:
            return keys, source
    return (), "unconfigured"


def save_shared_google_key(api_key: str) -> None:
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("API key không được để trống")
    GOOGLE_MEDIA_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    GOOGLE_MEDIA_CONFIG.write_text(
        json.dumps({"api_key": key}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_google_tts_settings() -> GoogleTTSSettings:
    data = _read_json(GOOGLE_TTS_CONFIG)
    keys, source = load_google_media_credentials()
    endpoint = str(os.environ.get("GOOGLE_TTS_ENDPOINT") or data.get("endpoint") or "https://generativelanguage.googleapis.com/v1beta/interactions").rstrip("/")
    if endpoint != "https://generativelanguage.googleapis.com/v1beta/interactions":
        raise ValueError("GOOGLE_TTS_ENDPOINT phải là endpoint Interactions HTTPS chính thức của Google")
    profiles = dict(DEFAULT_LANGUAGE_PROFILES)
    raw_profiles = data.get("language_profiles")
    if isinstance(raw_profiles, dict):
        for locale, profile in raw_profiles.items():
            if isinstance(profile, dict) and str(profile.get("style_prompt", "")).strip():
                profiles[str(locale)] = str(profile["style_prompt"]).strip()
    emotion = str(os.environ.get("GOOGLE_TTS_DEFAULT_EMOTION") or data.get("default_emotion") or "warm").lower()
    if emotion not in EMOTION_PROMPTS:
        raise ValueError(f"Cảm xúc Gemini TTS không hợp lệ: {emotion}")
    return GoogleTTSSettings(
        endpoint=endpoint,
        model=str(os.environ.get("GOOGLE_TTS_MODEL_NAME") or data.get("model_name") or "gemini-3.1-flash-tts-preview"),
        voice=str(os.environ.get("GOOGLE_TTS_VOICE_NAME") or data.get("voice_name") or "Charon"),
        sample_rate=max(8000, min(48000, int(data.get("sample_rate_hertz", 24000)))),
        timeout_seconds=max(10.0, min(180.0, float(data.get("timeout_seconds", 60)))),
        default_emotion=emotion,
        style_prompt=str(data.get("style_prompt") or "Deliver a polished, human, cinematic adult narration."),
        language_profiles=profiles,
        keys=keys,
        credential_source=source,
    )


def compile_performance(text: str) -> tuple[str, list[str]]:
    cues: list[str] = []
    aliases = {"laughs": "laughing", "laugh": "laughing", "whisper": "whispering", "shout": "shouting"}

    def replace(match: re.Match[str]) -> str:
        tag = match.group(1).strip().lower().replace("_", " ").replace("-", " ")
        tag = aliases.get(tag, tag)
        if tag in SAFE_MARKUP_TAGS:
            return f"[{tag}]"
        if tag in PERFORMANCE_CUES or tag in EMOTION_PROMPTS:
            if tag not in cues:
                cues.append(tag)
        return ""

    clean = re.sub(r"\[([^\[\]]{1,40})\]", replace, str(text or "").strip())
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, cues


def build_interaction_payload(
    text: str,
    *,
    lang: str,
    voice: str | None = None,
    emotion: str | None = None,
    style_prompt: str | None = None,
    continuation: bool = False,
    settings: GoogleTTSSettings | None = None,
) -> dict[str, Any]:
    settings = settings or load_google_tts_settings()
    clean_text, cues = compile_performance(text)
    if not clean_text:
        raise ValueError("Nội dung TTS trống sau khi loại markup không an toàn")
    locale = LANGUAGE_CODES.get(lang, lang if re.fullmatch(r"[a-z]{2,3}-[A-Z]{2,3}", lang) else "en-US")
    selected_voice = str(voice or settings.voice).strip()
    if not re.fullmatch(r"[A-Za-z0-9-]+", selected_voice):
        raise ValueError("Tên giọng Gemini TTS không hợp lệ")
    selected_emotion = str(emotion or settings.default_emotion).lower()
    if selected_emotion not in EMOTION_PROMPTS:
        raise ValueError(f"Cảm xúc Gemini TTS không hợp lệ: {selected_emotion}")
    prompt_parts = [settings.language_profiles.get(locale, "Speak naturally with clear articulation."), settings.style_prompt, EMOTION_PROMPTS[selected_emotion]]
    prompt_parts.extend(EMOTION_PROMPTS[cue] for cue in cues if cue in EMOTION_PROMPTS)
    if style_prompt:
        prompt_parts.append(str(style_prompt).strip())
    if continuation:
        prompt_parts.append("Continue seamlessly from the preceding narration; do not sound like a fresh introduction.")
    instruction = " ".join(part for part in prompt_parts if part)
    combined = (
        f"{instruction}\n\nSpeak the following text in {locale}. Recite the words exactly as written. "
        "Treat bracketed markup as silent performance directions and never speak tag names:\n"
        f"{clean_text}"
    )
    if len(combined.encode("utf-8")) > 8000:
        raise ValueError("Gemini TTS vượt giới hạn 8.000 byte cho một đoạn")
    return {
        "model": settings.model,
        "input": combined,
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": selected_voice}]},
    }


def _decode_audio_response(data: dict[str, Any], sample_rate: int) -> bytes:
    audio_b64 = None
    for step in data.get("steps") or []:
        if isinstance(step, dict) and step.get("type") == "model_output":
            for block in step.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "audio" and block.get("data"):
                    audio_b64 = block["data"]
    convenience = data.get("output_audio") or data.get("outputAudio")
    if not audio_b64 and isinstance(convenience, dict):
        audio_b64 = convenience.get("data")
    if not audio_b64:
        raise ValueError("Phản hồi Gemini TTS không có dữ liệu audio")
    pcm = base64.b64decode(audio_b64)
    if pcm.startswith(b"RIFF"):
        return pcm
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def _safe_error(value: Any, keys: tuple[str, ...]) -> str:
    message = str(value or "")
    for key in keys:
        message = message.replace(key, "[REDACTED]")
    message = re.sub(r'(?i)(api[_-]?key|key|token|authorization)(\s*[=:]\s*)[^\s,}\"]+', r'\1\2[REDACTED]', message)
    return message[:800]


def synthesize_to_wav(
    text: str,
    output_path: str | Path,
    *,
    lang: str,
    voice: str | None = None,
    emotion: str | None = None,
    style_prompt: str | None = None,
    continuation: bool = False,
    cancelled=None,
) -> str:
    settings = load_google_tts_settings()
    if not settings.configured:
        raise RuntimeError("Gemini TTS chưa được cấu hình. Hãy đặt GEMINI_API_KEY hoặc GOOGLE_MEDIA_API_KEY.")
    payload = build_interaction_payload(
        text, lang=lang, voice=voice, emotion=emotion,
        style_prompt=style_prompt, continuation=continuation, settings=settings,
    )
    last_error = "unknown error"
    with httpx.Client(timeout=settings.timeout_seconds) as client:
        for key in settings.keys:
            retries = 0
            while True:
                if cancelled and cancelled():
                    raise RuntimeError("Đã hủy TTS (Stop)")
                try:
                    response = client.post(
                        settings.endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json", "x-goog-api-key": key},
                    )
                except Exception as exc:
                    last_error = _safe_error(exc, settings.keys)
                    break
                if response.status_code == 200:
                    audio = _decode_audio_response(response.json(), settings.sample_rate)
                    destination = Path(output_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(audio)
                    return str(destination)
                try:
                    detail = response.json().get("error", {}).get("message") or response.text
                except Exception:
                    detail = response.text
                last_error = f"HTTP {response.status_code}: {_safe_error(detail, settings.keys)}"
                if response.status_code == 429 and retries < 2:
                    retries += 1
                    match = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", str(detail), re.I)
                    delay = min(60.0, max(1.0, float(match.group(1)) + 1.0 if match else 30.0))
                    deadline = time.time() + delay
                    while time.time() < deadline:
                        if cancelled and cancelled():
                            raise RuntimeError("Đã hủy TTS (Stop)")
                        time.sleep(min(1.0, deadline - time.time()))
                    continue
                if response.status_code == 400:
                    raise RuntimeError(last_error)
                break
    raise RuntimeError(f"Gemini TTS thất bại: {last_error}")


def get_google_tts_health() -> dict[str, Any]:
    settings = load_google_tts_settings()
    ffmpeg_available = shutil.which("ffmpeg") is not None
    ffprobe_available = shutil.which("ffprobe") is not None
    return {
        "ok": settings.configured and ffmpeg_available,
        "configured": settings.configured,
        "key_count": len(settings.keys),
        "credential_source": settings.credential_source,
        "model": settings.model,
        "voice": settings.voice,
        "sample_rate_hertz": settings.sample_rate,
        "supported_languages": LANGUAGE_CODES,
        "emotions": sorted(EMOTION_PROMPTS),
        "safe_markup_tags": sorted(SAFE_MARKUP_TAGS),
        "performance_cues": sorted(PERFORMANCE_CUES),
        "ffmpeg_available": ffmpeg_available,
        "ffprobe_available": ffprobe_available,
    }
