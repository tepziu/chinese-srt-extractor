"""Atomic export contract consumed by Douyin_TikTok_Spider content factory."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

_ALLOWED_RIGHTS = {"owned", "licensed", "approved"}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _write_ready(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def emit_publish_package(
    *, batch_id: str, item: dict[str, Any], result: dict[str, Any],
    options: dict[str, Any], outbox_dir: str | Path,
    work: dict[str, Any] | None = None,
    channel_config: dict[str, Any] | None = None,
) -> Path:
    work = work or {}
    channel_config = channel_config or {}
    rights = str(
        channel_config.get("rights_status")
        or options.get("publish_rights_status")
        or os.getenv("DOUYIN_TIKTOK_RIGHTS_STATUS")
        or "review_required"
    ).strip().lower()
    rights_approved = rights in _ALLOWED_RIGHTS
    video_path = Path(str(result.get("video_path") or "")).expanduser().resolve()
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise ValueError("A non-empty rendered video is required for publish outbox")
    source_path = Path(str(item.get("source_path") or "")).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError("Source video is missing")
    source_digest = str(item.get("source_sha256") or "").lower() or _sha256(source_path)
    fallback_used = bool(result.get("translation_fallbacks"))
    provider_warning = result.get("translation_provider_warning")
    if provider_warning:
        fallback_used = True
    strict = bool(options.get("strict_translation", True))
    translation_approved = bool(strict and not fallback_used)
    video_digest = _sha256(video_path)
    package_id = "dysrt_" + hashlib.sha256(
        f"{batch_id}|{item.get('item_id')}|{source_digest}|{video_digest}".encode("utf-8")
    ).hexdigest()[:24]
    payload = {
        "schema_version": 1,
        "package_id": package_id,
        "producer": "chinese-srt-extractor",
        "producer_job_id": batch_id,
        "source_type": str(options.get("source_type") or channel_config.get("source_type") or "douyin_localized"),
        "source_platform": str(options.get("source_platform") or channel_config.get("source_platform") or "douyin"),
        "source_channel_id": channel_config.get("channel_id") or options.get("source_channel_id"),
        "source_channel_nickname": channel_config.get("nickname") or options.get("source_channel_nickname"),
        "source_url": item.get("source_url") or options.get("source_url") or work.get("share_url") or work.get("url"),
        "source_work_id": item.get("source_work_id") or work.get("aweme_id") or source_digest[:24],
        "source_sha256": source_digest,
        "rights_status": rights,
        "final_video": {
            "path": str(video_path),
            "sha256": video_digest,
            "bytes": video_path.stat().st_size,
        },
        "language": {
            "source": str(options.get("source_lang") or "zh"),
            "target": str(options.get("target_lang") or "en"),
        },
        "translation": {
            "strict": strict,
            "fallback_used": fallback_used,
            "review_status": "approved" if translation_approved else "awaiting_review",
            "method": result.get("translation_method"),
            "quality": result.get("translation_quality") or {},
        },
        "qc_status": "passed" if rights_approved and translation_approved else "review_required",
        "niche": str(channel_config.get("niche") or options.get("publish_niche") or "general"),
        "target_profile_id": channel_config.get("target_profile_id") or options.get("target_profile_id"),
        "publish_policy": channel_config.get("approval_policy") or options.get("approval_policy") or "manual",
        "caption": channel_config.get("caption") or options.get("caption") or str(work.get("desc") or ""),
        "audio_policy": channel_config.get("audio_policy") or options.get("audio_policy") or "keep_original",
        "relative_path": item.get("relative_path"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    destination = Path(outbox_dir).expanduser().resolve() / f"{package_id}.ready.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_payload = dict(payload)
        comparable_existing.pop("created_at", None)
        comparable_payload.pop("created_at", None)
        if comparable_existing != comparable_payload:
            raise ValueError("Publish package identity collision")
        return destination
    _write_ready(destination, payload)
    return destination
