"""Batch folder translation and subtitle replacement pipeline.

The batch layer deliberately reuses the existing single-video extraction,
translation and burn services while adding durable manifest/checkpoint state.
It is sequential by default because OCR, FFmpeg and GPU work are serialized by
``services.runtime_state`` already.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from config import OUTPUT_FOLDER, UPLOAD_FOLDER, create_job, get_job, jobs
from services.srt_utils import validate_srt

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
BATCH_TERMINAL = {"done", "partial", "failed", "cancelled", "error"}
PROCESSOR = Callable[[dict[str, Any], dict[str, Any], "BatchRunner"], dict[str, Any]]


class BatchCancelled(RuntimeError):
    """Raised internally when a batch cancellation is observed."""


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalise_extensions(values: Any) -> list[str]:
    if values is None:
        return sorted(VIDEO_EXTENSIONS)
    if isinstance(values, str):
        values = re.split(r"[,;\s]+", values)
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("extensions phải là danh sách")
    result = []
    for value in values:
        extension = str(value).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = "." + extension
        if extension not in VIDEO_EXTENSIONS:
            raise ValueError(f"Định dạng video không được hỗ trợ: {extension}")
        result.append(extension)
    if not result:
        raise ValueError("Phải chọn ít nhất một định dạng video")
    return sorted(set(result))


def _resolve_allowed_roots(allowed_roots: Iterable[str | Path] | None) -> list[Path]:
    return [Path(root).expanduser().resolve() for root in (allowed_roots or [])]


def _is_under(path: Path, roots: list[Path]) -> bool:
    if not roots:
        return True
    return any(path == root or root in path.parents for root in roots)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return value or "video"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_batch_options(data: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize public batch options."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Cấu hình batch phải là JSON object")

    result = dict(data)
    result["recursive"] = _as_bool(result.get("recursive"), True)
    result["skip_existing"] = _as_bool(result.get("skip_existing"), True)
    result["strict_translation"] = _as_bool(result.get("strict_translation"), True)
    result["allow_partial_render"] = _as_bool(result.get("allow_partial_render"), False)
    result["use_sidecar_srt"] = _as_bool(result.get("use_sidecar_srt"), True)
    result["keep_original_audio"] = _as_bool(result.get("keep_original_audio"), True)
    result["extensions"] = _normalise_extensions(result.get("extensions"))
    result["publish_outbox_dir"] = str(
        result.get("publish_outbox_dir") or os.getenv("DOUYIN_TIKTOK_OUTBOX_DIR") or ""
    ).strip() or None
    result["publish_rights_status"] = str(
        result.get("publish_rights_status") or os.getenv("DOUYIN_TIKTOK_RIGHTS_STATUS") or "review_required"
    ).strip().lower()
    result["publish_niche"] = str(result.get("publish_niche") or "general").strip().lower()

    result["target_lang"] = str(result.get("target_lang") or "en").strip().lower()
    if result["target_lang"] != "en":
        raise ValueError("Batch hiện chỉ hỗ trợ dịch sang English (en)")

    result["output_mode"] = str(result.get("output_mode") or "srt_and_video").strip().lower()
    if result["output_mode"] not in {"srt_only", "srt_and_video"}:
        raise ValueError("output_mode phải là srt_only hoặc srt_and_video")

    result["subtitle_engine"] = str(result.get("subtitle_engine") or "hybrid").strip().lower()
    if result["subtitle_engine"] not in {"hybrid", "gemini", "whisper"}:
        raise ValueError("subtitle_engine không hợp lệ")

    result["translation_method"] = str(result.get("translation_method") or "ai").strip().lower()
    if result["translation_method"] not in {"ai", "google"}:
        raise ValueError("translation_method phải là ai hoặc google")

    result["translation_mode"] = str(result.get("translation_mode") or "movie").strip().lower()
    if result["translation_mode"] not in {"movie", "literal", "driving", "fun"}:
        raise ValueError("translation_mode không hợp lệ")

    result["old_subtitle_removal"] = str(
        result.get("old_subtitle_removal") or "inpaint_burn"
    ).strip().lower()
    if result["old_subtitle_removal"] not in {"opaque_band", "blur", "inpaint_burn"}:
        raise ValueError("old_subtitle_removal không hợp lệ")

    result["inpaint_engine"] = str(result.get("inpaint_engine") or "opencv").strip().lower()
    if result["inpaint_engine"] not in {"opencv", "lama"}:
        raise ValueError("inpaint_engine không hợp lệ")

    try:
        result["max_concurrency"] = int(result.get("max_concurrency", 1))
    except (TypeError, ValueError):
        raise ValueError("max_concurrency không hợp lệ") from None
    if result["max_concurrency"] != 1:
        raise ValueError("Batch hiện chỉ cho phép max_concurrency=1 để bảo vệ GPU/FFmpeg")

    raw_region_mode = result.get("region_mode")
    if raw_region_mode is None:
        raw_region_mode = "manual" if result.get("sub_region") is not None else "auto"
    result["region_mode"] = str(raw_region_mode).strip().lower()
    if result["region_mode"] not in {"auto", "manual"}:
        raise ValueError("region_mode phải là auto hoặc manual")

    if result["region_mode"] == "manual" and result.get("sub_region") is None:
        raise ValueError("sub_region là bắt buộc khi region_mode=manual")

    if result.get("sub_region") is not None:
        region = result["sub_region"]
        if not isinstance(region, dict):
            raise ValueError("sub_region phải là object")
        normalized_region = {}
        for key in ("x_ratio", "y_ratio", "w_ratio", "h_ratio"):
            try:
                number = float(region.get(key, 0))
            except (TypeError, ValueError):
                raise ValueError("sub_region không hợp lệ") from None
            if key in {"w_ratio", "h_ratio"} and number <= 0:
                raise ValueError("sub_region không hợp lệ")
            if not 0 <= number <= 1:
                raise ValueError("sub_region không hợp lệ")
            normalized_region[key] = number
        if normalized_region["x_ratio"] + normalized_region["w_ratio"] > 1.000001:
            raise ValueError("sub_region vượt quá chiều rộng video")
        if normalized_region["y_ratio"] + normalized_region["h_ratio"] > 1.000001:
            raise ValueError("sub_region vượt quá chiều cao video")
        result["sub_region"] = normalized_region

    if result["region_mode"] == "auto":
        # Never let stale manual coordinates silently override per-video OCR detection.
        result["sub_region"] = None

    return result


def scan_video_folder(
    folder_path: str | Path,
    *,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
    allowed_roots: Iterable[str | Path] | None = None,
) -> list[Path]:
    """Return stable, sorted video paths from a local folder."""
    folder = Path(folder_path).expanduser().resolve()
    roots = _resolve_allowed_roots(allowed_roots)
    if not folder.is_dir():
        raise ValueError("Thư mục đầu vào không tồn tại hoặc không phải thư mục")
    if not folder.is_absolute():
        raise ValueError("Đường dẫn thư mục phải là đường dẫn tuyệt đối")
    if not _is_under(folder, roots):
        raise ValueError("Thư mục nằm ngoài allowlist")

    application_output = Path(OUTPUT_FOLDER).expanduser().resolve()
    if folder == application_output or application_output in folder.parents:
        raise ValueError("Không được dùng thư mục outputs của ứng dụng làm nguồn batch")

    allowed = set(_normalise_extensions(extensions))
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    files = []
    for candidate in iterator:
        try:
            if candidate.is_file() and candidate.suffix.lower() in allowed:
                resolved = candidate.resolve()
                if application_output == resolved or application_output in resolved.parents:
                    continue
                if _is_under(resolved, [folder]):
                    files.append(resolved)
        except OSError:
            continue
    return sorted(set(files), key=lambda item: item.relative_to(folder).as_posix().lower())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _manifest_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"total": len(items), "queued": 0, "processing": 0, "done": 0, "failed": 0, "partial": 0, "cancelled": 0}
    for item in items:
        status = item.get("status", "queued")
        if status not in counts:
            status = "failed"
        counts[status] += 1
    return counts


def create_batch_manifest(
    batch_id: str,
    source_folder: str | Path,
    files: Iterable[str | Path],
    output_dir: str | Path,
    options: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Create a durable manifest without starting any worker."""
    source = Path(source_folder).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    for index, raw_path in enumerate(files, start=1):
        path = Path(raw_path).expanduser().resolve()
        if source not in path.parents and path != source:
            raise ValueError("File batch nằm ngoài thư mục nguồn")
        relative = path.relative_to(source).as_posix()
        try:
            size = path.stat().st_size
            modified = path.stat().st_mtime
            digest = _sha256(path)
        except OSError as exc:
            raise ValueError(f"Không thể đọc file batch: {path}") from exc
        items.append({
            "item_id": f"{batch_id}_{index:04d}",
            "source_path": str(path),
            "relative_path": relative,
            "source_size": size,
            "source_mtime": modified,
            "source_sha256": digest,
            "output_stem": _safe_name(path.stem),
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "attempts": 0,
            "artifacts": {},
            "error": None,
        })

    # Two files such as episode.mp4 and episode.mkv must never overwrite the
    # same episode.en.srt/episode.en.mp4 artifact.
    stem_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        key = (str(Path(item["relative_path"]).parent).lower(), item["output_stem"].lower())
        stem_groups.setdefault(key, []).append(item)
    for group in stem_groups.values():
        if len(group) <= 1:
            continue
        for item in group:
            source_path = Path(item["source_path"])
            item["output_stem"] = f"{item['output_stem']}_{source_path.suffix.lstrip('.').lower()}_{item['source_sha256'][:8]}"

    manifest = {
        "version": 1,
        "batch_id": batch_id,
        "source_folder": str(source),
        "output_dir": str(destination),
        "created_at": time.time(),
        "updated_at": time.time(),
        "status": "queued",
        "cancel_requested": False,
        "options": dict(options),
        "items": items,
        "counts": _manifest_counts(items),
    }
    manifest_path = destination / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path, manifest


def load_batch_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("Không tìm thấy manifest batch")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def recover_stale_batch(manifest_path: str | Path, parent_job: dict[str, Any] | None) -> dict[str, Any]:
    """Turn an orphaned processing manifest into a resumable state.

    JobRegistry marks jobs whose owner process died as ``interrupted``. The
    manifest is separate durable state, so the API must reconcile the two
    before telling the UI that a batch is still actively running forever.
    """
    manifest = load_batch_manifest(manifest_path)
    if manifest.get("status") != "processing" or not parent_job or parent_job.get("status") != "interrupted":
        return manifest
    manifest["status"] = "interrupted"
    manifest["active_step"] = "interrupted"
    manifest["error"] = "Worker process stopped before the batch completed"
    for item in manifest.get("items", []):
        if item.get("status") == "processing":
            item.update(status="queued", stage="queued", progress=0, error="Worker interrupted")
    _write_json_atomic(Path(manifest_path).expanduser().resolve(), manifest)
    return manifest


def _set_parent_job(batch_id: str, manifest: dict[str, Any]) -> None:
    job = get_job(batch_id)
    if not job:
        return
    counts = manifest["counts"]
    job.update({
        "status": manifest["status"],
        "progress": int(manifest.get("overall_progress", 0)),
        "message": f"Batch: {counts['done']}/{counts['total']} hoàn tất; lỗi {counts['failed']}",
        "active_step": manifest.get("active_step", "batch"),
        "batch_counts": counts,
        "batch_manifest_path": manifest.get("manifest_path"),
    })


class BatchRunner:
    """Sequential, resumable batch runner backed by a JSON manifest."""

    def __init__(self, batch_id: str, manifest_path: str | Path, options: dict[str, Any] | None = None):
        self.batch_id = batch_id
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = load_batch_manifest(self.manifest_path)
        self.options = validate_batch_options(options or self.manifest.get("options") or {})
        self.manifest["options"] = dict(self.options)
        self.manifest["manifest_path"] = str(self.manifest_path)

    def _save(self) -> None:
        self.manifest["updated_at"] = time.time()
        self.manifest["counts"] = _manifest_counts(self.manifest.get("items", []))
        total = self.manifest["counts"]["total"]
        completed = self.manifest["counts"]["done"]
        current = next((item for item in self.manifest["items"] if item.get("status") == "processing"), None)
        self.manifest["overall_progress"] = int(((completed + (current or {}).get("progress", 0) / 100) / total) * 100) if total else 100
        _write_json_atomic(self.manifest_path, self.manifest)
        _set_parent_job(self.batch_id, self.manifest)

    def _cancelled(self) -> bool:
        try:
            latest = load_batch_manifest(self.manifest_path)
            if latest.get("cancel_requested"):
                self.manifest = latest
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return bool(get_job(self.batch_id) and get_job(self.batch_id).get("cancel"))

    def request_cancel(self) -> dict[str, Any]:
        self.manifest = load_batch_manifest(self.manifest_path)
        self.manifest["cancel_requested"] = True
        if get_job(self.batch_id):
            get_job(self.batch_id)["cancel"] = True
        for item in self.manifest.get("items", []):
            if item.get("status") == "processing":
                child = get_job(item.get("item_id"))
                if child:
                    child["cancel"] = True
        self._save()
        return self.manifest

    def run(self, processor: PROCESSOR | None = None) -> dict[str, Any]:
        processor = processor or process_video_item
        self.manifest["status"] = "processing"
        self.manifest["active_step"] = "batch"
        self._save()
        try:
            for item in self.manifest.get("items", []):
                if item.get("status") == "done" and self.options.get("skip_existing", True):
                    continue
                if self._cancelled():
                    self.manifest["status"] = "cancelled"
                    break
                item["status"] = "processing"
                item["stage"] = "starting"
                item["progress"] = 1
                item["attempts"] = int(item.get("attempts", 0)) + 1
                item["error"] = None
                self._save()
                try:
                    result = processor(item, self.options, self)
                    item["artifacts"] = dict(result or {})
                    if self.options.get("publish_outbox_dir") and (result or {}).get("video_path"):
                        from services.publish_outbox import emit_publish_package
                        package_path = emit_publish_package(
                            batch_id=self.batch_id,
                            item=item,
                            result=result or {},
                            options=self.options,
                            outbox_dir=self.options["publish_outbox_dir"],
                        )
                        item["artifacts"]["publish_package_path"] = str(package_path)
                    item["status"] = "done"
                    item["stage"] = "completed"
                    item["progress"] = 100
                except BatchCancelled:
                    item["status"] = "cancelled"
                    item["stage"] = "cancelled"
                    self.manifest["status"] = "cancelled"
                    self._save()
                    break
                except Exception as exc:
                    item["status"] = "failed"
                    item["stage"] = "failed"
                    item["progress"] = 0
                    item["error"] = str(exc)[:2000]
                self._save()
        except Exception as exc:
            self.manifest["status"] = "error"
            self.manifest["error"] = str(exc)[:2000]
        else:
            counts = self.manifest["counts"]
            if self.manifest.get("status") != "cancelled":
                if counts["failed"] and counts["done"]:
                    self.manifest["status"] = "partial"
                elif counts["failed"]:
                    self.manifest["status"] = "failed"
                else:
                    self.manifest["status"] = "done"
        self.manifest["active_step"] = "completed" if self.manifest["status"] == "done" else self.manifest.get("status")
        self._save()
        return self.manifest


def run_batch(batch_id: str, manifest_path: str | Path, options: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point used by the Flask API."""
    return BatchRunner(batch_id, manifest_path, options).run()


def _find_sidecar(source: Path) -> Path | None:
    for candidate in (source.with_suffix(".zh.srt"), source.with_suffix(".srt")):
        if candidate.is_file():
            return candidate
    return None


def _write_srt(path: Path, content: str) -> None:
    valid, errors = validate_srt(content)
    if not valid:
        raise RuntimeError("SRT không hợp lệ: " + "; ".join(errors[:3]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8-sig")


def _assert_strict_translation_quality(child: dict[str, Any], options: dict[str, Any]) -> None:
    """Prevent unattended rendering of an AI/fallback translation.

    The existing translation services intentionally preserve partial output for
    review. Batch strict mode must turn that quality signal into an item failure
    instead of silently producing a video containing untranslated lines.
    """
    if not options.get("strict_translation") or options.get("allow_partial_render"):
        return
    quality = (child.get("translation_quality") or {}).get("en") or {}
    fallbacks = (child.get("translation_fallbacks") or {}).get("en") or []
    warnings = (child.get("translation_provider_warnings") or {}).get("en")
    unchanged = quality.get("unchanged_segments") or []
    needs_review = bool(quality.get("needs_review"))
    provider_problem = options.get("translation_method") == "ai" and (fallbacks or warnings)
    if provider_problem or unchanged or needs_review:
        details = []
        if fallbacks:
            details.append(f"fallback {len(fallbacks)} cue")
        if unchanged:
            details.append(f"chưa dịch {len(unchanged)} cue")
        if needs_review:
            details.append("semantic quality cần review")
        if warnings:
            details.append(str(warnings)[:160])
        raise RuntimeError("Bản dịch không đạt strict mode: " + "; ".join(details))


def _check_batch_cancel(runner: BatchRunner, child_id: str) -> None:
    child = get_job(child_id) if child_id else None
    if runner._cancelled() or (child and child.get("cancel")):
        raise BatchCancelled("Batch đã được yêu cầu dừng")


def process_video_item(item: dict[str, Any], options: dict[str, Any], runner: BatchRunner) -> dict[str, Any]:
    """Process one source video using the existing Studio services."""
    source = Path(item["source_path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {source}")
    source_stat = source.stat()
    expected_size = item.get("source_size")
    expected_mtime = item.get("source_mtime")
    if (
        expected_size is not None
        and int(source_stat.st_size) != int(expected_size)
    ) or (
        expected_mtime is not None
        and abs(float(source_stat.st_mtime) - float(expected_mtime)) > 1e-6
    ):
        raise RuntimeError("File nguồn đã thay đổi sau khi quét batch; tạo batch mới để xử lý lại")

    child_id = item["item_id"]
    create_job(
        child_id,
        mode="batch_item",
        batch_id=runner.batch_id,
        original_name=source.name,
        video_path=str(source),
        video_file={"path": str(source), "filename": source.name, "size": source.stat().st_size},
        status="processing",
        progress=1,
    )
    child = get_job(child_id)
    output_root = Path(runner.manifest["output_dir"])
    relative = Path(item["relative_path"])
    item_root = output_root / relative.parent
    srt_root = item_root / "subtitles"
    video_root = item_root / "videos"
    stem = item.get("output_stem") or _safe_name(source.stem)
    zh_path = srt_root / f"{stem}.zh.srt"
    en_path = srt_root / f"{stem}.en.srt"

    child["active_step"] = "extract_sub"
    item["stage"] = "extract_sub"
    item["progress"] = 10
    runner._save()
    _check_batch_cancel(runner, child_id)

    sidecar = _find_sidecar(source) if options.get("use_sidecar_srt", True) else None
    whisper_context = None
    if sidecar:
        zh_content = sidecar.read_text(encoding="utf-8-sig")
        child.setdefault("subtitle_source_tracks", {})["sidecar"] = {"path": str(sidecar)}
    else:
        from services.pipeline_orchestrator import _extract_gemini_source, _extract_whisper_source
        engine = options["subtitle_engine"]
        if engine in {"whisper", "hybrid"}:
            try:
                whisper_context, _ = _extract_whisper_source(child_id, str(source), {
                    "whisper_model": options.get("whisper_model", "large-v3-turbo"),
                    "style": options.get("translation_mode", "movie"),
                })
            except Exception as exc:
                child.setdefault("subtitle_source_warnings", {})["audio"] = str(exc)[:500]
                if engine == "whisper":
                    raise
                whisper_context = None
        if engine in {"gemini", "hybrid"}:
            try:
                zh_content = _extract_gemini_source(child_id, str(source), {
                    "model": options.get("gemini_model", "gemini-3.8-flash-high"),
                })
            except Exception:
                if engine == "hybrid" and whisper_context:
                    zh_content = whisper_context
                else:
                    raise
        else:
            zh_content = whisper_context or ""

    _write_srt(zh_path, zh_content)
    child.setdefault("srt_files", {})["zh"] = {"path": str(zh_path), "filename": zh_path.name}
    item["artifacts"]["zh_srt_path"] = str(zh_path)
    item["progress"] = 30
    runner._save()
    _check_batch_cancel(runner, child_id)

    child["active_step"] = "translate"
    item["stage"] = "translate"
    item["progress"] = 35
    runner._save()
    from config import AI_TRANSLATE_CONFIG
    if (
        options["strict_translation"]
        and not options.get("allow_partial_render")
        and options["translation_method"] == "ai"
        and not AI_TRANSLATE_CONFIG.get("api_key")
    ):
        raise RuntimeError("Chưa cấu hình AI translation gateway; batch strict không dùng fallback tự động")

    if options["translation_method"] == "google":
        from services.translation import translate_srt
        en_content = translate_srt(zh_content, "en", child_id)
    elif options["subtitle_engine"] == "hybrid":
        from services.hybrid_subtitles import semantic_fuse_translate
        fusion = semantic_fuse_translate(
            zh_content,
            "en",
            child_id,
            translation_mode=options["translation_mode"],
            whisper_context=whisper_context,
            model=options.get("ai_model"),
        )
        if (
            options["strict_translation"]
            and not options.get("allow_partial_render")
            and fusion.get("used_fallback")
        ):
            raise RuntimeError("Semantic fusion phải dùng fallback; batch strict không render")
        if fusion.get("quality") is not None:
            child.setdefault("translation_quality", {})["en"] = fusion.get("quality") or {}
        if fusion.get("warning"):
            child.setdefault("translation_provider_warnings", {})["en"] = fusion["warning"]
        en_content = fusion["display_srt"]
    else:
        from services.translation import translate_srt_ai
        en_content = translate_srt_ai(
            zh_content,
            "en",
            child_id,
            ai_model=options.get("ai_model"),
            translation_mode=options["translation_mode"],
        )

    _assert_strict_translation_quality(child, options)
    _write_srt(en_path, en_content)
    child.setdefault("srt_files", {})["en"] = {"path": str(en_path), "filename": en_path.name}
    item["artifacts"]["en_srt_path"] = str(en_path)
    item["progress"] = 60
    runner._save()
    _check_batch_cancel(runner, child_id)

    translation_quality = (child.get("translation_quality") or {}).get("en") or {}
    result = {
        "zh_srt_path": str(zh_path),
        "en_srt_path": str(en_path),
        "translation_method": options["translation_method"],
        "translation_quality": translation_quality,
        "translation_fallbacks": (child.get("translation_fallbacks") or {}).get("en", []),
        "translation_provider_warning": (child.get("translation_provider_warnings") or {}).get("en"),
        "subtitle_source_warnings": child.get("subtitle_source_warnings", {}),
    }
    if options["output_mode"] == "srt_only":
        child["status"] = "done"
        return result

    child["active_step"] = "render"
    item["stage"] = "render"
    item["progress"] = 65
    runner._save()
    from services.burn_sub import burn_sub_video
    render_mode = options["old_subtitle_removal"]
    selected_region = options.get("sub_region") if options.get("region_mode") == "manual" else None
    burn_result = burn_sub_video(
        job_id=child_id,
        lang="en",
        srt_content=en_content,
        clean_timing_srt=zh_content,
        sub_region=selected_region,
        extra_regions=None,
        render_mode=render_mode,
        inpaint_engine=options.get("inpaint_engine", "opencv"),
        trim_intro="off",
        translate_title=False,
        title_lang="en",
        brand_name="",
        bgm_mode="keep_original",
        bgm_volume=1.0,
        clean_hardsub=True,
        clean_logo=False,
        clean_title=False,
        burn_new_sub=True,
        keep_original_audio=True,
        video_path=str(source),
    )
    rendered = Path(burn_result.get("path", ""))
    if not rendered.is_file() or rendered.stat().st_size == 0:
        raise RuntimeError("Không tạo được video tiếng Anh")
    output_video = video_root / f"{stem}.en.mp4"
    output_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rendered, output_video)
    result["video_path"] = str(output_video)
    result["removal_method"] = render_mode
    result["region_mode"] = options.get("region_mode", "auto")
    result["sub_region"] = burn_result.get("sub_region") or selected_region
    result["region_method"] = burn_result.get("region_method") or burn_result.get("method") or (
        "manual" if selected_region else "auto"
    )
    child["status"] = "done"
    return result
