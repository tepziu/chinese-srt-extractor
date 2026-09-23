import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from services.batch_pipeline import (
    BatchRunner,
    create_batch_manifest,
    scan_video_folder,
    validate_batch_options,
)


def test_scan_video_folder_is_recursive_and_deterministic(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"b")
    (tmp_path / "a.MKV").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.mov").write_bytes(b"c")
    (tmp_path / "ignore.txt").write_text("x")
    (tmp_path / "partial.mp4.part").write_bytes(b"x")

    result = scan_video_folder(str(tmp_path), recursive=True, allowed_roots=[tmp_path])

    assert [item.relative_to(tmp_path).as_posix() for item in result] == [
        "a.MKV", "b.mp4", "nested/c.mov"
    ]


def test_scan_video_folder_rejects_path_outside_allowlist(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "video.mp4").write_bytes(b"x")

    with pytest.raises(ValueError, match="allowlist"):
        scan_video_folder(str(outside), allowed_roots=[allowed])


def test_validate_batch_options_defaults_to_english_and_strict_opaque_mode():
    options = validate_batch_options({})

    assert options["target_lang"] == "en"
    assert options["output_mode"] == "srt_and_video"
    assert options["old_subtitle_removal"] == "opaque_band"
    assert options["strict_translation"] is True
    assert options["max_concurrency"] == 1


def test_create_manifest_preserves_relative_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "episode.mp4").write_bytes(b"video")
    batch_dir = tmp_path / "output"

    manifest_path, manifest = create_batch_manifest(
        "batch_test_123",
        source,
        [source / "episode.mp4"],
        batch_dir,
        {"target_lang": "en"},
    )

    assert manifest_path.exists()
    assert manifest["batch_id"] == "batch_test_123"
    assert manifest["items"][0]["relative_path"] == "episode.mp4"
    assert manifest["items"][0]["status"] == "queued"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["items"][0]["source_path"] == str(source / "episode.mp4")


def test_batch_runner_resumes_and_aggregates_item_results(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = []
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        path = source / name
        path.write_bytes(name.encode())
        files.append(path)
    manifest_path, _ = create_batch_manifest(
        "batch_resume_123", source, files, tmp_path / "output", {"target_lang": "en"}
    )

    calls = []

    def processor(item, options, runner):
        calls.append(item["relative_path"])
        if item["relative_path"] == "b.mp4" and calls.count("b.mp4") == 1:
            raise RuntimeError("provider failed")
        return {"en_srt_path": f"{item['relative_path']}.en.srt"}

    first = BatchRunner("batch_resume_123", manifest_path, {"target_lang": "en"})
    first.run(processor=processor)
    assert first.manifest["status"] == "partial"
    assert first.manifest["counts"]["done"] == 2
    assert first.manifest["counts"]["failed"] == 1

    second = BatchRunner("batch_resume_123", manifest_path, {"target_lang": "en"})
    second.run(processor=processor)
    assert second.manifest["status"] == "done"
    assert calls == ["a.mp4", "b.mp4", "c.mp4", "b.mp4"]
    assert second.manifest["counts"]["failed"] == 0


def test_batch_runner_cancel_is_persisted(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "a.mp4"
    video.write_bytes(b"video")
    manifest_path, _ = create_batch_manifest(
        "batch_cancel_123", source, [video], tmp_path / "output", {"target_lang": "en"}
    )
    runner = BatchRunner("batch_cancel_123", manifest_path, {})
    runner.request_cancel()
    processor = Mock()
    runner.run(processor=processor)

    assert runner.manifest["status"] == "cancelled"
    processor.assert_not_called()
from pathlib import Path
from unittest.mock import patch

from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_real_batch_item_uses_sidecar_and_writes_english_srt(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"not decoded in srt-only mode")
    (source / "episode.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    manifest_path, _ = create_batch_manifest(
        "batch_real_item_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_real_item_123",
        manifest_path,
        {"output_mode": "srt_only", "translation_method": "google", "subtitle_engine": "hybrid"},
    )

    with patch("services.translation.translate_srt", return_value="1\n00:00:00,000 --> 00:00:01,000\nHello\n"):
        result = runner.run(processor=process_video_item)

    assert result["status"] == "done"
    item = result["items"][0]
    assert Path(item["artifacts"]["zh_srt_path"]).exists()
    assert Path(item["artifacts"]["en_srt_path"]).read_text(encoding="utf-8-sig").find("Hello") >= 0
from unittest.mock import patch

import config
from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_strict_batch_rejects_provider_fallback_from_non_hybrid_translation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"not decoded in srt-only mode")
    (source / "episode.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    manifest_path, _ = create_batch_manifest(
        "batch_strict_fallback_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_strict_fallback_123",
        manifest_path,
        {"output_mode": "srt_only", "translation_method": "ai", "subtitle_engine": "whisper"},
    )

    def fake_translate(*_args, **_kwargs):
        child = config.get_job("batch_strict_fallback_123_0001")
        child.setdefault("translation_provider_warnings", {})["en"] = "provider unavailable"
        child.setdefault("translation_fallbacks", {})["en"] = [1]
        return "1\n00:00:00,000 --> 00:00:01,000\nHello\n"

    with patch.dict(config.AI_TRANSLATE_CONFIG, {"api_key": "configured"}), patch(
        "services.translation.translate_srt_ai", side_effect=fake_translate
    ):
        result = runner.run(processor=process_video_item)

    assert result["status"] == "failed"
    assert "fallback" in result["items"][0]["error"].lower()
from services.batch_pipeline import scan_video_folder


def test_scan_video_folder_excludes_application_outputs(tmp_path, monkeypatch):
    source = tmp_path / "project"
    source.mkdir()
    output = source / "outputs"
    output.mkdir()
    (source / "input.mp4").write_bytes(b"input")
    (output / "old-result.mp4").write_bytes(b"old")
    monkeypatch.setattr("services.batch_pipeline.OUTPUT_FOLDER", output)

    result = scan_video_folder(str(source), allowed_roots=[source])

    assert [item.name for item in result] == ["input.mp4"]


def test_manifest_disambiguates_duplicate_stems(tmp_path):
    from services.batch_pipeline import create_batch_manifest

    source = tmp_path / "source"
    source.mkdir()
    first = source / "episode.mp4"
    second = source / "episode.mkv"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manifest_path, manifest = create_batch_manifest(
        "batch_collision_123", source, [first, second], tmp_path / "output", {}
    )

    stems = [item["output_stem"] for item in manifest["items"]]
    assert len(set(stems)) == 2
    assert all(stem.startswith("episode_") for stem in stems)
from unittest.mock import patch

from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_hybrid_batch_continues_with_gemini_when_whisper_fails(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"not decoded in srt-only mode")
    manifest_path, _ = create_batch_manifest(
        "batch_hybrid_fallback_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_hybrid_fallback_123",
        manifest_path,
        {"output_mode": "srt_only", "translation_method": "google", "subtitle_engine": "hybrid"},
    )
    source_srt = "1\n00:00:00,000 --> 00:00:01,000\n你好\n"

    with patch("services.pipeline_orchestrator._extract_whisper_source", side_effect=RuntimeError("whisper unavailable")), patch(
        "services.pipeline_orchestrator._extract_gemini_source", return_value=source_srt
    ), patch("services.translation.translate_srt", return_value="1\n00:00:00,000 --> 00:00:01,000\nHello\n"):
        result = runner.run(processor=process_video_item)

    assert result["status"] == "done"
from unittest.mock import patch

from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_strict_batch_rejects_unchanged_chinese_even_with_google_method(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"not decoded in srt-only mode")
    (source / "episode.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    manifest_path, _ = create_batch_manifest(
        "batch_strict_unchanged_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_strict_unchanged_123",
        manifest_path,
        {"output_mode": "srt_only", "translation_method": "google", "subtitle_engine": "hybrid"},
    )

    def fake_translate(*_args, **_kwargs):
        import config
        child = config.get_job("batch_strict_unchanged_123_0001")
        child.setdefault("translation_quality", {})["en"] = {"unchanged_segments": [1]}
        return "1\n00:00:00,000 --> 00:00:01,000\n你好\n"

    with patch("services.translation.translate_srt", side_effect=fake_translate):
        result = runner.run(processor=process_video_item)

    assert result["status"] == "failed"
    assert "chưa dịch" in result["items"][0]["error"]
from unittest.mock import patch

import config
from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_strict_hybrid_rejects_semantic_quality_needing_review(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"not decoded in srt-only mode")
    manifest_path, _ = create_batch_manifest(
        "batch_quality_review_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_quality_review_123",
        manifest_path,
        {"output_mode": "srt_only", "translation_method": "ai", "subtitle_engine": "hybrid"},
    )
    source_srt = "1\n00:00:00,000 --> 00:00:01,000\n你好\n"

    def fake_fusion(*_args, **_kwargs):
        return {
            "display_srt": "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
            "used_fallback": False,
            "quality": {"needs_review": True, "unchanged_segments": []},
        }

    with patch.dict(config.AI_TRANSLATE_CONFIG, {"api_key": "configured"}), patch(
        "services.pipeline_orchestrator._extract_whisper_source", return_value=(source_srt, "whisper.srt")
    ), patch("services.pipeline_orchestrator._extract_gemini_source", return_value=source_srt), patch(
        "services.hybrid_subtitles.semantic_fuse_translate", side_effect=fake_fusion
    ):
        result = runner.run(processor=process_video_item)

    assert result["status"] == "failed"
    assert "review" in result["items"][0]["error"].lower()
from services.batch_pipeline import create_batch_manifest, recover_stale_batch, load_batch_manifest


def test_recover_stale_manifest_makes_processing_items_resumable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"video")
    manifest_path, manifest = create_batch_manifest(
        "batch_interrupted_123", source, [video], tmp_path / "output", {}
    )
    manifest["status"] = "processing"
    manifest["items"][0].update(status="processing", stage="render", progress=55)
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    recovered = recover_stale_batch(manifest_path, {"status": "interrupted"})

    assert recovered["status"] == "interrupted"
    assert recovered["items"][0]["status"] == "queued"
    assert load_batch_manifest(manifest_path)["active_step"] == "interrupted"
from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_batch_refuses_source_file_changed_after_scan(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"before")
    manifest_path, _ = create_batch_manifest(
        "batch_source_changed_123", source, [video], tmp_path / "output", {}
    )
    video.write_bytes(b"after-with-different-size")
    runner = BatchRunner(
        "batch_source_changed_123",
        manifest_path,
        {"output_mode": "srt_only", "translation_method": "google"},
    )

    result = runner.run(processor=process_video_item)

    assert result["status"] == "failed"
    assert "thay đổi" in result["items"][0]["error"]
import pytest

from services.batch_pipeline import validate_batch_options


def test_manual_region_mode_requires_and_normalizes_region():
    with pytest.raises(ValueError, match="sub_region"):
        validate_batch_options({"region_mode": "manual"})

    options = validate_batch_options({
        "region_mode": "manual",
        "sub_region": {"x_ratio": 0.1, "y_ratio": 0.72, "w_ratio": 0.8, "h_ratio": 0.16},
    })
    assert options["region_mode"] == "manual"
    assert options["sub_region"] == {
        "x_ratio": 0.1, "y_ratio": 0.72, "w_ratio": 0.8, "h_ratio": 0.16,
    }


def test_auto_region_mode_does_not_reuse_stale_manual_region():
    options = validate_batch_options({
        "region_mode": "auto",
        "sub_region": {"x_ratio": 0.1, "y_ratio": 0.72, "w_ratio": 0.8, "h_ratio": 0.16},
    })
    assert options["region_mode"] == "auto"
    assert options["sub_region"] is None
from services.batch_pipeline import validate_batch_options

def test_legacy_sub_region_without_mode_is_treated_as_manual():
    options = validate_batch_options({
        "sub_region": {"x_ratio": 0.05, "y_ratio": 0.78, "w_ratio": 0.9, "h_ratio": 0.17}
    })
    assert options["region_mode"] == "manual"
    assert options["sub_region"]["y_ratio"] == 0.78
