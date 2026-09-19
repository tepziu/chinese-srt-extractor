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
