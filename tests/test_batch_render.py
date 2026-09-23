from pathlib import Path

import cv2

import config
from services.burn_sub import _build_opaque_band_filter, burn_sub_video


def test_opaque_band_filter_covers_entire_region_before_ass():
    result = _build_opaque_band_filter(
        1920,
        1080,
        {"x_ratio": 0.05, "y_ratio": 0.80, "w_ratio": 0.90, "h_ratio": 0.12},
        "subs.ass",
    )

    assert "drawbox=x=96:y=864:w=1728:h=128" in result
    assert "color=black@1.0:t=fill" in result
    assert "ass='subs.ass'" in result


def test_opaque_band_render_creates_video_and_keeps_audio(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(10):
        frame = __import__("numpy").full((240, 320, 3), 90, dtype=__import__("numpy").uint8)
        cv2.putText(frame, "ZH", (45, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()

    job_id = "batch_render_test"
    config.create_job(job_id, original_name=source.name, video_path=str(source), video_file={"path": str(source)})
    result = burn_sub_video(
        job_id=job_id,
        lang="en",
        srt_content="1\n00:00:00,000 --> 00:00:01,000\nEnglish\n",
        sub_region={"x_ratio": 0.05, "y_ratio": 0.78, "w_ratio": 0.90, "h_ratio": 0.17},
        render_mode="opaque_band",
        keep_original_audio=True,
        video_path=str(source),
    )

    output = Path(result["path"])
    assert result["status"] == "done"
    assert output.exists() and output.stat().st_size > 0
    probe = __import__("subprocess").run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert "video" in probe.stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_real_batch_item_renders_output_video_to_manifest_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(10):
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        cv2.putText(frame, "ZH", (45, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    (source / "episode.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    manifest_path, _ = create_batch_manifest(
        "batch_real_video_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_real_video_123",
        manifest_path,
        {
            "output_mode": "srt_and_video",
            "translation_method": "google",
            "subtitle_engine": "hybrid",
            "old_subtitle_removal": "opaque_band",
            "sub_region": {"x_ratio": 0.05, "y_ratio": 0.78, "w_ratio": 0.90, "h_ratio": 0.17},
        },
    )
    translated = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    with patch("services.translation.translate_srt", return_value=translated):
        result = runner.run(processor=process_video_item)

    assert result["status"] == "done"
    output = Path(result["items"][0]["artifacts"]["video_path"])
    assert output.exists() and output.stat().st_size > 0
import subprocess
from pathlib import Path
from unittest.mock import patch

from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_real_batch_render_preserves_original_audio_stream(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=10:d=1",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video),
        ], check=True,
    )
    (source / "episode.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    manifest_path, _ = create_batch_manifest(
        "batch_audio_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_audio_123",
        manifest_path,
        {
            "output_mode": "srt_and_video",
            "translation_method": "google",
            "old_subtitle_removal": "opaque_band",
            "sub_region": {"x_ratio": 0.05, "y_ratio": 0.78, "w_ratio": 0.90, "h_ratio": 0.17},
        },
    )
    with patch("services.translation.translate_srt", return_value="1\n00:00:00,000 --> 00:00:01,000\nHello\n"):
        result = runner.run(processor=process_video_item)

    output = Path(result["items"][0]["artifacts"]["video_path"])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert "video" in probe.stdout
    assert "audio" in probe.stdout
from pathlib import Path
from unittest.mock import patch

from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_manual_region_is_used_for_removal_and_english_burn(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"video")
    (source / "episode.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    manifest_path, _ = create_batch_manifest(
        "batch_manual_region_123", source, [video], tmp_path / "output", {}
    )
    region = {"x_ratio": 0.1, "y_ratio": 0.72, "w_ratio": 0.8, "h_ratio": 0.16}
    runner = BatchRunner(
        "batch_manual_region_123",
        manifest_path,
        {
            "output_mode": "srt_and_video",
            "translation_method": "google",
            "region_mode": "manual",
            "sub_region": region,
        },
    )
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"rendered")

    with patch("services.translation.translate_srt", return_value="1\n00:00:00,000 --> 00:00:01,000\nHello\n"), patch(
        "services.burn_sub.burn_sub_video",
        return_value={"path": str(rendered), "sub_region": region, "method": "manual"},
    ) as burn:
        result = runner.run(processor=process_video_item)

    assert result["status"] == "done"
    assert burn.call_args.kwargs["sub_region"] == region
    artifacts = result["items"][0]["artifacts"]
    assert artifacts["sub_region"] == region
    assert artifacts["region_mode"] == "manual"
    assert artifacts["region_method"] == "manual"
from pathlib import Path
from unittest.mock import patch

from services.batch_pipeline import BatchRunner, create_batch_manifest, process_video_item


def test_auto_region_lets_burn_detect_and_records_actual_region(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"video")
    (source / "episode.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    manifest_path, _ = create_batch_manifest(
        "batch_auto_region_123", source, [video], tmp_path / "output", {}
    )
    runner = BatchRunner(
        "batch_auto_region_123",
        manifest_path,
        {"output_mode": "srt_and_video", "translation_method": "google", "region_mode": "auto"},
    )
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"rendered")
    detected = {"x_ratio": 0.08, "y_ratio": 0.8, "w_ratio": 0.84, "h_ratio": 0.12}

    with patch("services.translation.translate_srt", return_value="1\n00:00:00,000 --> 00:00:01,000\nHello\n"), patch(
        "services.burn_sub.burn_sub_video",
        return_value={"path": str(rendered), "sub_region": detected, "region_method": "ocr_detected"},
    ) as burn:
        result = runner.run(processor=process_video_item)

    assert burn.call_args.kwargs["sub_region"] is None
    artifacts = result["items"][0]["artifacts"]
    assert artifacts["sub_region"] == detected
    assert artifacts["region_mode"] == "auto"
    assert artifacts["region_method"] == "ocr_detected"
