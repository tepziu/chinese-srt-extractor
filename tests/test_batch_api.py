from pathlib import Path
from unittest.mock import patch

from app import app
import config


def test_batch_api_creates_manifest_and_queues_worker(tmp_path, monkeypatch):
    source = tmp_path / "videos"
    source.mkdir()
    (source / "episode.mp4").write_bytes(b"video")
    monkeypatch.setattr("routes.api.BATCH_ALLOWED_ROOTS", [tmp_path])
    queued = []

    with patch("routes.api.submit", side_effect=lambda *args: queued.append(args)):
        response = app.test_client().post(
            "/api/batch/run",
            json={"folder_path": str(source), "output_mode": "srt_only"},
        )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["total_files"] == 1
    assert queued and queued[0][0].__name__ == "run_batch"
    manifest = Path(payload["manifest_path"])
    assert manifest.exists()
    assert config.get_job(payload["batch_id"])["mode"] == "batch"


def test_batch_api_rejects_empty_folder(tmp_path, monkeypatch):
    source = tmp_path / "empty"
    source.mkdir()
    monkeypatch.setattr("routes.api.BATCH_ALLOWED_ROOTS", [tmp_path])

    response = app.test_client().post("/api/batch/run", json={"folder_path": str(source)})

    assert response.status_code == 400
    assert "Không tìm thấy" in response.get_json()["error"]
import json
from pathlib import Path

import config
from app import app
from services.batch_pipeline import create_batch_manifest


def test_batch_status_reconciles_interrupted_parent_job(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"video")
    output = config.OUTPUT_FOLDER / "batches" / "batch_api_interrupt_123"
    manifest_path, manifest = create_batch_manifest(
        "batch_api_interrupt_123", source, [video], output, {}
    )
    manifest["status"] = "processing"
    manifest["items"][0]["status"] = "processing"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config.create_job(
        "batch_api_interrupt_123",
        mode="batch",
        batch_manifest_path=str(manifest_path),
        status="interrupted",
    )

    response = app.test_client().get("/api/batch/batch_api_interrupt_123")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "interrupted"
    assert payload["items"][0]["status"] == "queued"
from unittest.mock import patch

import cv2
import numpy as np

from app import app


def test_batch_region_preview_auto_detects_and_returns_overlay(tmp_path, monkeypatch):
    source = tmp_path / "videos"
    source.mkdir()
    video = source / "episode.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(10):
        writer.write(np.full((240, 320, 3), 80, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr("routes.api.BATCH_ALLOWED_ROOTS", [tmp_path])
    region = {"x_ratio": 0.1, "y_ratio": 0.72, "w_ratio": 0.8, "h_ratio": 0.16, "method": "ocr_detected"}

    with patch("services.burn_sub.detect_hardsub_region", return_value=region):
        response = app.test_client().post(
            "/api/batch/region-preview",
            json={"folder_path": str(source), "region_mode": "auto"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["region"]["y_ratio"] == 0.72
    assert payload["region_method"] == "ocr_detected"
    assert payload["video_name"] == "episode.mp4"
    image = app.test_client().get(payload["preview_url"])
    assert image.status_code == 200
    assert image.mimetype == "image/jpeg"
