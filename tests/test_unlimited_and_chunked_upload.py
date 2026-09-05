import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import config
from app import app
from routes.api import (
    _chunk_uploads,
    _chunk_lock,
    _validate_media_path,
)
from services.downloader import _check_download_size
from services.hardsub_gemini import create_gemini_proxy_video


class TestUnlimitedAndChunkedUpload(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = app.test_client()
        with _chunk_lock:
            _chunk_uploads.clear()

    def test_config_limits_are_unlimited(self):
        """Verify default limits are 0 (unlimited)."""
        self.assertEqual(config.MAX_UPLOAD_BYTES, 0)
        self.assertEqual(config.MAX_DOWNLOAD_BYTES, 0)
        self.assertEqual(config.MAX_VIDEO_DURATION_SECONDS, 0)
        self.assertGreaterEqual(config.GEMINI_MAX_DIRECT_UPLOAD_BYTES, 1024 * 1024 * 1024)

    def test_flask_app_max_content_length_is_none(self):
        """Flask app should have MAX_CONTENT_LENGTH = None for unlimited file upload."""
        self.assertIsNone(self.app.config.get("MAX_CONTENT_LENGTH"))

    def test_device_info_reports_unlimited(self):
        """API device endpoint should report unlimited upload."""
        res = self.client.get("/api/device")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("unlimited_upload"))
        self.assertEqual(data.get("max_upload_bytes"), 0)

    @patch("subprocess.run")
    def test_validate_media_path_allows_large_file_and_long_duration(self, mock_probe):
        """Validate media path must not reject 5GB file or 5-hour video when limits are 0."""
        mock_probe.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "18000.0"}}),  # 5 hours
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"fake video data")
            temp_path = Path(tf.name)

        try:
            # Mock file size to simulate a 5GB file (5 * 1024^3 bytes)
            with patch.object(Path, "stat") as mock_stat:
                stat_obj = MagicMock()
                stat_obj.st_size = 5 * 1024 * 1024 * 1024
                mock_stat.return_value = stat_obj

                valid, msg = _validate_media_path(temp_path)
                self.assertTrue(valid, f"Validation failed with error: {msg}")
                self.assertEqual(msg, "")
        finally:
            temp_path.unlink(missing_ok=True)

    def test_downloader_allows_large_downloads(self):
        """Downloader size check should not raise error for 10GB file when MAX_DOWNLOAD_BYTES is 0."""
        # 10 GB
        large_size = 10 * 1024 * 1024 * 1024
        try:
            _check_download_size(large_size)
        except Exception as e:
            self.fail(f"_check_download_size raised unexpected exception: {e}")

    def test_chunked_upload_flow(self):
        """Test full Chunked Upload lifecycle: init -> chunk -> finish."""
        total_data = b"Hello, this is a simulated chunked video payload! " * 200
        total_size = len(total_data)
        chunk_size = 1000
        total_chunks = (total_size + chunk_size - 1) // chunk_size

        # 1. Init
        init_res = self.client.post("/api/upload/init", json={
            "filename": "heavy_test_video.mp4",
            "total_size": total_size,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "upload_type": "whisper",
            "model_size": "large-v3-turbo",
            "translate_langs": ["vi"],
        })
        self.assertEqual(init_res.status_code, 200)
        init_data = init_res.get_json()
        job_id = init_data["job_id"]
        self.assertTrue(job_id)

        # 2. Send chunks
        for idx in range(total_chunks):
            start = idx * chunk_size
            end = min(total_size, start + chunk_size)
            chunk_slice = total_data[start:end]

            chunk_res = self.client.post("/api/upload/chunk", data={
                "job_id": job_id,
                "chunk_index": idx,
                "chunk": (io.BytesIO(chunk_slice), "heavy_test_video.mp4"),
            }, content_type="multipart/form-data")
            self.assertEqual(chunk_res.status_code, 200)
            chunk_info = chunk_res.get_json()
            self.assertEqual(chunk_info["chunk_index"], idx)
            self.assertEqual(chunk_info["status"], "ok")

        # 3. Finish (Mock validation and worker thread to avoid real ffmpeg probe on dummy data)
        with patch("routes.api._validate_media_path", return_value=(True, "")):
            with patch("threading.Thread.start"):
                finish_res = self.client.post("/api/upload/finish", json={"job_id": job_id})
                self.assertEqual(finish_res.status_code, 200)
                finish_data = finish_res.get_json()
                self.assertEqual(finish_data["job_id"], job_id)
                self.assertEqual(finish_data["status"], "queued")

        # Verify on-disk file content matches original total_data
        job_dir = config.UPLOAD_FOLDER / job_id
        saved_file = job_dir / "source.mp4"
        self.assertTrue(saved_file.exists())
        self.assertEqual(saved_file.read_bytes(), total_data)

        # Clean up
        shutil.rmtree(job_dir, ignore_errors=True)

    def test_chunked_upload_detects_missing_chunks(self):
        """Finish should reject if chunks are missing."""
        init_res = self.client.post("/api/upload/init", json={
            "filename": "missing_test.mp4",
            "total_size": 3000,
            "chunk_size": 1000,
            "total_chunks": 3,
            "upload_type": "whisper",
        })
        job_id = init_res.get_json()["job_id"]

        # Only upload chunk 0 and 2 (skip chunk 1)
        for idx in (0, 2):
            self.client.post("/api/upload/chunk", data={
                "job_id": job_id,
                "chunk_index": idx,
                "chunk": (io.BytesIO(b"chunk data"), "missing_test.mp4"),
            }, content_type="multipart/form-data")

        finish_res = self.client.post("/api/upload/finish", json={"job_id": job_id})
        self.assertEqual(finish_res.status_code, 400)
        self.assertIn("Chưa nhận đủ dữ liệu", finish_res.get_json()["error"])

        # Clean up
        shutil.rmtree(config.UPLOAD_FOLDER / job_id, ignore_errors=True)

    def test_chunked_upload_cancel(self):
        """Cancel should remove the session and delete files."""
        init_res = self.client.post("/api/upload/init", json={
            "filename": "cancel_test.mp4",
            "total_size": 1000,
            "chunk_size": 1000,
            "total_chunks": 1,
        })
        job_id = init_res.get_json()["job_id"]

        cancel_res = self.client.post("/api/upload/cancel", json={"job_id": job_id})
        self.assertEqual(cancel_res.status_code, 200)
        self.assertEqual(cancel_res.get_json()["status"], "cancelled")
        self.assertFalse((config.UPLOAD_FOLDER / job_id).exists())

    def test_gemini_proxy_generation_decision(self):
        """create_gemini_proxy_video returns original if <= 1.5GB, and triggers proxy if > 1.5GB."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"dummy")
            test_file = tf.name

        try:
            # Case 1: Small file (100MB) -> returns original path, is_proxy=False
            with patch("os.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=100 * 1024 * 1024)
                path_res, is_proxy = create_gemini_proxy_video(test_file, "job_test_small")
                self.assertEqual(path_res, test_file)
                self.assertFalse(is_proxy)

            # Case 2: Large file (2.5GB) -> triggers ffmpeg proxy creation
            with patch("os.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=int(2.5 * 1024 * 1024 * 1024))
                with patch("subprocess.run") as mock_proc:
                    mock_proc.return_value = MagicMock(returncode=0)
                    with patch.object(Path, "exists", return_value=True):
                        path_res, is_proxy = create_gemini_proxy_video(test_file, "job_test_large")
                        self.assertTrue(is_proxy)
                        self.assertIn("gemini_visual_proxy.mp4", path_res)
        finally:
            Path(test_file).unlink(missing_ok=True)
            shutil.rmtree(config.OUTPUT_FOLDER / "job_test_small", ignore_errors=True)
            shutil.rmtree(config.OUTPUT_FOLDER / "job_test_large", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
