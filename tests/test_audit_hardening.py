import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app import app
import config
from config import parse_bool, acquire_gpu_slot, UPLOAD_FOLDER, OUTPUT_FOLDER
from bot import is_telegram_authorized
from services.pipeline_orchestrator import start_pipeline_job, resume_pipeline_job
from services.srt_to_tts import process_srt_to_tts


class TestAuditHardening(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_parse_bool_truthy_and_falsy(self):
        # Truthy
        self.assertTrue(parse_bool(True))
        self.assertTrue(parse_bool("true"))
        self.assertTrue(parse_bool("True"))
        self.assertTrue(parse_bool("1"))
        self.assertTrue(parse_bool(1))
        self.assertTrue(parse_bool("yes"))
        self.assertTrue(parse_bool("on"))

        # Falsy
        self.assertFalse(parse_bool(False))
        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool("False"))
        self.assertFalse(parse_bool("0"))
        self.assertFalse(parse_bool(0))
        self.assertFalse(parse_bool("no"))
        self.assertFalse(parse_bool("off"))

        # Default fallback
        self.assertTrue(parse_bool(None, default=True))
        self.assertFalse(parse_bool(None, default=False))
        self.assertTrue(parse_bool("invalid_val", default=True))
        self.assertFalse(parse_bool("invalid_val", default=False))

    def test_acquire_gpu_slot_lock(self):
        # Successfully acquire and release
        with acquire_gpu_slot(timeout=1.0):
            pass

        # Verify semaphore value remains intact
        self.assertTrue(config.GPU_SEMAPHORE.acquire(blocking=False))
        config.GPU_SEMAPHORE.release()

    def test_telegram_authorization_gate(self):
        # Mock update with allowed user
        mock_update = MagicMock()
        mock_update.effective_user.id = 1046227432
        mock_update.effective_chat.id = 1046227432

        with patch("bot.TELEGRAM_ALLOWED_USER_IDS", {1046227432}):
            with patch("bot.TELEGRAM_ALLOWED_CHAT_IDS", {1046227432}):
                self.assertTrue(is_telegram_authorized(mock_update))

        # Mock update with unauthorized stranger
        mock_stranger = MagicMock()
        mock_stranger.effective_user.id = 999999999
        mock_stranger.effective_chat.id = 888888888

        with patch("bot.TELEGRAM_ALLOWED_USER_IDS", {1046227432}):
            with patch("bot.TELEGRAM_ALLOWED_CHAT_IDS", {1046227432}):
                self.assertFalse(is_telegram_authorized(mock_stranger))

    def test_pipeline_job_id_validation(self):
        # Path traversal / invalid job_id
        with self.assertRaises(ValueError):
            start_pipeline_job("../../etc/passwd", {"source_type": "upload"})

        with self.assertRaises(ValueError):
            start_pipeline_job("bad job id with spaces!", {"source_type": "upload"})

    def test_pipeline_video_path_traversal_rejection(self):
        # Arbitrary file outside uploads/outputs
        with self.assertRaises(ValueError):
            start_pipeline_job(
                "safe_job_12345",
                {
                    "source_type": "upload",
                    "video_path": "C:\\Windows\\System32\\notepad.exe",
                },
            )

    def test_api_pipeline_run_rejects_illegal_path_and_job_id(self):
        # 1. Invalid job_id
        resp = self.app.post(
            "/api/pipeline/run",
            json={"job_id": "../evil_job", "source_type": "upload"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("job_id không hợp lệ", resp.get_json().get("error", ""))

        # 2. Outside video_path
        resp2 = self.app.post(
            "/api/pipeline/run",
            json={
                "job_id": "test_pipe_valid_01",
                "video_path": "C:\\Windows\\System32\\cmd.exe",
            },
        )
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("Đường dẫn video không hợp lệ", resp2.get_json().get("error", ""))

    def test_api_pipeline_continue_validates_srt(self):
        # Malformed SRT
        resp = self.app.post(
            "/api/pipeline/continue/test_pipe_valid_01",
            json={"updated_srt": "NOT A VALID SRT CONTENT"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Phụ đề chỉnh sửa không hợp lệ", resp.get_json().get("error", ""))

    @patch("services.srt_to_tts.synthesize_all_segments")
    def test_tts_partial_status_reporting(self, mock_synth):
        # Simulate 2 segments where segment 1 fails
        from pydub import AudioSegment

        test_job_id = "test_tts_partial_job"
        config.create_job(test_job_id)

        # 1st ok, 2nd failed (None)
        mock_synth.return_value = [
            AudioSegment.silent(duration=1000),
            None,
        ]

        sample_srt = """1
00:00:01,000 --> 00:00:02,000
Câu một

2
00:00:02,500 --> 00:00:03,500
Câu hai bị lỗi
"""
        res = process_srt_to_tts(
            srt_content=sample_srt,
            lang="vi",
            engine="edge",
            job_id=test_job_id,
        )

        # Status must be 'partial'
        self.assertEqual(res["status"], "partial")
        self.assertEqual(res["failed_segments"], [2])

        # Job in global jobs dict must also reflect 'partial'
        job_record = config.get_job(test_job_id)
        self.assertEqual(job_record["status"], "partial")
        self.assertEqual(job_record["tts_vi"]["status"], "partial")
        self.assertEqual(job_record["tts_vi"]["failed_segments"], [2])
        self.assertIn("thiếu 1/2 đoạn", job_record["tts_vi"]["message"])


if __name__ == "__main__":
    unittest.main()
