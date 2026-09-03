import unittest
from unittest.mock import MagicMock, patch
from config import TRANSLATION_MODES, DEFAULT_TRANSLATION_MODE, create_job, jobs
from services.translation import _parse_numbered_result, translate_srt_ai
from app import app
import json

class TranslationModeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_modes_defined(self):
        self.assertIn("movie", TRANSLATION_MODES)
        self.assertIn("driving", TRANSLATION_MODES)
        self.assertEqual(DEFAULT_TRANSLATION_MODE, "movie")
        self.assertIn("🎬", TRANSLATION_MODES["movie"]["icon"])
        self.assertIn("🚗", TRANSLATION_MODES["driving"]["icon"])

    def test_driving_mode_forces_single_speaker_m1(self):
        sample_srt = """1
00:00:01,000 --> 00:00:03,500
倒车入库时，先观察左后视镜。

2
00:00:03,600 --> 00:00:06,000
当车身与库角距离达到三十公分时，向右打满方向盘。
"""
        job_id = "test_driving_job"
        create_job(job_id, translation_mode="driving")

        mock_response = MagicMock()
        # Even if mock AI outputs F1 or M2, driving mode must force M1
        mock_response.choices = [
            MagicMock(message=MagicMock(content="1. [F1] Khi lùi chuồng, trước tiên hãy quan sát gương chiếu hậu trái.\n2. [M2] Khi khoảng cách thân xe đạt ba mươi phân, hãy đánh kịch lái sang phải."))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            with patch.dict("services.translation.AI_TRANSLATE_CONFIG", {"api_key": "test_key", "base_url": "http://test", "model": "test_model"}):
                result = translate_srt_ai(sample_srt, "vi", job_id, translation_mode="driving")
                self.assertIn("Khi lùi chuồng", result)
                # Check job speakers in driving mode
                job = jobs[job_id]
                self.assertEqual(job["translation_mode"], "driving")
                self.assertEqual(job["segment_speakers"], ["M1", "M1"])
                self.assertEqual(job["speakers"], {"M1": 2})

    def test_movie_mode_preserves_multi_speakers(self):
        sample_srt = """1
00:00:01,000 --> 00:00:03,000
你终于回来了！

2
00:00:03,500 --> 00:00:05,500
是的，我回来了。
"""
        job_id = "test_movie_job"
        create_job(job_id, translation_mode="movie")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="1. [F1] Anh cuối cùng cũng đã trở về rồi!\n2. [M1] Đúng vậy, anh đã về rồi."))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            with patch.dict("services.translation.AI_TRANSLATE_CONFIG", {"api_key": "test_key", "base_url": "http://test", "model": "test_model"}):
                result = translate_srt_ai(sample_srt, "vi", job_id, translation_mode="movie")
                job = jobs[job_id]
                self.assertEqual(job["translation_mode"], "movie")
                self.assertEqual(job["segment_speakers"], ["F1", "M1"])
                self.assertEqual(job["speakers"], {"F1": 1, "M1": 1})

    def test_api_modes_endpoint(self):
        res = self.client.get("/api/translation-modes")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("modes", data)
        self.assertIn("movie", data["modes"])
        self.assertIn("driving", data["modes"])
        self.assertEqual(data["default"], "movie")

    def test_api_url_records_translation_mode(self):
        with patch("routes.api.threading.Thread") as mock_thread, \
             patch("routes.api.validate_download_url", return_value=(True, "")):
            res = self.client.post("/api/url", json={
                "url": "https://v.douyin.com/test1234/",
                "model_size": "large-v3-turbo",
                "translate_langs": ["vi"],
                "translate_method": "ai",
                "translation_mode": "driving",
            })
            self.assertEqual(res.status_code, 200)
            job_id = res.get_json()["job_id"]
            self.assertEqual(jobs[job_id]["translation_mode"], "driving")

if __name__ == "__main__":
    unittest.main()
