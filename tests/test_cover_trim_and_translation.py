import os
import unittest
import numpy as np

from services.video.trimmer import (
    shift_srt_timestamps,
    detect_intro_cover,
    preprocess_trim_video,
)
from services.video.title_detector import (
    translate_title,
    generate_title_ass_style_and_event,
)
from services.audio_mixer import (
    has_audio_stream,
    mix_voiceover_with_bgm,
)
from app import app


class TestCoverTrimAndTranslation(unittest.TestCase):

    def test_shift_srt_timestamps(self):
        sample = """1
00:00:01,000 --> 00:00:03,000
Xin chào thế giới

2
00:00:03,500 --> 00:00:06,000
Đoạn thứ hai
"""
        shifted = shift_srt_timestamps(sample, 1.0)
        self.assertIn("00:00:00,000 --> 00:00:02,000", shifted)
        self.assertIn("00:00:02,500 --> 00:00:05,000", shifted)
        self.assertIn("Xin chào thế giới", shifted)

    def test_shift_srt_drops_prior_to_cut(self):
        sample = """1
00:00:00,100 --> 00:00:00,800
Chữ trên bìa cũ

2
00:00:01,500 --> 00:00:04,000
Câu thoại đầu tiên
"""
        shifted = shift_srt_timestamps(sample, 1.2)
        self.assertNotIn("Chữ trên bìa cũ", shifted)
        self.assertIn("Câu thoại đầu tiên", shifted)
        self.assertIn("00:00:00,300 --> 00:00:02,800", shifted)

    def test_preprocess_trim_off_returns_original(self):
        path, sec = preprocess_trim_video("test_nvenc.mp4", trim_mode="off")
        self.assertEqual(path, "test_nvenc.mp4")
        self.assertEqual(sec, 0.0)

    def test_title_ass_style_and_event(self):
        style, event = generate_title_ass_style_and_event("COOL TRICKS", 720, 1280, y_ratio=0.08, duration_sec=5.0)
        self.assertIn("TopTitle", style)
        self.assertIn("COOL TRICKS", event)
        self.assertIn("0:00:05.00", event)

    def test_api_presets_endpoint(self):
        client = app.test_client()
        res = client.get("/api/presets")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("default", data)
        self.assertIn("douyin_top_title", data)

    def test_has_audio_stream(self):
        self.assertTrue(has_audio_stream("test_whisper_sample.wav"))
        self.assertFalse(has_audio_stream("non_existent_file.mp4"))

    def test_mix_voiceover_none_mode(self):
        res = mix_voiceover_with_bgm(
            video_path="test_whisper_sample.wav",
            tts_audio_path="test_whisper_sample.wav",
            output_audio_path="test_unused.m4a",
            job_id="test_job",
            bgm_mode="none",
        )
        self.assertEqual(res, "test_whisper_sample.wav")

    def test_mix_voiceover_duck_mode(self):
        out_p = "test_temp_mixed.m4a"
        try:
            res = mix_voiceover_with_bgm(
                video_path="test_whisper_sample.wav",
                tts_audio_path="test_whisper_sample.wav",
                output_audio_path=out_p,
                job_id="test_job",
                bgm_mode="duck",
                bgm_volume=0.8,
            )
            self.assertTrue(os.path.exists(res))
            self.assertGreater(os.path.getsize(res), 1000)
        finally:
            if os.path.exists(out_p):
                try:
                    os.remove(out_p)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
