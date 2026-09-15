import io
import json
import unittest
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine

from app import app
from config import jobs
from services.srt_to_tts import (
    clean_subtitle_line,
    parse_srt_data,
    trim_silence_padding,
    speed_adjust_audio,
    align_srt_to_tts_timeline,
    process_srt_to_tts,
)


class TestSrtToTts(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_clean_subtitle_line(self):
        text, spk = clean_subtitle_line("[M1] <i>Xin chào mọi người</i>")
        self.assertEqual(text, "Xin chào mọi người")
        self.assertEqual(spk, "M1")

        text2, spk2 = clean_subtitle_line("(F1) <b>Cảm ơn bạn!</b> {\an8}")
        self.assertEqual(text2, "Cảm ơn bạn!")
        self.assertEqual(spk2, "F1")

        text3, spk3 = clean_subtitle_line("Không có nhãn nhân vật.")
        self.assertEqual(text3, "Không có nhãn nhân vật.")
        self.assertIsNone(spk3)

    def test_parse_srt_data(self):
        srt = """1
00:00:01,000 --> 00:00:03,500
[M1] Dòng phụ đề thứ nhất

2
00:00:04,200 --> 00:00:06,000
[F1] Dòng phụ đề thứ hai
"""
        segs = parse_srt_data(srt)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["start_ms"], 1000)
        self.assertEqual(segs[0]["end_ms"], 3500)
        self.assertEqual(segs[0]["duration_ms"], 2500)
        self.assertEqual(segs[0]["speaker"], "M1")
        self.assertEqual(segs[0]["text"], "Dòng phụ đề thứ nhất")

        self.assertEqual(segs[1]["start_ms"], 4200)
        self.assertEqual(segs[1]["end_ms"], 6000)
        self.assertEqual(segs[1]["speaker"], "F1")

    def test_zero_overlap_and_duration_matching(self):
        # 3 segments where segment 2 audio would overflow without adjustment
        srt = """1
00:00:01,000 --> 00:00:03,000
Câu một

2
00:00:03,100 --> 00:00:04,500
Câu hai ngắn nhưng audio dài

3
00:00:04,600 --> 00:00:07,000
Câu ba
"""
        segs = parse_srt_data(srt)
        # Mock audios:
        # seg 0: 1500ms
        # seg 1: 2500ms (orig slot: 1400ms, next at 4600ms. Without atempo it would overlap!)
        # seg 2: 1800ms
        audios = [
            Sine(300).to_audio_segment(duration=1500),
            Sine(400).to_audio_segment(duration=2500),
            Sine(500).to_audio_segment(duration=1800),
        ]

        final_audio, seg_results, aligned_srt, total_dur = align_srt_to_tts_timeline(
            segs, audios, options={"safety_margin_ms": 50, "max_speed_ratio": 1.7}
        )

        # 1. Check zero overlap across all consecutive segments
        for i in range(len(seg_results) - 1):
            curr_end = seg_results[i]["actual_end_ms"]
            next_start = seg_results[i + 1]["actual_start_ms"]
            self.assertLessEqual(
                curr_end,
                next_start,
                f"Overlap detected between segment {i+1} (ends {curr_end}) and {i+2} (starts {next_start})"
            )

        # 2. Check total duration matches subtitle timeline
        self.assertGreaterEqual(total_dur, 7000)
        self.assertEqual(len(final_audio), total_dur)

        # 3. Check aligned SRT timestamps match actual placements
        self.assertIn("00:00:01,000 -->", aligned_srt)
        self.assertIn("Câu một", aligned_srt)
        self.assertIn("Câu ba", aligned_srt)

    def test_api_srt_to_tts_endpoint_json(self):
        sample_srt = """1
00:00:01,000 --> 00:00:03,000
Xin chào Việt Nam.
"""
        resp = self.app.post(
            "/api/srt-to-tts",
            data=json.dumps({"srt_content": sample_srt, "lang": "vi", "engine": "edge"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "started")
        self.assertIn("job_id", data)
        self.assertEqual(data["total_segments"], 1)

    def test_api_srt_to_tts_endpoint_file_upload(self):
        sample_srt = """1
00:00:00,500 --> 00:00:02,500
Thử nghiệm tải file SRT.
"""
        data = {
            "file": (io.BytesIO(sample_srt.encode("utf-8")), "test.srt"),
            "lang": "vi",
            "engine": "edge",
        }
        resp = self.app.post("/api/srt-to-tts", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        res_data = resp.get_json()
        self.assertEqual(res_data["status"], "started")
        self.assertIn("job_id", res_data)

    def test_api_srt_to_tts_voices(self):
        resp = self.app.get("/api/srt-to-tts/voices")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("edge", data)
        self.assertIn("defaults", data)


if __name__ == "__main__":
    unittest.main()
