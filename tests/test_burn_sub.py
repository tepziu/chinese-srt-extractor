import unittest
import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
from services.burn_sub import (
    create_feather_mask,
    _build_blur_filter,
    _srt_to_ass,
    detect_hardsub_region,
    extract_subtitle_intervals,
    build_timeline_enable_expression,
)

class BurnSubTests(unittest.TestCase):
    def test_create_feather_mask_dimensions_and_soft_edges(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = os.path.join(tmp_dir, "test_mask.png")
            create_feather_mask(mask_path, 200, 50, fade_x=20, fade_y=10)
            self.assertTrue(os.path.exists(mask_path))
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.assertEqual(img.shape, (50, 200))
            # Center should be full white (255)
            self.assertEqual(img[25, 100], 255)
            # Corners / edges should fade to 0
            self.assertEqual(img[0, 0], 0)
            self.assertEqual(img[0, 100], 0)
            self.assertEqual(img[25, 0], 0)

    def test_build_blur_filter_contains_alphamerge_and_feathered(self):
        regions = [{"x_ratio": 0.1, "y_ratio": 0.8, "w_ratio": 0.8, "h_ratio": 0.08}]
        fc = _build_blur_filter(720, 1280, regions, "test_sub.ass", mask_input_start=1, timeline_enable="between(t,1,3)")
        self.assertIn("alphamerge", fc)
        self.assertIn("boxblur", fc)
        self.assertIn("ass='test_sub.ass'", fc)
        self.assertIn("crop=", fc)
        self.assertIn(":enable='between(t,1,3)'", fc)

    def test_srt_to_ass_formatting(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nXin chào\n"
        region = {"x_ratio": 0.1, "y_ratio": 0.8, "w_ratio": 0.8, "h_ratio": 0.08}
        ass_text = _srt_to_ass(srt, 720, 1280, region)
        self.assertIn("[Script Info]", ass_text)
        self.assertIn("Style: BurnSub", ass_text)
        self.assertIn("Dialogue:", ass_text)
        self.assertIn("Xin chào", ass_text)

    def test_detect_hardsub_fallback_is_tight(self):
        # Passing non-existent or empty path should return tight fallback
        region = detect_hardsub_region("non_existent_video.mp4")
        self.assertLessEqual(region["h_ratio"], 0.11)
        self.assertLess(region["w_ratio"], 1.0)
        self.assertGreater(region["x_ratio"], 0.0)

    def test_extract_subtitle_intervals_and_merging(self):
        srt = """1
00:00:01,000 --> 00:00:03,000
Câu một

2
00:00:03,200 --> 00:00:05,000
Câu hai (ngắt rất ngắn 0.2s)

3
00:00:09,000 --> 00:00:11,000
Câu ba (cách xa 4s)
"""
        intervals = extract_subtitle_intervals(srt, min_gap=0.35, pad_start=0.12, pad_end=0.15)
        # Sentence 1 and 2 should be merged into 1 single interval because gap (0.2s) < min_gap (0.35s)
        self.assertEqual(len(intervals), 2)
        # First interval should start at ~0.88s and end at ~5.15s
        self.assertAlmostEqual(intervals[0][0], 0.88, places=2)
        self.assertAlmostEqual(intervals[0][1], 5.15, places=2)
        # Second interval starts at ~8.88s and ends at ~11.15s
        self.assertAlmostEqual(intervals[1][0], 8.88, places=2)
        self.assertAlmostEqual(intervals[1][1], 11.15, places=2)

        expr = build_timeline_enable_expression(intervals)
        self.assertIn("between(t,0.880,5.150)", expr)
        self.assertIn("between(t,8.880,11.150)", expr)
        self.assertIn("+", expr)

if __name__ == "__main__":
    unittest.main()
