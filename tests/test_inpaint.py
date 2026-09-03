import unittest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

from services.video.inpainters.opencv_inpaint import OpenCVInpainter
from services.video.inpainters.lama_inpaint import LamaInpainter
from services.video.mask_generator import generate_text_mask, feather_blend
from services.video.clean_pipeline import clean_video_pipeline

class InpaintTests(unittest.TestCase):
    def test_opencv_inpainter_runs_and_preserves_shape(self):
        inpainter = OpenCVInpainter(method="telea")
        img = np.full((60, 400, 3), 100, dtype=np.uint8)
        mask = np.zeros((60, 400), dtype=np.uint8)
        mask[20:40, 50:350] = 255 # subtitle box

        res = inpainter.inpaint(img, mask)
        self.assertEqual(res.shape, (60, 400, 3))
        self.assertEqual(res.dtype, np.uint8)

    def test_lama_inpainter_initializes_and_runs(self):
        inpainter = LamaInpainter()
        img = np.full((60, 400, 3), 120, dtype=np.uint8)
        mask = np.zeros((60, 400), dtype=np.uint8)
        mask[10:30, 20:200] = 255

        res = inpainter.inpaint(img, mask)
        self.assertEqual(res.shape, (60, 400, 3))

    def test_generate_text_mask_and_feather_blend(self):
        # Create an image with bright text on dark background
        img = np.full((60, 300, 3), 50, dtype=np.uint8)
        cv2.putText(img, "TEST", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        mask = generate_text_mask(img)
        self.assertEqual(mask.shape, (60, 300))
        self.assertGreater(mask.max(), 0)

        # Inpaint and blend
        inpainter = OpenCVInpainter()
        inpainted = inpainter.inpaint(img, mask)
        blended = feather_blend(img, inpainted, mask)
        self.assertEqual(blended.shape, (60, 300, 3))

    def test_clean_video_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_in = os.path.join(tmp_dir, "in.mp4")
            test_out = os.path.join(tmp_dir, "out.mp4")

            # Create a 25-frame test video (1 second)
            vw = cv2.VideoWriter(test_in, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
            for i in range(25):
                frame = np.full((240, 320, 3), 80, dtype=np.uint8)
                if 5 <= i <= 20:
                    cv2.putText(frame, "SUB", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                vw.write(frame)
            vw.release()

            srt = "1\n00:00:00,200 --> 00:00:00,800\nSUB\n"
            sub_region = {"x_ratio": 0.1, "y_ratio": 0.75, "w_ratio": 0.8, "h_ratio": 0.15}

            res = clean_video_pipeline(
                video_path=test_in,
                sub_region=sub_region,
                srt_content=srt,
                output_path=test_out,
                job_id="test_job_clean",
                engine="opencv",
            )
            self.assertEqual(res["status"], "done")
            self.assertTrue(os.path.exists(test_out))
            self.assertGreater(os.path.getsize(test_out), 0)

if __name__ == "__main__":
    unittest.main()
