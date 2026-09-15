import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from app import app
import config
from services.burn_sub import burn_sub_video
from services.hardsub_gemini import hardsub_worker


class TestHardsubInPlaceBurn(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.job_id = "test_hardsub_inplace_job"
        config.create_job(
            self.job_id,
            video_path="test_nvenc.mp4",
            video_file={"path": "test_nvenc.mp4"},
            tts_vi={"status": "done", "path": "fake_tts.mp3"},
        )

    def tearDown(self):
        config.jobs.pop(self.job_id, None)

    @patch("services.burn_sub.subprocess.Popen")
    @patch("services.burn_sub.detect_hardsub_region")
    def test_auto_detect_region_and_keep_original_audio_blur(self, mock_detect, mock_popen):
        """When sub_region is None and keep_original_audio=True in blur mode:
        1. detect_hardsub_region must be called automatically.
        2. FFmpeg must copy original audio (no TTS)."""
        mock_detect.return_value = {
            "x_ratio": 0.08, "y_ratio": 0.86, "w_ratio": 0.84, "h_ratio": 0.09, "method": "ocr_detected"
        }
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (
            json.dumps({"streams": [{"width": 720, "height": 1280}], "format": {"duration": "10"}}),
            "",
        )
        mock_popen.return_value = mock_proc
        mock_popen.return_value.__enter__.return_value = mock_proc

        # Create dummy output file
        out_dir = Path(config.OUTPUT_FOLDER) / self.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "burned_vi.mp4").write_bytes(b"dummy")

        burn_sub_video(
            job_id=self.job_id,
            lang="vi",
            srt_content="1\n00:00:01,000 --> 00:00:03,000\nXin chào",
            sub_region=None,
            render_mode="blur",
            clean_hardsub=True,
            burn_new_sub=True,
            keep_original_audio=True,
            trim_intro="off",
        )

        # 1. detect_hardsub_region must have been called
        mock_detect.assert_called_once()

        # 2. Check FFmpeg command passed to Popen
        ffmpeg_calls = [c[0][0] for c in mock_popen.call_args_list if c[0][0][0] == "ffmpeg"]
        self.assertTrue(len(ffmpeg_calls) > 0)
        cmd = ffmpeg_calls[0]
        self.assertIn("-map", cmd)
        self.assertIn("0:a?", cmd)
        self.assertIn("copy", cmd)
        # TTS file must NOT be in inputs
        self.assertNotIn("fake_tts.mp3", cmd)

    @patch("services.burn_sub.subprocess.Popen")
    @patch("services.burn_sub.detect_hardsub_region")
    def test_manual_custom_region_passed_through(self, mock_detect, mock_popen):
        """When user provides custom sub_region:
        1. detect_hardsub_region must NOT be called.
        2. Custom coordinates are used."""
        custom_region = {"x_ratio": 0.12, "y_ratio": 0.72, "w_ratio": 0.76, "h_ratio": 0.12}
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (
            json.dumps({"streams": [{"width": 720, "height": 1280}], "format": {"duration": "10"}}),
            "",
        )
        mock_popen.return_value = mock_proc
        mock_popen.return_value.__enter__.return_value = mock_proc

        # Create dummy output file
        out_dir = Path(config.OUTPUT_FOLDER) / self.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "burned_vi.mp4").write_bytes(b"dummy")

        res = burn_sub_video(
            job_id=self.job_id,
            lang="vi",
            srt_content="1\n00:00:01,000 --> 00:00:03,000\nXin chào",
            sub_region=custom_region,
            render_mode="blur",
            clean_hardsub=True,
            burn_new_sub=True,
            keep_original_audio=True,
            trim_intro="off",
        )

        # detect_hardsub_region should NOT be called when manual region is provided
        mock_detect.assert_not_called()
        self.assertEqual(res["method"], "manual")
        self.assertEqual(res["sub_region"], custom_region)

    @patch("services.burn_sub.burn_sub_video")
    @patch("services.hardsub_gemini.extract_hardsub_via_local_gateway")
    @patch("services.translation.translate_srt")
    def test_hardsub_worker_auto_burn(self, mock_translate, mock_gateway, mock_burn):
        """When auto_burn=True in job, hardsub_worker must automatically call burn_sub_video with keep_original_audio=True."""
        mock_gateway.return_value = "1\n00:00:01,000 --> 00:00:03,000\n你好\n"
        mock_translate.return_value = "1\n00:00:01,000 --> 00:00:03,000\nXin chào\n"
        mock_burn.return_value = {"status": "done", "path": "burned_video.mp4"}

        job = config.create_job(
            "test_auto_burn_job",
            video_path="test_nvenc.mp4",
            gemini_model="gemini-3.8-flash-high",
            translate_langs=["vi"],
            translate_method="google",
            auto_burn=True,
            burn_region_mode="auto",
            render_mode="inpaint_burn",
            keep_original_audio=True,
            trim_intro="off",
        )

        hardsub_worker(job)

        mock_burn.assert_called_once()
        kwargs = mock_burn.call_args.kwargs
        self.assertEqual(kwargs.get("lang"), "vi")
        self.assertTrue(kwargs.get("keep_original_audio"))
        self.assertTrue(kwargs.get("clean_hardsub"))
        self.assertTrue(kwargs.get("burn_new_sub"))
        self.assertIsNone(kwargs.get("sub_region"))  # auto mode means None

    def test_api_burnsub_receives_manual_sub_region_and_keep_audio(self):
        """API endpoint /api/burnsub receives sub_region and keep_original_audio flag."""
        with patch("routes.api.burnsub_worker") as mock_worker:
            custom_region = {"x_ratio": 0.1, "y_ratio": 0.8, "w_ratio": 0.8, "h_ratio": 0.1}
            srt_path = Path(config.OUTPUT_FOLDER) / self.job_id / "test.srt"
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest", encoding="utf-8")

            config.jobs[self.job_id]["status"] = "done"
            config.jobs[self.job_id]["srt_files"] = {"vi": {"path": str(srt_path)}}

            resp = self.client.post(
                f"/api/burnsub/{self.job_id}/vi",
                json={
                    "render_mode": "blur",
                    "sub_region": custom_region,
                    "bgm_mode": "orig_only",
                    "keep_original_audio": True,
                },
            )
            self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
