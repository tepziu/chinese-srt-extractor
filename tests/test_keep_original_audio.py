import os
import unittest
from unittest.mock import patch, MagicMock
from services.burn_sub import burn_sub_video
import config

class TestKeepOriginalAudio(unittest.TestCase):
    def setUp(self):
        self.job_id = "test_audio_orig_job"
        config.create_job(
            self.job_id,
            video_path="test_nvenc.mp4",
            video_file={"path": "test_nvenc.mp4"},
            tts_vi={"status": "done", "path": "fake_tts.mp3"},
        )

    def tearDown(self):
        config.jobs.pop(self.job_id, None)

    @patch("services.video.clean_pipeline.clean_video_pipeline")
    def test_inpaint_burn_with_keep_original_audio(self, mock_clean_pipe):
        """When keep_original_audio=True, tts_audio_path must be None even if tts_vi exists."""
        mock_clean_pipe.return_value = {"status": "done", "audio_replaced": False}

        burn_sub_video(
            job_id=self.job_id,
            lang="vi",
            srt_content="1\n00:00:01,000 --> 00:00:02,000\nXin chao",
            render_mode="inpaint_burn",
            keep_original_audio=True,
        )

        mock_clean_pipe.assert_called_once()
        call_kwargs = mock_clean_pipe.call_args.kwargs
        self.assertIsNone(call_kwargs.get("tts_audio_path"))

    @patch("services.video.clean_pipeline.clean_video_pipeline")
    def test_clean_plate_always_keeps_original_audio(self, mock_clean_pipe):
        """Clean plate mode must always keep original audio (tts_audio_path=None)."""
        mock_clean_pipe.return_value = {"status": "done", "audio_replaced": False}

        burn_sub_video(
            job_id=self.job_id,
            lang="clean",
            srt_content="1\n00:00:01,000 --> 00:00:02,000\nXin chao",
            render_mode="clean",
        )

        mock_clean_pipe.assert_called_once()
        call_kwargs = mock_clean_pipe.call_args.kwargs
        self.assertIsNone(call_kwargs.get("tts_audio_path"))

    @patch("services.video.clean_pipeline.clean_video_pipeline")
    def test_orig_only_bgm_mode_keeps_original_audio(self, mock_clean_pipe):
        """When bgm_mode='orig_only', tts_audio_path must be None."""
        mock_clean_pipe.return_value = {"status": "done", "audio_replaced": False}

        burn_sub_video(
            job_id=self.job_id,
            lang="vi",
            srt_content="1\n00:00:01,000 --> 00:00:02,000\nXin chao",
            render_mode="inpaint_burn",
            bgm_mode="orig_only",
        )

        mock_clean_pipe.assert_called_once()
        call_kwargs = mock_clean_pipe.call_args.kwargs
        self.assertIsNone(call_kwargs.get("tts_audio_path"))


if __name__ == "__main__":
    unittest.main()
