"""
test_douyin_monitor.py — Unit & Integration tests for Douyin Monitor & All-in-One Studio integration.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import json
import os

from app import app
import services.douyin_monitor as dm
from services.douyin_monitor.channel_manager import (
    get_channels, add_channel, remove_channel, toggle_channel,
    get_downloaded_history, mark_as_downloaded,
    set_notify_chat_id, get_notify_chat_id
)
from services.douyin_monitor.crawler import extract_url
from services.douyin_monitor.daemon import (
    start_monitor, stop_monitor, is_monitor_running, get_monitor_status
)


class TestDouyinMonitor(unittest.TestCase):

    def setUp(self):
        self.test_cid = f"test_chan_{int(time.time())}"
        self.app_client = app.test_client()

    def tearDown(self):
        remove_channel(self.test_cid)
        stop_monitor()

    def test_extract_url(self):
        text1 = "4.21 02/10 G@s.OG 12/28 汽车知识 https://v.douyin.com/hBUbjDqMhOI/ 复制此链接，打开DouYin搜索"
        self.assertEqual(extract_url(text1), "https://v.douyin.com/hBUbjDqMhOI/")
        text2 = "https://www.douyin.com/user/MS4wLjABAAAAmaxdB_47fvPWI3k"
        self.assertEqual(extract_url(text2), text2)
        text3 = "Binbinbin9993"
        self.assertEqual(extract_url(text3), "Binbinbin9993")

    def test_channel_manager_crud(self):
        # 1. Add channel
        ch = add_channel(
            channel_id=self.test_cid,
            nickname="Test Channel Nick",
            sec_uid="MS4wLjABAAAATestSecUid1234567890",
            target_lang="vi",
            style="driving",
            bgm_mode="ai"
        )
        self.assertEqual(ch["channel_id"], self.test_cid)
        self.assertEqual(ch["nickname"], "Test Channel Nick")
        self.assertTrue(ch["enabled"])

        # 2. Verify in list
        channels = get_channels()
        found = [c for c in channels if c["channel_id"] == self.test_cid]
        self.assertTrue(len(found) == 1)

        # 3. Toggle channel
        toggle_channel(self.test_cid, False)
        channels = get_channels()
        found = [c for c in channels if c["channel_id"] == self.test_cid]
        self.assertFalse(found[0]["enabled"])

        # 4. Remove channel
        removed = remove_channel(self.test_cid)
        self.assertTrue(removed)
        channels = get_channels()
        found = [c for c in channels if c["channel_id"] == self.test_cid]
        self.assertEqual(len(found), 0)

    def test_history_tracking(self):
        test_aweme = "9999999999999999999"
        mark_as_downloaded(test_aweme)
        hist = get_downloaded_history()
        self.assertIn(test_aweme, hist)

    def test_notify_chat_id(self):
        old_chat = get_notify_chat_id()
        try:
            test_chat = "987654321_test"
            set_notify_chat_id(test_chat)
            self.assertEqual(get_notify_chat_id(), test_chat)
        finally:
            if old_chat:
                set_notify_chat_id(old_chat)
            else:
                from services.douyin_monitor.channel_manager import NOTIFY_CHAT_FILE
                if NOTIFY_CHAT_FILE.exists():
                    NOTIFY_CHAT_FILE.unlink(missing_ok=True)

    def test_daemon_lifecycle(self):
        # Ensure stopped first
        stop_monitor()
        self.assertFalse(is_monitor_running())

        # Start
        started = start_monitor(interval=120)
        self.assertTrue(started)
        self.assertTrue(is_monitor_running())

        st = get_monitor_status()
        self.assertTrue(st["running"])
        self.assertEqual(st["interval"], 120)

        # Stop
        stopped = stop_monitor()
        self.assertTrue(stopped)
        self.assertFalse(is_monitor_running())

    def test_flask_api_endpoints(self):
        # 1. GET channels
        res = self.app_client.get("/api/douyin/channels")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("channels", data)

        # 2. POST add channel
        res = self.app_client.post("/api/douyin/channels", json={
            "channel_id": self.test_cid,
            "nickname": "API Test",
            "sec_uid": "MS4wLjABAAAAApiTestSecUid123456",
            "target_lang": "en",
            "style": "movie",
            "bgm_mode": "duck"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["channel"]["channel_id"], self.test_cid)

        # 3. POST toggle channel
        res = self.app_client.post(f"/api/douyin/channels/{self.test_cid}/toggle", json={"enabled": False})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

        # 4. GET status
        res = self.app_client.get("/api/douyin/monitor/status")
        self.assertEqual(res.status_code, 200)
        st = res.get_json()
        self.assertIn("running", st)
        self.assertIn("total_channels", st)

        # 5. POST toggle monitor
        res = self.app_client.post("/api/douyin/monitor/toggle", json={"enable": False})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.get_json()["running"])

        # 6. DELETE channel
        res = self.app_client.delete(f"/api/douyin/channels/{self.test_cid}")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])


    def test_download_from_url_routes_douyin_master(self):
        """Verify that download_from_url invokes download_douyin_master for Douyin URLs."""
        from unittest.mock import patch, MagicMock
        from services.downloader import download_from_url
        import config

        test_job_id = "test_dy_dl_job"
        config.create_job(test_job_id)

        with patch("services.downloader.download_douyin_master") as mock_master:
            mock_master.return_value = "fake/path/source.mp4"
            with patch("services.downloader.validate_download_url", return_value=(True, "")):
                res = download_from_url(test_job_id, "https://v.douyin.com/xyz123/")
                self.assertEqual(res, "fake/path/source.mp4")
                mock_master.assert_called_once_with(test_job_id, "https://v.douyin.com/xyz123/")


if __name__ == "__main__":
    unittest.main()