"""
crawler.py — Interfaces with Douyin API to search channels, fetch works, and download Master videos.
Uses the integrated spider core in services/douyin_monitor/spider/.
"""

from __future__ import annotations

import os
import re
import secrets
import sys
import time
from pathlib import Path

# Add integrated spider modules to sys.path
SPIDER_LOCAL_DIR = Path(__file__).resolve().parent / "spider"
if SPIDER_LOCAL_DIR.exists() and str(SPIDER_LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(SPIDER_LOCAL_DIR))

# Secondary fallback if needed
SPIDER_DIR = r"D:\Naldo\Tools\Douyin_Spider"
if SPIDER_DIR not in sys.path and os.path.exists(SPIDER_DIR):
    sys.path.append(SPIDER_DIR)

_auth = None


def get_douyin_auth(force_refresh_uifid: bool = True):
    """Lazy initialize Douyin authentication using local .env credentials."""
    global _auth
    if _auth is None:
        try:
            from utils.common_util import load_env

            env_file = Path(__file__).resolve().parent.parent.parent / ".env"
            if not env_file.exists():
                env_file = Path(SPIDER_DIR) / ".env"

            _auth = load_env(env_path=str(env_file), bootstrap_creator=False)
            _auth.cookie["UIFID"] = secrets.token_hex(192)
            print(f"[DouyinCrawler] Auth initialized successfully ({len(_auth.cookie)} cookie keys)")
        except Exception as exc:
            print(f"[DouyinCrawler] Error initializing Douyin auth: {exc}")
            _auth = None
    elif force_refresh_uifid and _auth:
        _auth.cookie["UIFID"] = secrets.token_hex(192)

    return _auth


def extract_url(text: str) -> str:
    """Extract URL from copy-pasted Douyin share text."""
    match = re.search(r'https?://[a-zA-Z0-9\.\-_/]+', text)
    if match:
        return match.group(0)
    return text.strip()


def resolve_channel_sec_uid(channel_input: str) -> tuple[str | None, str, dict]:
    """
    Resolve Douyin unique_id, short_id, profile URL, or short link into sec_uid, nickname, and metadata.
    Returns: (sec_uid, nickname, meta_dict)
    """
    clean_input = extract_url(channel_input.strip())
    meta = {}

    # Case 1: Already a sec_uid
    if clean_input.startswith("MS4wLjABAAAA") and len(clean_input) > 30:
        return clean_input, clean_input, {"sec_uid": clean_input}

    # Case 2: Direct user profile link https://www.douyin.com/user/MS4wLjABAAAA...
    user_match = re.search(r"/user/(MS4wLjABAAAA[a-zA-Z0-9_\-]+)", clean_input)
    if user_match:
        sec_uid = user_match.group(1)
        return sec_uid, sec_uid, {"sec_uid": sec_uid}

    # Case 3: Short URL v.douyin.com
    if "v.douyin.com" in clean_input:
        try:
            import requests
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
            r = requests.get(clean_input, headers=headers, allow_redirects=False, timeout=10)
            location = r.headers.get("Location", "")
            m_user = re.search(r"/user/(MS4wLjABAAAA[a-zA-Z0-9_\-]+)", location)
            if m_user:
                sec_uid = m_user.group(1)
                return sec_uid, sec_uid, {"sec_uid": sec_uid}
        except Exception as exc:
            print(f"[DouyinCrawler] Error following short url: {exc}")

    # Case 4: Search user by username or ID
    auth = get_douyin_auth()
    if not auth:
        return None, clean_input, meta

    try:
        from dy_apis.douyin_api import DouyinAPI

        clean_id = clean_input.replace("@", "").strip()
        users = DouyinAPI.search_some_user(auth, clean_id, 3)
        target = None
        for u in users:
            info = u.get("user_info", {})
            if (
                info.get("unique_id", "").lower() == clean_id.lower()
                or info.get("short_id", "").lower() == clean_id.lower()
            ):
                target = info
                break
        if not target and users:
            target = users[0].get("user_info", {})

        if target:
            sec_uid = target.get("sec_uid")
            nickname = target.get("nickname", clean_id)
            meta = {
                "sec_uid": sec_uid,
                "nickname": nickname,
                "unique_id": target.get("unique_id"),
                "signature": target.get("signature"),
                "follower_count": target.get("follower_count"),
                "avatar": target.get("avatar_thumb", {}).get("url_list", [""])[0],
            }
            return sec_uid, nickname, meta
    except Exception as exc:
        print(f"[DouyinCrawler] Error resolving sec_uid for {clean_input}: {exc}")

    return None, clean_input, meta


def fetch_channel_videos(sec_uid: str, max_count: int = 18) -> list[dict]:
    """Fetch latest videos of a channel, sorted by publish time (newest first)."""
    if not sec_uid:
        return []

    auth = get_douyin_auth(force_refresh_uifid=True)
    if not auth:
        return []

    user_url = f"https://www.douyin.com/user/{sec_uid}"
    try:
        from dy_apis.douyin_api import DouyinAPI
    except Exception as imp_err:
        print(f"[DouyinCrawler] Import error: {imp_err}")
        return []

    for attempt in range(3):
        try:
            auth.cookie["UIFID"] = secrets.token_hex(192)
            res = DouyinAPI.get_user_work_info(auth, user_url, "0")
            aweme_list = res.get("aweme_list", []) if isinstance(res, dict) else []
            if aweme_list:
                works = [w for w in aweme_list if w.get("aweme_type") in [0, 4, 51, 53, 55, 68] or "video" in w]
                if not works:
                    works = aweme_list
                sorted_works = sorted(works, key=lambda x: x.get("create_time", 0), reverse=True)
                return sorted_works[:max_count]
            time.sleep(1.5)
        except Exception as exc:
            if attempt < 2:
                time.sleep(2)
                continue
            print(f"[DouyinCrawler] Error fetching works for {sec_uid[:20]}: {exc}")
            return []

    return []


def download_master_video(work: dict, dest_dir: Path) -> str | None:
    """
    Download uncompressed Master video for a given work dict.
    Returns the absolute path to the downloaded MP4 file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from utils.data_util import download_work, handle_work_info

        work_info = handle_work_info(work)
        saved_path = download_work(work_info, str(dest_dir), save_choice="media-video")
        if not saved_path:
            return None

        # Case 1: saved_path is directly a file
        if os.path.isfile(saved_path) and os.path.getsize(saved_path) > 1000:
            return str(saved_path)

        # Case 2: saved_path is a directory containing video.mp4
        p_dir = Path(saved_path)
        standard_file = p_dir / "video.mp4"
        if standard_file.exists() and standard_file.stat().st_size > 1000:
            return str(standard_file)

        # Case 3: Search any mp4 in the directory
        for f in p_dir.glob("*.mp4"):
            if f.stat().st_size > 1000:
                return str(f)

    except Exception as exc:
        print(f"[DouyinCrawler] Error downloading work: {exc}")
    return None