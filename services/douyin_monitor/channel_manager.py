"""
channel_manager.py — Manages monitored Douyin channels and downloaded video history.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
CHANNELS_FILE = CONFIG_DIR / "monitor_channels.json"
HISTORY_FILE = CONFIG_DIR / "downloaded_history.json"
NOTIFY_CHAT_FILE = CONFIG_DIR / "telegram_notify_chat.json"

_lock = threading.RLock()

DEFAULT_CHANNELS = [
    {
        "channel_id": "Binbinbin9993",
        "nickname": "彬彬说车 (Bân Bân Nói Về Xe)",
        "sec_uid": "MS4wLjABAAAAmaxdB_47fvPWI3k-b2FnxnM561zj829V24EkPOH3pojMLmICQI7r931RDr5NzyjN",
        "target_lang": "en",
        "style": "driving",
        "bgm_mode": "ai",
        "clean_hardsub": True,
        "clean_logo": True,
        "translate_title": True,
        "auto_burn": True,
        "enabled": True,
        "last_check": 0,
    }
]


def load_channels() -> list[dict]:
    """Load list of monitored Douyin channels."""
    with _lock:
        if CHANNELS_FILE.exists():
            try:
                data = json.loads(CHANNELS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as exc:
                print(f"[ChannelManager] Error loading channels: {exc}")
        save_channels(DEFAULT_CHANNELS)
        return list(DEFAULT_CHANNELS)


def save_channels(channels: list[dict]) -> None:
    """Save list of monitored Douyin channels."""
    with _lock:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CHANNELS_FILE.write_text(json.dumps(channels, ensure_ascii=False, indent=2), encoding="utf-8")


def get_channels() -> list[dict]:
    return load_channels()


def add_channel(
    channel_id: str,
    nickname: str = "",
    sec_uid: str = "",
    target_lang: str = "vi",
    style: str = "driving",
    bgm_mode: str = "ai",
    auto_burn: bool = True,
    clean_hardsub: bool = True,
    clean_logo: bool = True,
    translate_title: bool = True,
) -> dict:
    """Add or update a monitored channel."""
    clean_id = channel_id.strip()
    if not clean_id:
        raise ValueError("Channel ID không được để trống")

    with _lock:
        channels = load_channels()
        for ch in channels:
            if ch.get("channel_id", "").lower() == clean_id.lower():
                ch["nickname"] = nickname or ch.get("nickname", clean_id)
                ch["sec_uid"] = sec_uid or ch.get("sec_uid", "")
                ch["target_lang"] = target_lang
                ch["style"] = style
                ch["bgm_mode"] = bgm_mode
                ch["auto_burn"] = auto_burn
                ch["clean_hardsub"] = clean_hardsub
                ch["clean_logo"] = clean_logo
                ch["translate_title"] = translate_title
                ch["enabled"] = True
                save_channels(channels)
                return ch

        new_ch = {
            "channel_id": clean_id,
            "nickname": nickname or clean_id,
            "sec_uid": sec_uid,
            "target_lang": target_lang,
            "style": style,
            "bgm_mode": bgm_mode,
            "clean_hardsub": clean_hardsub,
            "clean_logo": clean_logo,
            "translate_title": translate_title,
            "auto_burn": auto_burn,
            "enabled": True,
            "last_check": 0,
        }
        channels.append(new_ch)
        save_channels(channels)
        return new_ch


def remove_channel(channel_id: str) -> bool:
    """Remove a channel from the monitoring list."""
    clean_id = channel_id.strip().lower()
    with _lock:
        channels = load_channels()
        initial_len = len(channels)
        channels = [ch for ch in channels if ch.get("channel_id", "").lower() != clean_id]
        if len(channels) < initial_len:
            save_channels(channels)
            return True
        return False


def toggle_channel(channel_id: str, enabled: bool) -> bool:
    """Enable or disable monitoring for a specific channel."""
    clean_id = channel_id.strip().lower()
    with _lock:
        channels = load_channels()
        for ch in channels:
            if ch.get("channel_id", "").lower() == clean_id:
                ch["enabled"] = enabled
                save_channels(channels)
                return True
        return False


def update_channel(channel_id: str, updates: dict) -> dict | None:
    """Update settings for a channel."""
    clean_id = channel_id.strip().lower()
    with _lock:
        channels = load_channels()
        for ch in channels:
            if ch.get("channel_id", "").lower() == clean_id:
                for k, v in updates.items():
                    if k != "channel_id":
                        ch[k] = v
                save_channels(channels)
                return ch
    return None


def load_history() -> set[str]:
    """Load set of already processed aweme_ids."""
    with _lock:
        if HISTORY_FILE.exists():
            try:
                data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return set(str(x) for x in data)
            except Exception:
                pass
        return set()


def save_history(history: set[str]) -> None:
    """Save set of processed aweme_ids."""
    with _lock:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(sorted(list(history)), ensure_ascii=False, indent=2), encoding="utf-8")


def get_downloaded_history() -> set[str]:
    return load_history()


def mark_as_downloaded(aweme_id: str) -> None:
    if not aweme_id:
        return
    with _lock:
        h = load_history()
        h.add(str(aweme_id))
        save_history(h)


def set_notify_chat_id(chat_id: int | str) -> None:
    """Save the Telegram chat ID to receive automated video alerts."""
    with _lock:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        NOTIFY_CHAT_FILE.write_text(json.dumps({"chat_id": str(chat_id)}), encoding="utf-8")


def get_notify_chat_id() -> str | None:
    """Get the active Telegram chat ID for notifications."""
    with _lock:
        if NOTIFY_CHAT_FILE.exists():
            try:
                data = json.loads(NOTIFY_CHAT_FILE.read_text(encoding="utf-8"))
                return data.get("chat_id")
            except Exception:
                pass
    return os.getenv("TELEGRAM_CHAT_ID")