"""
services/douyin_monitor — Automated Douyin Channel Monitoring & Pipeline Dispatcher.
"""

from services.douyin_monitor.channel_manager import (
    get_channels,
    add_channel,
    remove_channel,
    toggle_channel,
    update_channel,
    get_downloaded_history,
    mark_as_downloaded,
    set_notify_chat_id,
    get_notify_chat_id,
)
from services.douyin_monitor.crawler import (
    resolve_channel_sec_uid,
    fetch_channel_videos,
    download_master_video,
)
from services.douyin_monitor.daemon import (
    start_monitor,
    stop_monitor,
    is_monitor_running,
    get_monitor_status,
    scan_now,
)

__all__ = [
    "get_channels",
    "add_channel",
    "remove_channel",
    "toggle_channel",
    "update_channel",
    "get_downloaded_history",
    "mark_as_downloaded",
    "set_notify_chat_id",
    "get_notify_chat_id",
    "start_monitor",
    "stop_monitor",
    "is_monitor_running",
    "get_monitor_status",
    "scan_now",
    "resolve_channel_sec_uid",
    "fetch_channel_videos",
    "download_master_video",
]