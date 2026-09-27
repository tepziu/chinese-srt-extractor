"""
daemon.py — Background Daemon for Automated Douyin Channel Monitoring.
Periodically inspects monitored channels for new uploads and triggers the Studio pipeline.
"""

from __future__ import annotations

import os
import json
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from services.runtime_state import host_lock, runtime_dir

from services.douyin_monitor.channel_manager import (
    get_channels,
    get_downloaded_history,
    mark_as_downloaded,
    save_channels,
    update_channel,
)
from services.douyin_monitor.crawler import (
    download_master_video,
    fetch_channel_videos,
    resolve_channel_sec_uid,
)
from services.douyin_monitor.pipeline_bridge import (
    execute_auto_pipeline,
    send_telegram_message,
)
from services.douyin_monitor.ledger import SourceLedger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOWNLOADS_TMP = BASE_DIR / "uploads" / "douyin_monitor_temp"

_monitor_thread: threading.Thread | None = None
_stop_event = threading.Event()
_wake_event = threading.Event()
_lock = threading.Lock()

_status = {
    "running": False,
    "interval": 3600,
    "last_scan_time": None,
    "last_scan_status": "idle",
    "last_error": None,
    "total_scans": 0,
    "current_task": None,
}


def get_monitor_status() -> dict:
    """Return current status of the monitoring daemon."""
    with _lock:
        st = dict(_status)
        if not (_monitor_thread and _monitor_thread.is_alive()):
            try:
                st.update(json.loads((runtime_dir()/'monitor-status.json').read_text(encoding='utf-8')))
            except (OSError, ValueError):
                pass
        st["running"] = is_monitor_running()
        return st


def is_monitor_running() -> bool:
    """Check if background monitoring daemon is active."""
    global _monitor_thread
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return True
    try:
        with host_lock('douyin-monitor', timeout=0):
            return False
    except TimeoutError:
        return True


def _control(**updates):
    with host_lock('monitor-control', timeout=5):
        path = runtime_dir()/'monitor-control.json'
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            data = {'enabled': True, 'scan': 0}
        if updates:
            data.update(updates)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(data), encoding='utf-8')
            temp.replace(path)
        return data


def start_monitor(interval: int = 3600) -> bool:
    """Start the background monitoring loop."""
    global _monitor_thread, _stop_event

    with _lock:
        if is_monitor_running():
            print("[DouyinDaemon] Monitor is already running.")
            return True

        _stop_event.clear()
        _control(enabled=True)
        _wake_event.clear()
        _status["interval"] = interval
        _status["running"] = True
        _status["last_scan_status"] = "starting"

        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(interval,),
            name="DouyinMonitorDaemon",
            daemon=True,
        )
        _monitor_thread.start()
        print(f"[DouyinDaemon] Daemon started with check interval {interval}s.")
        return True


def stop_monitor() -> bool:
    """Stop the background monitoring loop."""
    global _monitor_thread, _stop_event, _wake_event
    _control(enabled=False)

    with _lock:
        if not is_monitor_running():
            _status["running"] = False
            return True

        _stop_event.set()
        _wake_event.set()
        _status["running"] = False
        _status["last_scan_status"] = "stopped"

    if _monitor_thread and _monitor_thread.is_alive():
        _monitor_thread.join(timeout=5)
    print("[DouyinDaemon] Monitor stopped.")
    return not is_monitor_running()


def scan_now() -> dict:
    """Trigger an immediate scan pass without waiting for interval."""
    if not is_monitor_running():
        _stop_event.clear()
        t = threading.Thread(target=_scan_pass, daemon=True)
        t.start()
        return {"status": "started", "message": "Đã kích hoạt quét kênh thủ công (One-off)"}
    else:
        _control(scan=time.time())
        _wake_event.set()
        return {"status": "triggered", "message": "Đã đánh thức tiến trình quét kênh ngay lập tức"}


def _scan_pass() -> dict:
    try:
        with host_lock('douyin-scan', timeout=0):
            return _perform_scan()
    except TimeoutError:
        return {'status': 'busy', 'message': 'Một lượt quét đang hoạt động'}


def _perform_scan() -> dict:
    """Perform one full pass across all enabled channels."""
    from services.douyin_monitor.pipeline_bridge import retry_pending_deliveries
    try:
        retry_pending_deliveries()
    except TimeoutError:
        pass
    with _lock:
        _status["last_scan_status"] = "scanning"
        _status["current_task"] = "Đang nạp danh sách kênh..."

    history = get_downloaded_history()
    ledger = SourceLedger()
    channels = get_channels()
    enabled_channels = [ch for ch in channels if ch.get("enabled", True)]
    found_videos = 0
    errors = []

    print(f"[DouyinDaemon] Bắt đầu lượt quét {len(enabled_channels)} kênh được kích hoạt...")

    for ch in enabled_channels:
        if _stop_event.is_set():
            break

        cid = ch.get("channel_id", "")
        nickname = ch.get("nickname", cid)
        sec_uid = ch.get("sec_uid", "")

        with _lock:
            _status["current_task"] = f"Đang quét kênh: {nickname} ({cid})"

        # 1. Resolve sec_uid if missing or incomplete
        if not sec_uid or sec_uid.startswith("MS4wLjABAAAA_rP") or len(sec_uid) < 30:
            print(f"[DouyinDaemon] Đang phân giải sec_uid cho kênh {cid}...")
            resolved_uid, resolved_nick, *_ = resolve_channel_sec_uid(cid)
            if resolved_uid:
                sec_uid = resolved_uid
                ch["sec_uid"] = resolved_uid
                ch["nickname"] = resolved_nick or nickname
                update_channel(cid, {'sec_uid': resolved_uid, 'nickname': ch['nickname']})
            else:
                err_msg = f"Không tìm thấy sec_uid cho kênh {cid}"
                print(f"[DouyinDaemon] ❌ {err_msg}")
                errors.append(err_msg)
                continue

        # 2. Fetch latest works
        try:
            works = fetch_channel_videos(sec_uid, max_count=100)
            ch["last_check"] = int(time.time())
            update_channel(cid, {'last_check': ch['last_check']})

            if not works:
                continue

            # Cutoff: Only process videos published from today onwards (start of day)
            # Bounded backfill across midnight and downtime; never mark skipped work successful.
            cutoff = int(time.time()) - int(os.getenv('DOUYIN_BACKFILL_HOURS', '72')) * 3600
            
            new_works = []
            for w in works:
                wid = str(w.get("aweme_id", ""))
                ctime = w.get("create_time", 0)
                if not wid or wid in history:
                    continue
                if ctime < cutoff:
                    continue
                new_works.append(w)

            if new_works:
                # Process oldest new video first
                new_works.sort(key=lambda x: x.get("create_time", 0))
                print(f"[DouyinDaemon] 🔥 Kênh [{nickname}] có {len(new_works)} video mới chưa xử lý!")

                for w in new_works:
                    if _stop_event.is_set():
                        break

                    wid = str(w.get("aweme_id", ""))
                    if not ledger.claim(channel_id=cid, work=w):
                        continue
                    wtitle = w.get("desc", "No title")
                    found_videos += 1

                    with _lock:
                        _status["current_task"] = f"Đang tải & xử lý video mới: {wtitle[:30]} ({wid})"

                    # Send immediate Telegram alert upon detection
                    ctime = w.get("create_time", 0)
                    dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ctime)) if ctime else "Hôm nay"
                    send_telegram_message(
                        f"🔥 *Phát hiện video mới từ kênh [{nickname}]!*\n\n"
                        f"📝 *Tiêu đề:* {wtitle}\n"
                        f"🎬 *ID:* `{wid}`\n"
                        f"📅 *Đăng lúc:* {dt_str}\n\n"
                        f"⬇️ *Đang tự động tải video gốc Master về máy...*"
                    )

                    video_dir = DOWNLOADS_TMP / f"{cid}_{wid}"
                    downloaded_file = download_master_video(w, video_dir)

                    if downloaded_file and os.path.exists(downloaded_file):
                        print(f"[DouyinDaemon] ✅ Đã tải video master: {downloaded_file}")
                        ledger.mark(channel_id=cid, work_id=wid, status="DOWNLOADED", artifact_path=downloaded_file)
                        try:
                            ledger.mark(channel_id=cid, work_id=wid, status="PROCESSING")
                            result = execute_auto_pipeline(w, ch, downloaded_file)
                            if not result or result.get('status') not in {'done', 'downloaded'}:
                                raise RuntimeError('Pipeline chưa hoàn tất; giữ lại để thử lại')
                            ledger.mark(channel_id=cid, work_id=wid, status="PACKAGE_READY", artifact_path=result.get("path"), package_path=result.get("publish_package_path"))
                            history.add(wid)
                            mark_as_downloaded(wid)
                        except Exception as proc_err:
                            ledger.mark(channel_id=cid, work_id=wid, status="FAILED", error=str(proc_err))
                            print(f"[DouyinDaemon] ❌ Lỗi xử lý pipeline cho {wid}: {proc_err}")
                            errors.append(f"{wid}: {proc_err}")
                    else:
                        ledger.mark(channel_id=cid, work_id=wid, status="FAILED", error="download failed")
                        print(f"[DouyinDaemon] ⚠️ Không thể tải video master cho {wid}")

                    time.sleep(2)

        except Exception as scan_err:
            err_msg = f"Lỗi quét kênh {nickname}: {scan_err}"
            print(f"[DouyinDaemon] {err_msg}")
            errors.append(err_msg)

        # Anti-ban sleep between channels
        time.sleep(random.uniform(2.5, 4.5))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        _status["last_scan_time"] = now_str
        _status["last_scan_status"] = "success" if not errors else "warning"
        _status["last_error"] = "; ".join(errors) if errors else None
        _status["total_scans"] += 1
        _status["current_task"] = None

    return {
        "time": now_str,
        "channels_scanned": len(enabled_channels),
        "new_videos_found": found_videos,
        "errors": errors,
    }


def _monitor_loop(interval: int):
    try:
        with host_lock('douyin-monitor', timeout=0):
            _run_monitor_loop(interval)
    except TimeoutError:
        print('[DouyinMonitor] Đã có monitor chạy trong tiến trình khác.')


def _run_monitor_loop(interval: int):
    """Main infinite daemon loop."""
    print(f"[DouyinDaemon] Vòng lặp giám sát kênh Douyin bắt đầu (Chu kỳ: {interval}s)...")

    while not _stop_event.is_set() and _control().get('enabled', True):
        scan_revision = _control().get('scan', 0)
        try:
            _scan_pass()
        except Exception as exc:
            print(f"[DouyinDaemon] Unhandled error in scan pass: {exc}")
            with _lock:
                _status["last_error"] = str(exc)

        # Wait for interval or wake event
        (runtime_dir()/'monitor-status.json').write_text(json.dumps(_status, ensure_ascii=False), encoding='utf-8')
        deadline = time.monotonic() + interval
        while time.monotonic() < deadline and not _stop_event.is_set():
            control = _control()
            if not control.get('enabled', True) or control.get('scan', 0) != scan_revision:
                break
            if _wake_event.wait(timeout=0.5):
                break
        _wake_event.clear()

    print("[DouyinDaemon] Vòng lặp giám sát đã kết thúc.")
