#!/usr/bin/env python3
"""Chinese SRT Extractor & Translator web entry point."""

from flask import Flask, jsonify

from config import (
    MAX_UPLOAD_BYTES,
    OUTPUT_FOLDER,
    UPLOAD_FOLDER,
    WEB_HOST,
    WEB_PORT,
    cleanup_old_files,
    start_cleanup_worker,
)
from routes.api import api_bp

app = Flask(__name__)
# Neu MAX_UPLOAD_BYTES <= 0: Dat None de Flask/Werkzeug khong gioi han kich thuoc file upload
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES if MAX_UPLOAD_BYTES > 0 else None
app.config["JSON_AS_ASCII"] = False
app.register_blueprint(api_bp)


@app.errorhandler(413)
def request_too_large(_error):
    limit_text = f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB" if MAX_UPLOAD_BYTES > 0 else "Không giới hạn"
    return jsonify({
        "error": f"File quá lớn. Giới hạn hiện tại: {limit_text}",
    }), 413


if __name__ == "__main__":
    print("Đang dọn dẹp file tạm cũ...")
    cleanup_old_files()
    start_cleanup_worker()

    # Khởi động Daemon Giám Sát Kênh Douyin Tự Động
    try:
        from services.douyin_monitor import is_monitor_running, start_monitor, get_channels
        if not is_monitor_running():
            start_monitor(interval=3600)
            channels = get_channels()
            active_chs = len([c for c in channels if c.get("enabled")])
            print(f"  [DouyinMonitor] Daemon tự động chạy ngầm: Đã kích hoạt {active_chs}/{len(channels)} kênh (quét mỗi 1 tiếng)")
    except Exception as e:
        print(f"  [DouyinMonitor] Khởi động monitor warning: {e}")

    print("=" * 60)
    print("  Chinese SRT Extractor & Translator — All-in-One Studio")
    print("  Trích xuất, dịch AI & Tự động quét kênh Douyin")
    print("=" * 60)
    print(f"  Upload folder: {UPLOAD_FOLDER}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  Mở trình duyệt: http://{WEB_HOST}:{WEB_PORT}")
    print("  Chế độ Web mặc định chỉ lắng nghe trên localhost")
    print("=" * 60)
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)