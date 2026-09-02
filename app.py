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
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["JSON_AS_ASCII"] = False
app.register_blueprint(api_bp)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "error": f"File quá lớn. Giới hạn hiện tại: {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB",
    }), 413


if __name__ == "__main__":
    print("Đang dọn dẹp file tạm cũ...")
    cleanup_old_files()
    start_cleanup_worker()

    print("=" * 60)
    print("  Chinese SRT Extractor & Translator")
    print("  Trích xuất & dịch phụ đề .srt từ video tiếng Trung")
    print("=" * 60)
    print(f"  Upload folder: {UPLOAD_FOLDER}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  Mở trình duyệt: http://{WEB_HOST}:{WEB_PORT}")
    print("  Chế độ Web mặc định chỉ lắng nghe trên localhost")
    print("=" * 60)
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
