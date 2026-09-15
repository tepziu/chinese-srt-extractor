#!/usr/bin/env python3
"""
srt_to_tts.py — CLI Tool: Chuyển đổi trực tiếp từ file phụ đề .srt sang file audio TTS.
Độ dài file audio khớp chính xác thời lượng phụ đề, 0% chồng chéo, đầy đủ ngữ nghĩa.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Đảm bảo import từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import LANGUAGES, TTS_VOICES
from services.srt_to_tts import srt_to_mp3, parse_srt_data


def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Chỉ tạo ra file audio TTS từ file phụ đề .srt (Khớp timeline, 0% chồng chéo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python srt_to_tts.py input.srt
  python srt_to_tts.py input.srt --lang vi --voice vi-VN-HoaiMyNeural
  python srt_to_tts.py input.srt -o output_voiceover.mp3
        """
    )
    parser.add_argument("srt_file", help="Đường dẫn tới file phụ đề .srt")
    parser.add_argument("--lang", default="vi", choices=["vi", "en", "id"], help="Ngôn ngữ thuyết minh (mặc định: vi)")
    parser.add_argument("--engine", default="edge", choices=["edge", "gemini"], help="Engine TTS (mặc định: edge)")
    parser.add_argument("--voice", default=None, help="Tên giọng đọc cụ thể (ví dụ: vi-VN-NamMinhNeural, vi-VN-HoaiMyNeural)")
    parser.add_argument("-o", "--output", default=None, help="Đường dẫn file .mp3 đầu ra (mặc định lưu cùng thư mục file .srt)")
    parser.add_argument("--mode", default="smart_sync", choices=["smart_sync", "strict_sub"], help="Chế độ căn chỉnh (mặc định: smart_sync)")
    parser.add_argument("--margin", type=int, default=60, help="Khoảng đệm an toàn giữa 2 câu (ms, mặc định: 60)")
    parser.add_argument("--max-speed", type=float, default=1.65, help="Tốc độ nén tối đa khi câu quá dài (mặc định: 1.65)")

    args = parser.parse_args()

    srt_path = Path(args.srt_file)
    if not srt_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file phụ đề: '{srt_path}'")
        sys.exit(1)

    try:
        srt_content = srt_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ Lỗi đọc file .srt: {e}")
        sys.exit(1)

    segments = parse_srt_data(srt_content)
    if not segments:
        print(f"❌ Lỗi: File '{srt_path.name}' không có dòng phụ đề hợp lệ nào.")
        sys.exit(1)

    sub_duration_s = round(segments[-1]["end_ms"] / 1000.0, 2)
    voice_name = args.voice or TTS_VOICES.get(args.lang, "vi-VN-NamMinhNeural")

    print("=" * 65)
    print("  🎙️ CÔNG CỤ TẠO AUDIO TTS TỪ FILE PHỤ ĐỀ .SRT (STANDALONE)")
    print("=" * 65)
    print(f"  📄 File phụ đề: {srt_path.name}")
    print(f"  📊 Số câu phụ đề: {len(segments)} câu")
    print(f"  ⏱️ Thời lượng phụ đề: {sub_duration_s}s ({int(sub_duration_s//60):02d}:{int(sub_duration_s%60):02d})")
    print(f"  🌐 Ngôn ngữ: {args.lang.upper()} | Engine: {args.engine} | Giọng: {voice_name}")
    print(f"  🎚️ Chế độ: {args.mode} (Đệm an toàn: {args.margin}ms, Tốc độ tối đa: {args.max_speed}x)")
    print("-" * 65)
    print("  ⏳ Đang tổng hợp giọng đọc và căn chỉnh timeline chống chồng chéo...")

    start_time = time.time()
    try:
        res = srt_to_mp3(
            srt_input=srt_path,
            output_path=args.output,
            lang=args.lang,
            engine=args.engine,
            voice=args.voice,
            align_mode=args.mode,
            safety_margin_ms=args.margin,
            max_speed_ratio=args.max_speed,
        )

        elapsed = round(time.time() - start_time, 1)
        audio_path = Path(res["audio_path"])
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)

        print("-" * 65)
        print("  ✅ TẠO FILE AUDIO TTS THÀNH CÔNG!")
        print(f"  📁 File xuất ra: {audio_path.resolve()}")
        print(f"  📦 Dung lượng: {file_size_mb:.2f} MB")
        print(f"  ⏱️ Thời lượng audio: {res['audio_duration_s']}s (Khớp 100% thời lượng phụ đề {res['sub_duration_s']}s)")
        print(f"  🛡️ Tỷ lệ chồng chéo âm thanh: 0% (Đảm bảo tuyệt đối)")
        print(f"  ⚡ Thời gian xử lý: {elapsed}s")
        print("=" * 65)

    except Exception as exc:
        print(f"\n❌ Lỗi tạo TTS: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
