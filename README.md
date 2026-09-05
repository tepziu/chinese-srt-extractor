# Chinese SRT Extractor & Translator — All-in-One Studio

Studio all-in-one local trên Windows/Linux tích hợp trọn gói:

- **Trích xuất âm thanh & giọng nói**: Faster-Whisper (CUDA/CPU) hoặc trích hardsub từ hình ảnh bằng Gemini OCR.
- **Dịch thuật AI chuẩn phong cách**: Tiếng Việt, English, Bahasa Indonesia với các chế độ chuyên biệt: Lái xe & Mẹo xe (`driving`), Điện ảnh & Kịch bản (`movie`), Dịch sát nghĩa (`literal`), Hài hước Douyin (`fun`).
- **Lồng tiếng TTS đa dạng**: Edge-TTS (nhẹ, nhanh), Gemini TTS (biểu cảm), OmniVoice (voice clone).
- **Tách nhạc nền AI & Ducking**: Giữ nguyên âm thanh động cơ/SFX và nhạc nền gốc bằng Demucs v4 hoặc Sidechain Ducking tự động.
- **Tẩy xóa chữ & In phụ đề**: Dynamic Blur, Inpaint xóa sạch 100% hardsub cũ, xóa logo watermark, dịch & thay banner tiêu đề, tự động cắt ảnh bìa chữ Hán đầu video.
- **Giám sát kênh Douyin tự động (Auto Monitor Daemon)**: Tự động theo dõi các kênh chỉ định, tải bản gốc Master, tự động chạy toàn bộ dây chuyền xử lý và gửi video hoàn chỉnh về Telegram.
- **Điều khiển 2 trong 1**: Giao diện Web UI hiện đại & Telegram Bot tiện lợi.

---

## 🛰️ Tính Năng Mới: Giám Sát Kênh Douyin Tự Động (Unified Architecture)

Hệ thống đã được hợp nhất thành **1 dự án duy nhất**, tích hợp trực tiếp lõi crawler chuyên sâu từ `Douyin_Spider`:
1. **Quét ngầm định kỳ (Daemon)**: Định kỳ 3 phút quét các kênh Douyin được cấu hình (ví dụ: `Binbinbin9993` 彬彬说车).
2. **Bắt link Master không nén**: Tự động tải file video gốc chất lượng cao nhất trực tiếp từ CDN Douyin.
3. **Pipeline tự động khép kín**:
   - Cắt bỏ ảnh bìa tiếng Trung đầu video.
   - Nhận diện giọng nói & Dịch AI theo đúng phong cách.
   - Tách nhạc nền BGM bằng Demucs AI.
   - Xóa chữ cũ và in phụ đề mới sắc nét.
4. **Trả kết quả tức thì về Telegram**:
   - Gửi file video thành phẩm hoàn chỉnh (đã có sub + lồng tiếng).
   - Gửi kèm file phụ đề song ngữ `.srt`.

---

## 🚀 Tính Năng Mới: Xử Lý File Nặng & Không Giới Hạn Dung Lượng (Hỗ Trợ > 2GB)

Hệ thống đã được nâng cấp toàn diện để xử lý trơn tru các video dung lượng siêu nặng (2GB, 5GB, 10GB - 50GB+), phim dài tập và video 4K bitrate cao:

1. **Upload Phân Đoạn Thông Minh (Chunked Upload 20MB)**:
   - Client tự động cắt nhỏ file thành các phần 20MB và truyền tuần tự lên server.
   - Không bị tràn bộ nhớ RAM trình duyệt hay server, chống rớt kết nối mạng.
   - Tự động thử lại (retry 3 lần) cho từng phần nếu mạng chập chờn mà không cần upload lại từ đầu.
   - Hiển thị phần trăm, dung lượng đã tải và tốc độ thực tế (MB/s).
2. **Khắc Phục Triệt Để Giới Hạn 2GB của Google Gemini API (Visual Proxy)**:
   - Google Gemini File API áp đặt giới hạn cứng tối đa 2GB/file.
   - Khi trích xuất hardsub với video > 1.5GB, hệ thống tự động nén tạo một bản visual proxy 720p siêu tốc (bỏ âm thanh, nén nhanh H.264 qua GPU NVENC) trong vài giây.
   - File proxy chỉ ~50-100MB giúp upload lên Gemini chỉ mất vài giây thay vì 15-30 phút.
   - **Đặc biệt**: Video gốc chất lượng cao nguyên bản (2GB+) vẫn được giữ nguyên vẹn để dùng cho khâu in phụ đề, tẩy xóa chữ AI (Inpainting) và render xuất file thành phẩm cuối cùng.
3. **Loại Bỏ Mọi Rào Cản Giới Hạn Kích Thước & Thời Lượng**:
   - `MAX_UPLOAD_BYTES=0`, `MAX_DOWNLOAD_BYTES=0`, `MAX_VIDEO_DURATION_SECONDS=0` (0 = Không giới hạn).
   - Xóa bỏ cờ `--max-filesize 2G` trong yt-dlp.
4. **Dynamic Subprocess & Loại Bỏ Timeout**:
   - Chế độ in phụ đề trực tiếp (Pure Burn), tẩy xóa chữ (Clean Plate) và trích xuất audio được chuyển đổi sang dynamic streaming log (`subprocess.Popen`), theo dõi tiến độ thời gian thực `time=HH:MM:SS` và tốc độ encode `(X.Xx)`, loại bỏ hoàn toàn các lỗi timeout 300s/600s trên video dài.

---

## 🎮 Hướng Dẫn Sử Dụng

### 1. Khởi động 1-Click (`start_all.bat`)
Chỉ cần chạy file `start_all.bat`:
- **Web App**: `http://127.0.0.1:5000` (Tự động kích hoạt Daemon giám sát kênh).
- **Telegram Bot**: Tự động chạy và kết nối.

### 2. Quản lý trên Telegram Bot
- `/follow <id_hoặc_link> [vi|en] [driving|movie]`: Thêm kênh Douyin vào danh sách theo dõi.
  - *Ví dụ:* `/follow Binbinbin9993`
  - *Ví dụ:* `/follow Binbinbin9993 vi driving`
  - *Ví dụ:* `/follow https://v.douyin.com/...`
- `/unfollow <id>`: Bỏ theo dõi kênh.
- `/channels`: Xem danh sách tất cả các kênh đang theo dõi, trạng thái và lần quét cuối.
- `/monitor [on|off|scan]`: Bật, tắt hoặc kích hoạt quét tìm video mới ngay lập tức.
- `/status`: Xem toàn bộ cấu hình hệ thống hiện tại.

### 3. Quản lý trên Web UI
- Mở `http://127.0.0.1:5000`
- Chuyển sang tab **"🛰️ Giám sát Douyin"**:
  - Xem trạng thái tiến trình giám sát và các nút điều khiển nhanh (**Tạm dừng**, **Quét ngay**).
  - Nhập ID hoặc dán link chia sẻ Douyin, bấm **Kiểm tra kênh** để xem trước Tên & Avatar.
  - Tùy chỉnh ngôn ngữ, phong cách dịch, chế độ BGM, cắt bìa rồi bấm **"+ Thêm kênh"**.
  - Bật/tắt công tắc hoặc xóa kênh trực tiếp trên danh sách.

---

## Cài đặt Windows

Yêu cầu: Python 3.10+, FFmpeg, NVIDIA driver nếu dùng CUDA.

```bat
setup_windows.bat
```

Script cài core dependencies và các dependency AI/OCR.

## Cấu hình `.env`

Đảm bảo file `.env` đã có đầy đủ:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
AI_TRANSLATE_BASE_URL=http://127.0.0.1:8317/v1
AI_TRANSLATE_API_KEY=...
AI_TRANSLATE_MODEL=gemini-3.7-flash-high
GEMINI_API_KEY=...
DY_COOKIES=...
DY_TICKET=...
DY_TS_SIGN=...
DY_CLIENT_CERT=...
DY_PRIVATE_KEY=...
```

## Kiểm thử (Test Suite)

Chạy toàn bộ 32 test unit/integration:
```bat
venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```