# Chinese SRT Extractor & Translator — All-in-One Studio

Studio all-in-one local trên Windows/Linux tích hợp trọn gói:

- **Trích xuất âm thanh & giọng nói**: Faster-Whisper (CUDA/CPU) hoặc trích hardsub từ hình ảnh bằng Gemini OCR.
- **Dịch thuật AI chuẩn phong cách**: Tiếng Việt, English, Bahasa Indonesia với các chế độ chuyên biệt: Lái xe & Mẹo xe (`driving`), Điện ảnh & Kịch bản (`movie`), Dịch sát nghĩa (`literal`), Hài hước Douyin (`fun`).
- **Lồng tiếng TTS đa dạng**: Edge-TTS (nhẹ, nhanh), Gemini TTS (biểu cảm), OmniVoice (voice clone).
- **Tách nhạc nền AI & Ducking**: Giữ nguyên âm thanh động cơ/SFX và nhạc nền gốc bằng Demucs v4 hoặc Sidechain Ducking tự động.
- **Tẩy xóa chữ & In phụ đề**: Dynamic Blur, Inpaint xử lý hardsub cũ, xóa logo watermark, dịch & thay banner tiêu đề, tự động cắt ảnh bìa chữ Hán đầu video.
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

Upload dùng chunk 20 MiB, kiểm tra độ dài từng chunk và dung lượng cuối. Chọn lại cùng file để tiếp tục phiên còn lưu sau lỗi mạng hoặc khởi động lại server. `MAX_UPLOAD_BYTES=0` chỉ bỏ giới hạn ứng dụng; dung lượng đĩa, thời gian xử lý và giới hạn nhà cung cấp vẫn áp dụng.

OCR qua gateway được chia thành các cửa sổ 60 giây với 1 giây chồng biên; mỗi request tối đa 24 MiB video trước base64, có checkpoint và phát hiện phản hồi bị cắt. Nhánh Google SDK vẫn dùng File API/proxy và kiểm tra trạng thái/kết thúc; chưa có chia cửa sổ tương đương.

Clean Plate đưa frame trực tiếp vào FFmpeg để mã hóa một lần trong bước làm sạch, giữ đầy đủ ROI ngang/dọc. Nếu xuất riêng video sạch rồi in thêm phụ đề, bước in vẫn cần mã hóa riêng. Luồng này xuất H.264/CFR; chưa đảm bảo giữ nguyên VFR/HDR của nguồn.

### Căn mốc và chống sót hardsub cho các video mới

- Luồng Studio, trích hardsub độc lập và batch luôn giữ timeline tiếng Trung gốc riêng khỏi phụ đề dịch/TTS. Không cần bật tinh chỉnh mới được dùng thời gian gốc; mốc từ ASR/Gemini vẫn được ghi rõ là ước lượng.
- Clean Plate/Inpaint và che mờ có kiểm tra hình ảnh mặc định: tạo mẫu nét chữ từ chính video/ROI đang xử lý. Ngoài đối chiếu gần ranh giới cue, hệ thống tìm dòng chữ cùng hình thái/vị trí xuất hiện ngoài SRT, chỉ bổ sung khoảng mới sau ít nhất ba frame liên tiếp có mẫu nét ổn định. Các frame đầu của khoảng được bổ sung lại; nền sáng đơn thuần không đủ để xóa. Không OCR nặng trên mọi frame. Có thể tắt bằng `clean_options.visual_guard=false` để đối chiếu/rollback.
- Vùng hardsub gốc được lưu trước khi xóa và truyền sang bước in mới; không dò lại tọa độ trên video sạch chữ. ASS căn giữa khối chữ tại tâm vùng gốc, tự xuống dòng/giảm cỡ chữ theo chiều rộng và cao của vùng. Không cắt bỏ nội dung để ép vừa; câu không thể vừa sẽ có `subtitle_layout.overflow_cues` để kiểm tra.
- Tùy chọn **Thử căn mốc hardsub Trung theo frame gốc** ở Bước 2 vẫn tắt mặc định. Khi bật, OCR xác nhận nội dung ở một vài frame đại diện; xử lý ảnh theo dõi mẫu chữ để đo ranh giới trong cửa sổ tối đa 2 giây. Mốc đầu/cuối được xác nhận riêng; mốc chưa đủ bằng chứng giữ nguyên. Kết quả có tiến độ số câu và số lần OCR; `hardsub_visual_timing.json` ghi bằng chứng từng cue. Bản trước khi chỉnh nằm ở `hardsub_zh_original_timing.srt`.
- Thông tin render `timing_guard` ghi số mẫu, cue thiếu mẫu và frame bổ sung. `qc_status=not_checked` nghĩa là chưa có kiểm tra sạch chữ toàn video; hoàn tất xuất file không phải chứng nhận sạch 100%.
- Bản này ưu tiên chữ sáng/vàng có tương phản/viền tối trong vùng phụ đề đã xác định. Chữ chuyển động/karaoke, font/màu/hình thái khác các mẫu đã học, ROI sai hoặc VFR/HDR vẫn cần kiểm tra. Kiểm tra sự xuất hiện ngoài SRT phục vụ xóa chữ; không tự bổ sung nội dung dịch cho câu mà provider đã bỏ sót. Các trường hợp đó cần xem trước/chỉnh vùng hoặc kiểm tra thủ công; không tự coi là chính xác tuyệt đối.

TTS giữ đủ lời đọc, tạo SRT căn chỉnh và đánh dấu `partial` nếu thiếu câu hoặc lời đọc vượt khoảng thời gian. Trường hợp này cần kiểm tra trước khi render tiếp. Không có cam kết khớp tuyệt đối mọi câu với thời lượng gốc.

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
- Nút **Dừng** trong Studio/TTS/batch mở hộp xác nhận. Chọn **Tiếp tục xử lý** để giữ job; **Xác nhận dừng** mới gửi yêu cầu lên server. UI sẽ báo đã nhận yêu cầu, không khẳng định AI/render đã dừng tức thì. Các tệp đã hoàn tất vẫn được giữ lại. Nếu kết nối lỗi, nút Dừng được bật lại và job vẫn có thể tiếp tục chạy.
- Chuyển sang tab **"🛰️ Giám sát Douyin"**:
  - Xem trạng thái tiến trình giám sát và các nút điều khiển nhanh (**Tạm dừng**, **Quét ngay**).
  - Nhập ID hoặc dán link chia sẻ Douyin, bấm **Kiểm tra kênh** để xem trước Tên & Avatar.
  - Tùy chỉnh ngôn ngữ, phong cách dịch, chế độ BGM, cắt bìa rồi bấm **"+ Thêm kênh"**.
  - Bật/tắt công tắc hoặc xóa kênh trực tiếp trên danh sách.

---

## 📁 Dịch toàn bộ video trong một thư mục

Studio hiện hỗ trợ batch pipeline trên máy Windows local:

1. Mở tab **DỊCH CẢ THƯ MỤC**.
2. Nhập đường dẫn tuyệt đối tới thư mục video.
3. Chọn engine nhận diện phụ đề, phong cách dịch và cách xử lý hardsub Trung cũ.
4. Bấm **BẮT ĐẦU DỊCH THƯ MỤC**.

Mặc định hệ thống sẽ:

- Dịch tiếng Trung sang tiếng Anh.
- Lưu cả `*.zh.srt` và `*.en.srt`.
- Tạo video `*.en.mp4` nếu bật xuất video.
- Giữ âm thanh gốc.
- Che kín toàn bộ vùng phụ đề cũ bằng `opaque_band` để bảo đảm không còn hardsub Trung trước khi in phụ đề Anh.
- Chọn vùng phụ đề theo hai chế độ:
  - `auto`: tự nhận diện riêng trên từng video; vùng phát hiện vừa dùng để xóa/che chữ cũ vừa là vùng in phụ đề Anh.
  - `manual`: dùng một vùng X/Y/W/H chung cho cả thư mục; có preview và có thể kéo chuột trực tiếp trên ảnh để chọn vùng.
- Nút preview có thể nhận diện vùng trên video đầu tiên và cho phép dùng vùng đó làm preset thủ công cho toàn batch.
- Không ghi đè video nguồn.
- Lưu checkpoint tại `outputs/batches/<batch_id>/manifest.json` để có thể resume/retry.

Có thể giới hạn thư mục được phép xử lý bằng `.env`:

```env
BATCH_ALLOWED_ROOTS=D:\Videos;D:\Naldo\Media
BATCH_MAX_FILES=500
```

Nếu bật **Strict translation**, batch sẽ không render video khi provider dịch bị fallback hoặc còn cue tiếng Trung chưa được dịch. Khi đó SRT nguồn và manifest vẫn được giữ lại để retry hoặc sửa tiếp.

## Cài đặt Windows

Yêu cầu: Python 3.10–3.12, FFmpeg/FFprobe, NVIDIA driver nếu dùng CUDA.

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
# Giữ 0 để chỉ tải video gốc qua Master API
DOUYIN_BROWSER_FALLBACK=0
```

## Kiểm thử (Test Suite)

Chạy trong môi trường dự án (tests tự cách ly runtime, upload, output và cấu hình monitor):
```bat
venv\Scripts\python.exe -m pytest -q
node --test tests/studio-review.test.cjs
venv\Scripts\python.exe -m pip check
```

## Hybrid Subtitle Pipeline 17/09/2026

Studio mặc định có chế độ **Hybrid Auto**:

1. Whisper tạo transcript âm thanh để đối chiếu.
2. Gemini Vision đọc chữ hardsub làm nguồn chính.
3. AI Semantic Fusion gộp cue bằng `source_ids` và dịch; Python giữ timestamp.
4. Hệ thống xuất riêng SRT hiển thị và SRT cho TTS.
5. Nếu Vision lỗi, dùng Whisper; nếu AI contract lỗi, dùng gộp/dịch fallback và dừng để review.

Chi tiết và kết quả thử Episode 09: `docs/audits/2026-09-17/hybrid-implementation.vi.md`.

## Nâng cấp Studio 16/09/2026

- Trạng thái job được lưu trong `runtime/jobs.sqlite3`. Tác vụ đang chạy khi tiến trình chết được nhận diện là gián đoạn; dùng **Thử lại** để tiếp tục dựa trên SRT/cache còn tồn tại. Không tự động phát lại tác vụ trả phí sau crash.
- Trình duyệt nhớ job Studio gần nhất, tải đầy đủ SRT để duyệt, lưu nháp và kiểm tra revision nhằm tránh ghi đè bản sửa ở tab khác.
- Web/Bot dùng chung khóa tiến trình cho media/GPU và một monitor trên cùng runtime. Studio có hàng đợi chờ giới hạn (`STUDIO_MAX_PENDING_JOBS`, mặc định 16 mỗi tiến trình).
- Đây là SQLite snapshot + khóa hệ điều hành; chưa phải một dịch vụ worker duy nhất cho mọi route/Bot. Khi chạy nhiều bản sao, chúng phải dùng chung thư mục runtime và dữ liệu.
- Monitor có backfill mặc định 72 giờ, phân trang tối đa 100 video/20 trang mỗi lượt. Xử lý lỗi không ghi thành công. Gửi Telegram lỗi có thể thử lại file đã tạo, tối đa 5 lượt; file quá lớn/chưa còn trên đĩa chuyển sang cần xử lý thủ công.
- `start_all.bat`, `start_web.bat`, `start_bot.bat` chạy dịch vụ ẩn và ghi log vào `runtime/*.log`. `stop_all.bat`/`restart_all.bat` chỉ thao tác PID do launcher này tạo và xác minh; từ chối khi còn công việc đang chạy. Tiến trình được khởi động bằng script cũ cần dừng từ cửa sổ cũ.
- `setup_windows.bat` dừng ngay khi cài đặt lỗi và không xóa venv đang có.
- Upload tạm mặc định được giữ 24 giờ (`UPLOAD_FILE_MAX_AGE_SECONDS`); kết quả trong `outputs` được giữ 30 ngày (`OUTPUT_FILE_MAX_AGE_SECONDS`).
- Khi AI translation gateway lỗi và phải dùng Google fallback, Studio bắt buộc dừng để duyệt trước khi TTS/render. Các câu còn nguyên tiếng Trung không thể được xác nhận nếu chưa sửa.

[Chi tiết triển khai, kiểm thử và giới hạn](docs/audits/2026-09-16/implementation.vi.md).

### Cập nhật phiên Douyin Master

Nếu log báo `Argus/403`, hãy thay đồng bộ các giá trị `DY_COOKIES`, `DY_TICKET`, `DY_TS_SIGN`, `DY_CLIENT_CERT` và `DY_PRIVATE_KEY` lấy từ cùng một phiên đăng nhập/capture. Giữ `DOUYIN_BROWSER_FALLBACK=0` để pipeline chỉ nhận video Master gốc; restart Web/Bot sau khi sửa `.env`. Không trộn cookie mới với ticket/chữ ký cũ.
