# Tối ưu xử lý video - kiểm chứng ngày 30/09/2026

## Phạm vi và trạng thái

Đã sửa code cục bộ; chưa commit/push, không khởi động lại Web/Bot. Baseline là commit
`50891eb`. Đợt này tối ưu công việc lặp trong xử lý ảnh, không đổi provider/model,
không gọi OCR/TTS trả phí và không đổi cấu hình `.env` của người dùng.

Các lỗi Stop-Retry, tín hiệu QC publish outbox, SRT có sẵn sau trim, cờ xử lý/title
và cleanup ở bản rà soát trước **không thuộc đợt tối ưu này và vẫn cần sửa riêng**.

## Thay đổi đã áp dụng

### 1. Tính viền glyph trên bounding box nhỏ, không trên cả vùng phụ đề

`services/video/timing_guard.py`:
- Mỗi component chỉ tạo mask tại bounding box + halo 2 pixel. Halo khớp chính xác
  phạm vi dilation 5x5 cũ, bao gồm trường hợp chạm biên ảnh.
- Thay `np.isin` của các label bằng bảng tra cứu uint8.
- Chia sẻ HSV, bright/dark mask và Canny trong cùng frame; không tính lại chúng khi
  kiểm tra liên tục hoặc tạo mask xóa chữ.
- Giữ nguyên kích thước frame, ROI, ngưỡng glyph và xác nhận ba frame liên tiếp cho
  nội dung chưa có mẫu. Không bỏ quét ngoài SRT và không tăng khoảng thời gian xóa.

### 2. Cache phần base inpaint theo pixel chính xác

`services/video/inpainters/opencv_inpaint.py`:
- Chỉ giữ một ROI gần nhất trong RAM. Chỉ hit khi ảnh, dtype/shape, binary mask,
  method và radius trùng chính xác. Không dùng perceptual hash hay so khớp gần đúng.
- Frame khác dù chỉ một pixel vẫn chạy `cv2.inpaint` bình thường. Cache không bỏ
  frame, không bỏ kiểm tra nguồn và không dùng lại kết quả giữa các video.
- Không cache hạt texture/grain ngẫu nhiên; phần này vẫn chạy riêng từng frame.
- Snapshot được copy, kết quả trả cho caller cũng không thể làm hỏng cache.
- Mặc định giới hạn 1.000.000 pixel: dữ liệu cache giữ lại tối đa khoảng 7 MB với
  ảnh BGR 8-bit; không dùng thêm VRAM. Đây không phải số đo RAM đỉnh toàn tiến trình.
- `OpenCVInpainter(cache_max_pixels=0)` tắt cache và giữ đường tính base cũ.

`services/video/clean_pipeline.py` thêm `inpaint_cache` vào metadata để xem hit,
miss, giới hạn và số byte thực sự được lưu. Khi hit bằng 0, không nên kỳ vọng tăng
thông lượng toàn video nhờ cache.

### 3. Không đổi độ chính xác lấy tốc độ

- Sửa trường hợp mẫu chữ đã học vẫn tồn tại nhưng việc chọn component thất bại vì
  nền đổi độ sáng. Fallback chỉ dùng mẫu nguồn cùng video trong cửa sổ thời gian cũ.
- Giới hạn coverage và kiểm tra spatial overlap hai chiều để panel trắng hoặc một
  dòng chữ khác không giả thành bằng chứng chữ gốc. Không nới điều kiện để chạy nhanh.
- Video không mở được khi presence scan phải báo lỗi, không báo thành công với tập
  interval rỗng.
- Test không còn kế thừa outbox thật từ `.env`; chỉ thay biến môi trường của tiến
  trình pytest. Không sửa cấu hình production hay phát manifest cho hệ thống nhận.

## Số đo trên đúng runtime của ứng dụng

Launcher sử dụng `venv/Scripts/python.exe`: Python 3.12.10, OpenCV 4.14.0,
NumPy 2.4.4. Benchmark xen kẽ baseline/worktree, median ba lần, cùng
nguồn video tổng hợp 960x540/25 fps. Nhận diện ảnh thử ba kích thước ROI. Render dùng
**CPU libx264 fast / CRF 20** cho cả hai bản để không tranh GPU hay đổi chất lượng.

| Công việc | Baseline | Sau sửa | Kết quả |
|---|---:|---:|---:|
| Nhận diện glyph 430x90, 30 lần | 63.87 ms | 43.83 ms | 1.46x |
| Nhận diện glyph 960x144, 30 lần | 144.40 ms | 74.36 ms | 1.94x |
| Nhận diện glyph 1920x216, 30 lần | 409.81 ms | 140.39 ms | 2.92x |
| Presence prepass, 100 frame | 0.259 s | 0.155 s | 1.67x |
| Clean + render, ROI có frame lặp, 100 frame | 9.397 s | 7.711 s | 1.22x; giảm 17.9% thời gian |
| Clean + render, ROI biến đổi, 50 frame | 5.391 s | 5.358 s | 1.01x; không coi là tăng tốc đáng kể |

Mẫu ROI có frame lặp: cache hit 14, miss 58, lưu
967680 byte (~0.92 MiB). Mẫu ROI biến
đổi liên tục: **0 hit**, 36 miss. Việc cache không có lợi đáng kể ở mẫu nền động là
kết quả đo được, không phải lý do để bỏ quét hoặc hạ chất lượng.

### Gate chất lượng

- Các mask core/edge ở ba kích thước ROI bằng baseline từng pixel.
- Presence intervals và số frame phát hiện ngoài SRT giống baseline trên các mẫu
  hỗ trợ cũ; ca đổi sáng mới có test riêng, không dựa vào equality của lỗi cũ.
- Toàn bộ video đầu ra được giải mã. Hai phiên bản có SHA-256 chuỗi pixel và số
  frame bằng nhau: **100/100** cho ROI có frame lặp, **50/50** cho ROI biến đổi.
- Seed grain cố định chỉ trong benchmark để so sánh có thể lặp lại. RNG production
  và mã hóa/audio pipeline không bị đổi.
- Đây không phải chứng nhận sạch chữ mọi font, màu, VFR/HDR hay video thực tế.
  `qc_status=not_checked` của kết quả render vẫn được giữ nguyên.

## Kiểm thử

- Môi trường system/OpenCV 5.0.0: **216 passed, 1 warning**.
- Môi trường venv/OpenCV 4.14.0: **216 passed, 14 warnings**. Warnings
  từ thư viện pydub/EasyOCR/PyTorch, không phải test fail.
- Test Node stop-confirm + studio-review: **6 passed**.
- Regression mới bao phủ rim chạm biên, một HSV/Canny mỗi frame, đổi nền sáng,
  panel trắng và nội dung khác, thay pixel/mask/radius/method/shape, cache budget,
  mutation của kết quả trả về và grain có seed so với đường không cache.
- Syntax và `git diff --check` đã kiểm tra.

## Lệnh tái lập (PowerShell, chạy từ root repository)

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider
node --test tests/stop-confirm.test.cjs tests/studio-review.test.cjs
.\venv\Scripts\python.exe scripts/benchmark_timing_guard.py --baseline-ref 50891eb --repeats 3 --calls 30 --video-frames 100 --render --report docs/audits/2026-09-30-performance-static.json
.\venv\Scripts\python.exe scripts/benchmark_timing_guard.py --baseline-ref 50891eb --repeats 3 --calls 30 --video-frames 50 --render --moving --report docs/audits/2026-09-30-performance-moving.json
```

Báo cáo JSON của runtime ứng dụng:
- `2026-09-30-performance-static.json`
- `2026-09-30-performance-moving.json`

Các bản `2026-09-30-performance-system-*.json` là phép đo phụ trên OpenCV 5.0.0;
không dùng thay số đo của runtime venv. Không so sánh trực tiếp thời gian của hai
case khác số frame để suy ra speedup; mỗi case chỉ so trước/sau cùng input.

## Phần nên đo và tối ưu tiếp

1. Khởi tạo và tái sử dụng EasyOCR Reader có giới hạn bộ nhớ; phải đo cùng Whisper/
   LaMa và tôn trọng lease GPU trước khi giữ model lâu trong tiến trình.
2. Chỉ encode một lần khi người dùng chỉ cần video cuối, không cần clean plate riêng.
   Phải giữ bước duyệt phụ đề, đồng bộ TTS, chính sách audio và artifact contract.
3. Với nền động, benchmark engine xóa chữ GPU trên holdout thực tế rồi kiểm tra
   sạch chữ và chất lượng; không tự đổi engine hay giảm radius/resolution/FPS.

Không tăng media/GPU concurrency, không đổi bitrate/preset, không giảm số frame
kiểm tra, không bỏ gate rights/approval để lấy tốc độ.
