# Báo cáo triển khai nâng cấp Studio — 16/09/2026

Đã triển khai các thay đổi về tính đúng đắn của pipeline, lưu trạng thái, upload, duyệt SRT, TTS, OCR, render và vận hành Windows. Thay đổi hiện nằm trong working tree, trên nền commit `bd36db8`. Đây là kết quả kiểm tra mã nguồn và kiểm thử cục bộ; không phải chứng nhận chất lượng đầu ra từ các nhà cung cấp AI.

## Kết quả kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Toàn bộ pytest | **95 passed, 14 warnings**, 24,79 giây; trước nâng cấp: 63 tests |
| Kiểm thử logic editor bằng Node | **3 passed**: polling không ghi đè bản sửa, xung đột revision chặn submit, khôi phục nháp local khác phiên bản |
| JavaScript trong HTML phục vụ bởi Flask | Parse thành công bằng `node --check` |
| Python compileall | Thành công |
| pip check trong venv dự án | No broken requirements found |
| PowerShell launcher | Parse thành công; không chạy start/stop trên dịch vụ đang dùng |
| git diff --check | Thành công |
| FFmpeg/FFprobe thực | Video mẫu 160×120, 2 giây: có video + audio, thời lượng sai lệch dưới 150 ms; ROI 40×24 pixel đúng lựa chọn |

14 cảnh báo thuộc `audioop`, EasyOCR/PyTorch quantization và thư viện phụ thuộc. Bộ kiểm thử tự cách ly thư mục upload/output/runtime và cấu hình monitor; các lời gọi dịch, TTS và Telegram trong các ca hồi quy mới đều dùng mock. Không gửi tin Telegram, crawl kênh thực hoặc gọi AI trả phí để kiểm chứng bản nâng cấp này.

Bằng chứng máy đọc được: [implementation-tests.xml](implementation-tests.xml), [alignment-benchmark.json](alignment-benchmark.json). Tests mới nằm tại `tests/test_studio_upgrade.py` và `tests/studio-review.test.cjs`.

## Đối chiếu 15 phát hiện của báo cáo phân tích

| Mã | Thay đổi đã triển khai | Phạm vi / giới hạn còn lại |
|---|---|---|
| F01 | Upload chunk loại `pipeline` chỉ trả nguồn đã upload; không gọi thêm Whisper. Finish idempotent. | Kiểm thử xác nhận không dispatch worker ngoài ý muốn. |
| F02 | GET/PUT toàn bộ SRT, autosave có revision, lưu nháp local, polling không ghi đè. Có khôi phục nháp khác phiên bản sau tải lại trang. | Xung đột revision cần người dùng chọn bản muốn giữ; không tự gộp nội dung SRT. |
| F03 | Kiểm tra đầu vào, trạng thái con, file đầu ra, cancellation; TTS partial dừng trước burn. Whisper/monitor có chế độ không kết thúc parent. Hardsub lỗi dịch/render không còn báo done. | Request HTTP đã gửi có thể phải chờ timeout/response mới nhận hủy. Không chứng minh mọi tích hợp bên ngoài đã hủy tức thời. |
| F04 | SQLite dùng chung + khóa OS cho media/GPU/monitor; Studio dùng executor với hàng chờ có giới hạn. | **Hoàn thành một phần đề xuất kiến trúc:** Web/Bot vẫn là các tiến trình và nhiều route cũ vẫn dispatch thread riêng. Chưa chuyển toàn bộ sang một backend worker duy nhất. |
| F05 | Chỉ ghi history khi tải/xử lý đạt kết quả yêu cầu; lỗi còn cơ hội thử lại trong cửa sổ backfill. Có phân trang và thử lại giao file đã render. | Backfill mặc định 72 giờ; tối đa 100 video/20 trang. Delivery giới hạn 5 lượt, không đảm bảo exactly-once khi timeout sau khi Telegram đã nhận file. |
| F06 | Timeout GPU phát sinh lỗi; không cho tác vụ chạy vượt semaphore. Khóa GPU liên tiến trình, cho phép gọi lồng trong cùng thread. | Đã kiểm thử timeout và loại trừ liên tiến trình bằng subprocess thật. |
| F07 | Bảo vệ file job đang chạy/chờ duyệt; snapshot được giữ qua restart, owner chết chuyển interrupted; không suy đoán job cũ đã thành công chỉ từ tên file. | **Hoàn thành phần bảo vệ/phục hồi:** chưa có lịch tự phát lại hàng đợi sau crash hoặc chính sách archive/vacuum DB. Bản upload/chờ duyệt có thể cần hủy chủ động để thu hồi dung lượng. |
| F08 | Chặn job_id chứa đường dẫn Windows; ràng buộc đường dẫn download; chuẩn hóa tên burn; lọc trường nội bộ/secret trong API. | Web vẫn là ứng dụng local theo cấu hình hiện có, không bổ sung hệ thống tài khoản nhiều người dùng. |
| F09 | Kiểm tra kích thước chunk/file, kiểm tra đĩa, giữ phiên thiếu chunk để tiếp tục; metadata atomic; reload phiên sau restart; không xóa file upload đã bàn giao cho worker. | Kiểm tra độ dài byte, không phải checksum toàn bộ file. |
| F10 | Ghép PCM không overlay toàn master mỗi câu; giữ audio khi câu vượt khoảng thời gian, đẩy câu sau và báo timing_overflow; xuất aligned SRT/manifest. Không xuất file im lặng nếu mọi câu đều lỗi. | PCM master vẫn nằm trong RAM và tăng tuyến tính theo thời lượng; chưa có mixer trên đĩa cho audio nhiều giờ. Duration trong manifest dựa trên PCM, không phải độ dài container MP3 sau padding của codec. |
| F11 | Giữ câu AI trả đúng, chỉ Google fallback câu thiếu; giữ speaker; ghi danh sách fallback/câu Trung chưa đổi và đưa vào kiểm tra. | Chưa có bộ đo chất lượng ngữ nghĩa/WER cho nội dung thực; tên riêng có thể cần người dùng xác nhận. |
| F12 | Web/Bot adapter và CLI/Studio dùng chung `process_srt_to_tts`; truyền speaker, cache theo text/voice/model/style, aligned SRT. Cache OmniVoice phân biệt nội dung/reference; ghi cache Edge/Gemini qua file tạm. | Helper cũ được giữ cho tương thích. Chưa kiểm chứng giọng/clone thực của từng provider. |
| F13 | Gateway OCR chia đoạn 60 giây + overlap 1 giây; encode thích ứng 960p/720p và giới hạn cấu hình mặc định 35 MiB trước base64; checkpoint gắn với nguồn/model; ghép offset và phát hiện output bị cắt. Google SDK cũng kiểm tra finish_reason. | **Hoàn thành phần gateway:** nhánh Google File API chưa chia cửa sổ tương đương. Checkpoint không tự sửa OCR sai/chữ mờ; mâu thuẫn biên nghiêm trọng báo lỗi để kiểm tra. |
| F14 | Clean Plate bỏ trung gian MP4V, pipe frame sang FFmpeg và chỉ encode một lần trong bước clean; ROI ngang/dọc đúng; con trỏ interval thay quét toàn bộ mỗi frame; báo engine/fallback thực. | Xuất CFR/H.264, chưa bảo toàn VFR/HDR. Xuất riêng video sạch rồi burn vẫn cần encode bước burn. Chưa thay toàn bộ decode bằng PyAV theo timestamp nguồn. |
| F15 | Validator options chung; giao diện partial/error/retry/recovery; launcher PID + thời điểm tạo + đường dẫn; từ chối stop khi bận; setup dừng khi lỗi; bổ sung tests/docs. | Chưa tách hết `routes/api.py`, `bot.py`, HTML lớn thành module nhỏ. Launcher mới chỉ quản lý tiến trình do chính nó tạo; không nhận quản lý mù tiến trình cũ. |

## Hiệu năng đo được

So sánh hàm căn chỉnh tại HEAD trước nâng cấp với hàm hiện tại. Mỗi câu là một tone dài 200 ms, bắt đầu cách nhau 1 giây; chạy một lần mỗi cấu hình, bật tracemalloc. Không gọi nhà cung cấp TTS, không đo tải video hoặc export MP3.

| Số câu | Trước | Sau | Bộ nhớ Python peak trước → sau |
|---:|---:|---:|---:|
| 100 | 0,2283 s | 0,0266 s | 20,52 → 9,54 MiB |
| 400 | 6,9509 s | 0,1223 s | 81,76 → 37,01 MiB |

Ở mẫu 400 câu, bước ghép nhanh hơn khoảng 56,8 lần, bộ nhớ peak được tracemalloc theo dõi giảm khoảng 55%. Đây là microbenchmark trên dữ liệu tổng hợp, không đại diện tốc độ toàn pipeline hoặc tổng RAM/GPU của ứng dụng.

## Cách sử dụng bản nâng cấp

1. Khi dịch vụ cũ đã hết công việc, dừng từ cửa sổ đã khởi động nó. Sau đó chạy `start_all.bat`; launcher mới ghi log tại `runtime/web.*.log` và `runtime/bot.*.log`.
2. Với upload gián đoạn, chọn lại đúng file để tiếp tục các chunk còn thiếu. Không dùng hủy upload nếu vẫn muốn giữ phiên.
3. Ở bước duyệt SRT, chờ tải toàn bộ nội dung; chỉnh sửa được lưu nháp. Nếu báo xung đột, tải lại và chọn khôi phục bản nháp trên máy khi cần.
4. Kết quả `partial` vẫn có thể tải artifact đã tạo. Kiểm tra câu thiếu/vượt thời gian; dùng nút thử lại. API retry cho phép cung cấp `target_srt_content` và `tts_options` đã chỉnh. Cache chỉ tái dùng khi nội dung và cấu hình tương ứng không đổi.
5. Job đang chạy khi ứng dụng dừng được hiển thị gián đoạn. Không tự chạy lại provider; người vận hành quyết định thử lại.

Không thay đổi `.env` thực hoặc nâng phiên bản model/thư viện đang cài trong quá trình triển khai. Nên kiểm tra thêm một video ngắn đại diện cho từng workflow thực tế trước khi giao thành phẩm: nghe đủ audio, đối chiếu SRT, kiểm tra phụ đề ở điểm nối OCR và đoạn lời đọc dày.
