# Rà soát lỗi hiện tại — 17/09/2026

**Đánh giá: 82/100 cho công cụ cá nhân chạy local; chưa phù hợp chạy unattended/production.** Pipeline cục bộ, trạng thái job và kiểm thử hồi quy đã khá chắc, nhưng nhà cung cấp AI hiện lỗi xác thực và chưa có E2E thật cho toàn bộ chuỗi Douyin → OCR/dịch → TTS → render → Telegram.

## Lỗi xác nhận từ runtime và trạng thái dữ liệu

| Mức | Lỗi | Bằng chứng | Xử lý |
|---|---|---|---|
| P1 | Gateway dịch `gemini-3.8-flash-high` trả `503 auth_unavailable` | `runtime/web.stdout.log`, hai batch gần nhất | Ghi `translation_provider_warnings`, đánh dấu các dòng fallback và bắt buộc dừng duyệt trước khi render. Cần sửa credential/provider tại gateway 8317 để khôi phục AI translation. |
| P1 | Job có thể được duyệt tiếp dù các câu fallback còn nguyên tiếng Trung | Job `pipe_d99f048904` có 25/25 `unchanged_segments` nhưng từng báo done | API continue giờ so sánh từng dòng đã đánh dấu; từ chối tiếp tục nếu chưa sửa. Có test hồi quy. |
| P1 | Metadata báo `done` nhưng file kết quả đã không còn | Nhiều path của `pipe_d99f048904`, `pipe_a81b584156` không tồn tại | Status endpoint kiểm tra artifact thật, hạ xuống `partial`, xóa link hỏng khỏi response và cho phép retry. Chưa xác định chắc tác nhân bên ngoài đã xóa các file cũ. |
| P1 | Output thành phẩm trước đây có thể bị cleanup sau 4 giờ | `FILE_MAX_AGE_SECONDS=14400` dùng chung upload/output | Tách retention: upload 24 giờ, output 30 ngày; đều cấu hình được qua `.env`. |
| P1 | Douyin crawler thay `UIFID` hợp lệ bằng giá trị ngẫu nhiên ở mỗi lần thử | `services/douyin_monitor/crawler.py` | Giữ nguyên identity đã cấu hình, chỉ sinh khi cookie thật sự thiếu. |
| P2 | Launcher từng báo started khi Web chưa sẵn sàng | PID wrapper tồn tại trước khi cổng 5000 mở | Chờ `/api/device` tối đa 30 giây; Bot phải còn process sau 2 giây. Restart trực tiếp đã kiểm chứng thành công. |
| P2 | Stop/restart bị kẹt bởi job đã cancel nhưng status còn processing | Job `pipe_bb98171d5b` | Stop cập nhật terminal ngay; preflight tự dọn cancel/dead-owner jobs. |
| P3 | `/favicon.ico` trả 404 | `runtime/web.stderr.log` | Chỉ là lỗi thẩm mỹ, chưa ảnh hưởng workflow. |
| P3 | EasyOCR/PyTorch và `audioop` phát cảnh báo deprecation | pytest/runtime warnings | Chưa gây hỏng hiện tại; cần kế hoạch nâng dependency trước Python 3.13/PyTorch loại API cũ. |

## Kiểm chứng sau sửa

- Pytest: **95 passed, 14 warnings**.
- Node editor tests: **3 passed**.
- `compileall`: đạt.
- `pip check`: không có dependency hỏng.
- `git diff --check`: đạt.
- PowerShell launcher parse: đạt.
- Restart thực: Web và Bot báo `ready`; `/api/device` trả HTTP 200; cổng 5000 lắng nghe.
- Artifact integrity: job `pipe_d99f048904` hiện trả `partial` và liệt kê 5 file bị thiếu thay vì báo thành công sai.

## Rủi ro còn lại

1. **Gateway 8317 chưa có auth khả dụng cho provider `antigravity`**. Đây là blocker nếu yêu cầu chất lượng dịch AI; Google fallback chỉ là phương án dự phòng.
2. **Chưa có E2E thật với provider trả phí/Telegram/Douyin monitor trong lượt rà soát này**. Tests mới đều mock các dịch vụ bên ngoài.
3. Web đang chạy Flask development server. Phù hợp `127.0.0.1` cho cá nhân; không nên mở ra LAN/Internet nếu chưa dùng WSGI production, auth và rate limit.
4. Working tree có thay đổi lớn chưa commit. Cần tạo commit/checkpoint trước lượt nâng cấp tiếp theo để rollback được.
5. Chưa cài `ruff`/`pyright`, nên lint và static typing chưa phải quality gate. Compile và tests đã đạt nhưng không thay thế hai kiểm tra này.
6. Các route legacy và Bot vẫn còn nhiều `except Exception`; lỗi có thể bị chuyển thành message chung và thiếu telemetry có cấu trúc.
7. Clean Plate xuất H.264 CFR; chưa bảo toàn VFR/HDR của nguồn.

## Khuyến nghị thứ tự tiếp theo

1. Sửa credential/provider của gateway 8317 và chạy một video thử ngắn, xác nhận AI translation không còn fallback.
2. Chạy E2E có kiểm soát: một video upload và một URL Douyin; xem/nghe toàn bộ output trước khi coi là đạt.
3. Commit checkpoint toàn bộ bản nâng cấp hiện tại.
4. Thêm `ruff`, `pyright` và một browser E2E cho upload → review → continue → download.
5. Nếu cần chạy ngoài localhost, thay Flask dev server và bổ sung auth/rate limiting.
