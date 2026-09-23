# Kế hoạch Hybrid Subtitle Pipeline — 17/09/2026

## Mục tiêu

Tạo một luồng chung cho Gemini Vision và Whisper:

1. Trích xuất nguồn chữ bằng Gemini Vision, Whisper hoặc cả hai.
2. Chuẩn hóa thành hợp đồng cue ID do Python quản lý timestamp.
3. Gemini AI gộp ngữ nghĩa và dịch bằng danh sách `source_ids`.
4. Python kiểm tra độ phủ, thứ tự và timestamp; AI không được tự quyết định timestamp.
5. Xuất riêng `display_srt` và `tts_srt`.
6. Giữ fallback về dịch 1:1/Google khi provider hoặc contract lỗi.

## Hợp đồng dữ liệu

### SourceCue

```json
{"id": 1, "start_ms": 700, "end_ms": 1800, "text": "安总监真的来了"}
```

### FusionGroup

```json
{
  "source_ids": [4, 5],
  "translation": "Liu Yue is still in CEO Shen's office.",
  "speaker": "F1"
}
```

Ràng buộc:

- Mọi source ID phải xuất hiện đúng một lần.
- ID trong một group phải liên tiếp, tăng dần.
- Các group phải giữ nguyên thứ tự.
- Python lấy `start_ms` của ID đầu và `end_ms` của ID cuối.
- AI response không chứa timestamp.

## Mốc TDD

| Mốc | Test RED | Kết quả GREEN yêu cầu |
|---|---|---|
| 1. Contract cue/group | Parse JSON, missing/duplicate/non-contiguous ID | Reject response sai, nhận response hợp lệ |
| 2. Display SRT | Group ID → timestamp deterministic | SRT hợp lệ, không overlap |
| 3. TTS timeline | Merge display groups theo gap/duration/speaker | Ít cụm hơn, không mất nội dung |
| 4. Provider fallback | Invalid JSON/provider error | Heuristic grouping + translator cũ |
| 5. Pipeline integration | Gemini/Whisper/Hybrid dùng chung postprocessor | `display_srt` và `tts_srt` được lưu đúng |
| 6. Episode 09 E2E | Gemini OCR thực + semantic fusion | Không lỗi timestamp; giảm cue; câu liền nghĩa |

## Phạm vi triển khai lượt này

- Hybrid v1 dùng Gemini video OCR hiện có đã sửa timestamp.
- Whisper được dùng làm transcript đối chiếu trong `hybrid` mode.
- Semantic Fusion dùng model AI qua gateway 8317.
- Python giữ timestamp và kiểm tra contract.
- Tách SRT hiển thị và SRT đầu vào TTS.
- Chưa thay Gemini video OCR bằng batch ảnh đại diện trong lượt này; đây là v2 sau khi cue detector được benchmark ổn định.

## Tiêu chí hoàn tất

- Tests mới và suite hiện có cùng xanh.
- Episode 09 tạo được OCR SRT hợp lệ.
- Semantic fusion giảm số cue hoặc giữ nguyên khi không an toàn.
- Không mất source ID.
- TTS dùng `tts_srt`; download/burn vẫn dùng `display_srt`.
- Provider lỗi không làm mất SRT nguồn.
