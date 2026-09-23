# Hybrid Subtitle Pipeline v1 — triển khai 17/09/2026

## Phạm vi đã triển khai

- Thêm engine `hybrid` và chọn mặc định trên Web Studio.
- Whisper tạo transcript tiếng Trung làm nguồn đối chiếu.
- Gemini Vision là nguồn chữ chính khi OCR thành công; tự rơi về Whisper nếu Vision lỗi.
- Chuẩn hóa timestamp SRT sai định dạng từ Gemini.
- Semantic Fusion dùng contract `source_ids`; AI không được tự tạo timestamp.
- Validator bắt buộc đủ ID, đúng thứ tự, không trùng, ID liên tiếp, không gộp qua khoảng nghỉ >800 ms hoặc nhóm >10 giây.
- Fallback an toàn: gộp deterministic rồi dùng translator hiện tại nếu AI trả JSON/contract sai.
- Xuất hai timeline:
  - `hardsub_<lang>.srt`: phụ đề hiển thị/review/burn.
  - `tts_script_<lang>.srt`: cụm dài hơn cho TTS.
- Khi review chỉnh text, TTS script được sinh lại.
- Aligned TTS SRT không còn thay thế display SRT khi burn ở luồng Semantic Fusion.
- Lưu manifest Semantic Fusion riêng thay vì đưa toàn bộ group vào status polling.
- Thêm quality gate cho chữ Hán còn sót và câu cụt ý.
- Thêm nền tảng Visual Cue Detector OpenCV: quét 4 tín hiệu ảnh, NMS, guard intervals, contact-sheet contract và dedupe OCR states. Chưa bật contact-sheet OCR làm mặc định; OCR video Gemini hiện tại vẫn là nguồn Vision trong v1.

## Kiểm thử thật với Episode 09

Nguồn:
`D:\Naldo\Tools\Douyin_TikTok_Spider\datas\media_datas\quark_dramas\ShenTaiTai_80Eps_Clean\Episode_09.mp4`

Kết quả:

| Track | Số cue | Thời lượng trung bình |
|---|---:|---:|
| Gemini Vision thô | 33 | 1,19 giây |
| Whisper thô | 27 | 1,37 giây |
| Display SRT Hybrid | 14 | 2,96 giây |
| TTS script Hybrid | 10 | khoảng 4,1 giây |

Job đầy đủ: `hybrid_ep09_demo`.

- Whisper đối chiếu thành công.
- Gemini OCR thành công.
- Semantic Fusion contract hợp lệ, không dùng fallback.
- Edge TTS tạo 10/10 cụm, không có segment thất bại hoặc needs-review segment.
- Job kết thúc `done`.
- Không burn video trong thử nghiệm này.

Artifact:

- `outputs/hybrid_ep09_demo/hybrid_zh.srt`
- `outputs/hybrid_ep09_demo/hardsub_en.srt`
- `outputs/hybrid_ep09_demo/tts_script_en.srt`
- `outputs/hybrid_ep09_demo/tts_en.mp3`
- `outputs/hybrid_ep09_demo/subtitles_aligned_en.srt`
- `outputs/hybrid_ep09_demo/tts_manifest_en.json`

## Giới hạn còn lại

- Visual Cue Detector/contact-sheet OCR mới ở mức module nền tảng và tests; chưa thay thế OCR video 60 giây trong pipeline mặc định.
- Speaker do AI suy luận, chưa có diarization âm thanh nên có thể gán sai M1/F1.
- Semantic Fusion hiện gửi một request cho toàn bộ SRT; video rất dài cần batching 30–50 cue.
- Cần browser E2E cho lựa chọn Hybrid Auto và review.
- Gemini/AI provider vẫn là dịch vụ ngoài; khi lỗi contract sẽ fallback và dừng review.
