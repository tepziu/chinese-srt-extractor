"""Validate Studio requests before any file write or worker dispatch."""
import math
from config import LANGUAGES, parse_bool, validate_region, UPLOAD_FOLDER, OUTPUT_FOLDER
from services.runtime_state import resolve_media_path
from services.srt_utils import validate_srt


def number(value, default, low, high, integer=False):
    value = default if value is None else value
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("Tùy chọn số không hợp lệ") from None
    if not math.isfinite(result) or not low <= result <= high or (integer and not result.is_integer()):
        raise ValueError(f"Tùy chọn số phải nằm trong [{low}, {high}]")
    return int(result) if integer else result


def validate_tts_options(options):
    result = dict(options or {})
    result['safety_margin_ms'] = number(result.get('safety_margin_ms'), 60, 0, 500, True)
    result['max_speed_ratio'] = number(result.get('max_speed_ratio'), 1.65, 1, 3)
    if result.get('target_duration_ms') is not None:
        result['target_duration_ms'] = number(result['target_duration_ms'], 0, 0, 7*86400000, True)
    if result.get('align_mode', 'smart_sync') not in {'smart_sync', 'strict_sub'}:
        raise ValueError("Chế độ căn chỉnh TTS không hợp lệ")
    return result


def validate_pipeline_options(data):
    if not isinstance(data, dict):
        raise ValueError("Cấu hình pipeline phải là JSON object")
    result = dict(data)
    if result.get('source_type', 'upload') not in {'upload', 'url', 'srt_only'}:
        raise ValueError("Loại nguồn không hợp lệ")
    defaults = {'download': True, 'extract_sub': True, 'clean_video': False, 'tts': True, 'burn_sub': True}
    if 'steps' in result and not isinstance(result['steps'], dict):
        raise ValueError("steps phải là object")
    # A supplied checklist is complete: omitted steps are disabled.
    result['steps'] = {k: parse_bool(result.get('steps', defaults).get(k), False) for k in defaults}
    if not any(result['steps'].values()):
        raise ValueError("Chưa chọn công đoạn xử lý")
    for name in ('sub_options', 'clean_options', 'tts_options', 'burn_options'):
        if not isinstance(result.get(name, {}), dict):
            raise ValueError(f"{name} phải là object")
        result[name] = dict(result.get(name, {}))
    sub, clean, tts, burn = (result[n] for n in ('sub_options','clean_options','tts_options','burn_options'))
    if sub.get('target_lang', 'vi') not in LANGUAGES:
        raise ValueError("Ngôn ngữ không được hỗ trợ")
    if sub.get('engine', 'hybrid') not in {'gemini', 'whisper', 'hybrid'}:
        raise ValueError("Engine phụ đề không được hỗ trợ")
    sub['pause_for_review'] = parse_bool(sub.get('pause_for_review'), False)
    if tts.get('engine', 'edge') not in {'edge', 'gemini', 'omnivoice'}:
        raise ValueError("Engine TTS không được hỗ trợ")
    if clean.get('engine', 'opencv') not in {'opencv', 'lama'}:
        raise ValueError("Engine xóa chữ không được hỗ trợ")
    if burn.get('render_mode', 'blur') not in {'pure_burn', 'blur', 'inpaint_burn'}:
        raise ValueError('Chế độ render không hợp lệ')
    if burn.get('audio_mode', 'keep_original') not in {'keep_original', 'tts_only', 'tts_ducking', 'tts_ai_bgm', 'tts_clean_bgm'}:
        raise ValueError('Chế độ âm thanh không hợp lệ')
    if result.get('source_type') == 'srt_only' and any(result['steps'][k] for k in ('extract_sub', 'clean_video', 'burn_sub')):
        raise ValueError('Các bước xử lý hình/nhận diện cần video đầu vào')
    tts.update(margin=number(tts.get('margin'), 60, 0, 500, True), max_speed=number(tts.get('max_speed'),1.45,1,3))
    if tts.get('align_mode', 'smart_sync') not in {'smart_sync','strict_sub'}:
        raise ValueError("Chế độ căn chỉnh TTS không hợp lệ")
    burn['bgm_volume'] = number(burn.get('bgm_volume'), 0.8, 0, 1.5)
    for opts in (clean, burn):
        if opts.get('sub_region') and not validate_region(opts['sub_region']):
            raise ValueError("Vùng phụ đề không hợp lệ")
        regions = opts.get('extra_regions') or []
        if not isinstance(regions, list) or len(regions) > 3 or any(not isinstance(r, dict) or not validate_region(r) for r in regions):
            raise ValueError("Vùng bổ sung không hợp lệ")
    for name in ('video_path', 'srt_path', 'target_srt_path'):
        if result.get(name):
            result[name] = str(resolve_media_path(result[name], UPLOAD_FOLDER, OUTPUT_FOLDER))
    for name in ('srt_content', 'target_srt_content', 'zh_srt_content'):
        if result.get(name):
            ok, errors = validate_srt(result[name])
            if not ok:
                raise ValueError('Phụ đề không hợp lệ: ' + '; '.join(errors[:2]))
    return result
