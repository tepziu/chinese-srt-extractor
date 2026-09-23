"""Bounded OCR requests with durable, source-specific checkpoints."""
import base64
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import requests

from services.media_process import run_media
from services.srt_utils import generate_srt, parse_srt_timing, validate_srt


def response_text(data):
    choices = data.get('choices') or []
    if not choices or choices[0].get('finish_reason') not in {'stop', None}:
        raise RuntimeError('OCR trả về thiếu nội dung hoặc bị cắt bởi giới hạn token; hãy thử lại.')
    text = choices[0].get('message', {}).get('content')
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError('OCR không trả về nội dung')
    return text


def extract_windows(video_path, job_id, model_name):
    from config import AI_TRANSLATE_CONFIG, OUTPUT_FOLDER, jobs
    from services.hardsub_gemini import HARDSUB_PROMPT, parse_srt_from_text
    source = Path(video_path)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'json', str(source)], capture_output=True, text=True, timeout=30, check=True)
    duration = float(json.loads(probe.stdout)['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Không đọc được thời lượng video OCR')
    base = AI_TRANSLATE_CONFIG.get('base_url', 'http://127.0.0.1:8317/v1').rstrip('/')
    identity = [str(source.resolve()), source.stat().st_size, source.stat().st_mtime_ns,
                model_name, base, HARDSUB_PROMPT, 'windows-v1']
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    cache = OUTPUT_FOLDER / job_id / 'ocr_cache' / digest
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    # One-second overlap avoids losing text at split boundaries.
    for index, start in enumerate(range(0, math.ceil(duration), 60)):
        if jobs.get(job_id, {}).get('cancel'):
            raise RuntimeError('Đã hủy OCR')
        length = min(61, duration - start)
        checkpoint = cache / f'{index:05d}.json'
        if checkpoint.exists():
            local = json.loads(checkpoint.read_text(encoding='utf-8'))
        else:
            clip = cache / f'{index:05d}.mp4'
            try:
                # Pass 1: Encode tối ưu cho AI OCR đọc chữ (960p, 10fps, maxrate 2000k)
                run_media(['ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-i', str(source),
                           '-t', str(length), '-vf', r'scale=min(960\,iw):-2', '-an',
                           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-r', '10',
                           '-maxrate', '2000k', '-bufsize', '4000k',
                           '-movflags', '+faststart', str(clip)], job_id, timeout=600)
                max_clip_bytes = int(os.getenv('OCR_MAX_CLIP_BYTES', str(35 * 1024 * 1024)))
                # Pass 2: Tự động nén thích ứng nếu video phức tạp vẫn vượt ngưỡng
                if max_clip_bytes > 0 and clip.stat().st_size > max_clip_bytes:
                    run_media(['ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-i', str(source),
                               '-t', str(length), '-vf', r'scale=min(720\,iw):-2', '-an',
                               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '32', '-r', '8',
                               '-maxrate', '1200k', '-bufsize', '2400k',
                               '-movflags', '+faststart', str(clip)], job_id, timeout=600)
                if max_clip_bytes > 0 and clip.stat().st_size > max_clip_bytes:
                    raise ValueError(f'Đoạn OCR ({clip.stat().st_size / (1024*1024):.1f} MiB) vượt giới hạn {max_clip_bytes / (1024*1024):.0f} MiB. Đặt OCR_MAX_CLIP_BYTES=0 trong .env để bỏ qua hoàn toàn.')
                encoded = base64.b64encode(clip.read_bytes()).decode('ascii')
                response = requests.post(base + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + AI_TRANSLATE_CONFIG.get('api_key', '')},
                    json={'model': model_name, 'max_tokens': 8192, 'messages': [{'role': 'user', 'content': [
                        {'type': 'text', 'text': HARDSUB_PROMPT + '\nTimestamps must start at zero for this clip.'},
                        {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + encoded}}]}]},
                    timeout=(15, 300))
                response.raise_for_status()
                raw = response_text(response.json())
                content = parse_srt_from_text(raw)
                if content and not validate_srt(content, duration_ms=round(length * 1000))[0]:
                    raise RuntimeError('OCR trả về SRT không hợp lệ; chưa lưu checkpoint')
                if not content and 'NO_SUBTITLES_FOUND' not in raw:
                    raise RuntimeError('Không đọc được phản hồi OCR')
                local = parse_srt_timing(content)
                temp = checkpoint.with_suffix('.tmp')
                temp.write_text(json.dumps(local, ensure_ascii=False), encoding='utf-8')
                os.replace(temp, checkpoint)
            finally:
                clip.unlink(missing_ok=True)
        for a, b, text in local:
            a, b = a / 1000 + start, min(b / 1000 + start, duration)
            if b <= a:
                continue
            if result and a < result[-1]['end']:
                if text.strip() == result[-1]['text'].strip():
                    result[-1]['end'] = max(result[-1]['end'], b)
                    continue
                # A boundary ambiguity needs correction, never silently discard text.
                if b <= result[-1]['end']:
                    raise RuntimeError('OCR không nhất quán tại ranh giới hai đoạn; cần kiểm tra phụ đề')
                a = result[-1]['end']
            result.append({'start': a, 'end': b, 'text': text})
        job = jobs.get(job_id)
        if job:
            job['message'] = f'OCR: {min(start + 60, duration):.0f}/{duration:.0f} giây (có checkpoint)'
    return generate_srt(result) if result else 'NO_SUBTITLES_FOUND'
