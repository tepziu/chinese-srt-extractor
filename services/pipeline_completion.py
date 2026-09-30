"""Conservative repair of a lost Studio terminal update after a completed render.

This never starts work or treats an arbitrary file as proof. The recorded final
render must have been reported done by the render stage, remain stable, and
pass a fresh media probe before the parent can be marked complete.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

from services.runtime_state import valid_job_id


def reconcile_final_render(job, job_id: str, output_root: Path, *, grace_seconds: int = 120) -> bool:
    if not valid_job_id(job_id) or job.get('status') not in {'processing', 'interrupted'}:
        return False
    if job.get('active_step') != 'step_5_burn' or job.get('cancel'):
        return False
    config = job.get('pipeline_config')
    if not isinstance(config, dict):
        return False
    steps = config.get('steps') or {}
    if not steps.get('burn_sub'):
        return False
    finished = job.get('step_results') or {}
    if any(steps.get(step) and (finished.get(result) or {}).get('status') != 'done'
           for step, result in (('extract_sub', 'subtitles'), ('clean_video', 'clean'), ('tts', 'tts'))):
        return False
    language = (config.get('sub_options') or {}).get('target_lang', 'vi')
    if language not in {'vi', 'en', 'id', 'zh'}:
        return False
    render = job.get(f'burn_{language}') or {}
    if render.get('status') != 'done' or not render.get('path'):
        return False
    process = job.get('_ffmpeg_process')
    if process is not None and hasattr(process, 'poll') and process.poll() is None:
        return False
    try:
        folder = (Path(output_root) / job_id).resolve()
        path = Path(render['path']).resolve(strict=True)
        if path.parent != folder or not path.is_file():
            return False
        stat = path.stat()
        if (stat.st_size <= 0 or stat.st_size != int(render['size']) or
                time.time() - stat.st_mtime < grace_seconds):
            return False
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type:format=duration',
             '-of', 'json', str(path)], capture_output=True, text=True, timeout=20,
        )
        if probe.returncode != 0:
            return False
        media = json.loads(probe.stdout)
        duration = float((media.get('format') or {}).get('duration') or 0)
        tracks = {stream.get('codec_type') for stream in media.get('streams', [])}
        if not math.isfinite(duration) or duration <= 0 or 'video' not in tracks:
            return False
        if steps.get('tts') and 'audio' not in tracks:
            return False
        # A legitimate worker may have completed during the probe.
        if job.get('status') not in {'processing', 'interrupted'} or job.get('cancel'):
            return False
        artifact_map = dict(job.get('artifacts') or {})
        artifact_map['final_video'] = str(path)
        results = dict(finished)
        results['burn'] = dict(render)
        job.update(
            artifacts=artifact_map, step_results=results, completion_recovered=True,
            status='done', progress=100, active_step='completed',
            message='Video đã xuất xong; trạng thái hoàn tất được khôi phục. Hãy kiểm tra video trước khi sử dụng.',
        )
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return False
