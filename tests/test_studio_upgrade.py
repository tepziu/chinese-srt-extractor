"""Regression cases from the September Studio audit. Providers are stubbed."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import config
from app import app
from services.runtime_state import JobRegistry, host_lock, public_data

SRT = '1\n00:00:00,000 --> 00:00:01,000\nXin chào\n'


def test_host_lock_excludes_another_process_and_releases():
    code = "from services.runtime_state import host_lock\nwith host_lock('regression',timeout=0.15): print('acquired')"
    with host_lock('regression'):
        with host_lock('regression'):  # same-thread reentrant
            result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and 'TimeoutError' in result.stderr
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0


def test_gpu_timeout_never_enters_work(monkeypatch):
    semaphore = MagicMock()
    semaphore.acquire.return_value = False
    monkeypatch.setattr(config, 'GPU_SEMAPHORE', semaphore)
    with pytest.raises(TimeoutError):
        with config.acquire_gpu_slot(timeout=0.01):
            pytest.fail('GPU work must not run after timeout')
    semaphore.release.assert_not_called()


def test_durable_nested_state_remote_cancel_and_dead_owner():
    job = config.create_job('persist01', status='processing')
    job['tts_vi'] = {'status': 'generating'}
    job['tts_vi']['status'] = 'partial'
    assert config.jobs.store.load('persist01')['tts_vi']['status'] == 'partial'
    other = JobRegistry()
    other.store.cancel('persist01')
    job._cancel_check = 0
    assert job.get('cancel')
    job.update(_owner_pid=99999999, status='processing')
    assert other.get('persist01')['status'] == 'interrupted'
    assert public_data({'nested': {'api_key': 'hidden', '_proc': 1, 'status': 'done'}}) == {'nested': {'status': 'done'}}


def test_full_review_revision_and_resume_claim(monkeypatch):
    from services import pipeline_orchestrator as pipeline
    full = '\n\n'.join(f'{i}\n00:00:{i:02},000 --> 00:00:{i:02},900\n' + 'Phụ đề dài ' * 10 for i in range(1, 10))
    config.create_job('review01', status='awaiting_review', review_revision=0,
        pending_pipeline={'target_lang': 'vi', 'target_srt_content': full})
    client = app.test_client()
    assert client.get('/api/pipeline/review/review01').json['content'] == full
    assert client.put('/api/pipeline/review/review01', json={'content': SRT, 'revision': 0}).status_code == 200
    assert client.put('/api/pipeline/review/review01', json={'content': 'stale', 'revision': 0}).status_code == 409
    dispatch = MagicMock()
    monkeypatch.setattr(pipeline, 'submit', dispatch)
    result = client.post('/api/pipeline/continue/review01', json={'updated_srt': SRT, 'revision': 1})
    assert result.status_code == 200
    state = config.jobs.store.load('review01')
    assert state['status'] == 'processing' and state['_owner_pid'] == os.getpid()
    assert state['pending_pipeline'] is None
    dispatch.assert_called_once()
    assert client.post('/api/pipeline/continue/review01', json={'updated_srt': SRT, 'revision': 1}).status_code != 200


def test_pipeline_chunk_resume_integrity_idempotency(monkeypatch):
    import routes.api as api
    client = app.test_client()
    initial = client.post('/api/upload/init', json={'filename': 'movie.mkv', 'total_size': 8,
        'chunk_size': 4, 'total_chunks': 2, 'upload_type': 'pipeline'}).json
    jid = initial['job_id']
    def send(index, data):
        return client.post('/api/upload/chunk', data={'job_id': jid, 'chunk_index': index,
            'chunk': (io.BytesIO(data), 'chunk')})
    assert send(0, b'bad').status_code == 400
    assert send(0, b'abcd').status_code == 200
    assert client.post('/api/upload/finish', json={'job_id': jid}).status_code == 400
    api._chunk_uploads.clear()  # simulate restart of upload endpoint
    assert client.get('/api/upload/session/' + jid).json['received_chunks'] == [0]
    assert send(1, b'efgh').status_code == 200
    monkeypatch.setattr(api, '_validate_media_path', lambda _: (True, ''))
    first = client.post('/api/upload/finish', json={'job_id': jid})
    assert first.status_code == 200 and first.json['status'] == 'uploaded'
    assert Path(first.json['video_path']).read_bytes() == b'abcdefgh'
    assert client.post('/api/upload/finish', json={'job_id': jid}).json == first.json
    api.threading.Thread.assert_not_called()
    assert client.post('/api/upload/cancel', json={'job_id': jid}).status_code == 409
    assert Path(first.json['video_path']).exists()


@pytest.mark.parametrize('job_id', ['..%5C..%5Cescape', 'C:%5Csecret', 'short'])
def test_reject_windows_path_in_status(job_id):
    assert app.test_client().get('/api/status/' + job_id).status_code == 400


def test_stop_review_is_terminal():
    config.create_job('cancel01', status='awaiting_review')
    assert app.test_client().post('/api/stop/cancel01').status_code == 200
    assert config.jobs.store.load('cancel01')['status'] == 'cancelled'


def test_dense_tts_preserves_speech_and_valid_timeline(monkeypatch):
    from pydub.generators import Sine
    from services import srt_to_tts as tts
    from services.srt_utils import validate_srt
    segments = tts.parse_srt_data('1\n00:00:00,000 --> 00:00:00,100\nA\n\n2\n00:00:00,110 --> 00:00:00,200\nB')
    tone = Sine(440).to_audio_segment(duration=1000)
    monkeypatch.setattr(tts, 'speed_adjust_audio', lambda audio, *a, **kw: (audio, 1.0))
    audio, rows, aligned, duration = tts.align_srt_to_tts_timeline(segments, [tone, tone])
    assert duration >= 2060 and len(audio) >= 2060
    assert all(r['actual_end_ms'] > r['actual_start_ms'] and r['overlap_ms'] == 0 for r in rows)
    assert all(r['status'] == 'timing_overflow' for r in rows)
    assert validate_srt(aligned)[0]


def test_pipeline_partial_tts_never_burns_or_finishes(monkeypatch, tmp_path):
    from services import pipeline_orchestrator as p, srt_to_tts as tts, burn_sub
    config.create_job('partial1', status='processing', artifacts={})
    monkeypatch.setattr(tts, 'process_srt_to_tts', lambda **kw: {'status': 'partial', 'failed_segments': [1]})
    burn = MagicMock()
    monkeypatch.setattr(burn_sub, 'burn_sub_video', burn)
    p._execute_remaining_steps('partial1', str(tmp_path/'input.mp4'), 'vi', SRT, '',
                           {'tts': True, 'burn_sub': True}, {}, {}, {})
    assert config.jobs['partial1']['status'] == 'partial'
    burn.assert_not_called()


@pytest.mark.parametrize('finish', ['length', 'content_filter', 'error'])
def test_ocr_rejects_truncated_response(finish):
    from services.ocr_windows import response_text
    with pytest.raises(RuntimeError):
        response_text({'choices': [{'finish_reason': finish, 'message': {'content': SRT}}]})


def test_ocr_windows_checkpoint_and_offsets(monkeypatch, tmp_path):
    from services import ocr_windows as ocr
    source = tmp_path/'source.mp4'
    source.write_bytes(b'video')
    monkeypatch.setattr(ocr.subprocess, 'run', lambda *a, **kw: MagicMock(stdout='{"format":{"duration":"121"}}'))
    monkeypatch.setattr(ocr, 'run_media', lambda command, *a, **kw: Path(command[-1]).write_bytes(b'clip'))
    response = MagicMock()
    response.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': SRT}}]}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(ocr.requests, 'post', post)
    output = ocr.extract_windows(str(source), 'ocrtest1', 'test-model')
    assert '00:01:00,000' in output and '00:02:00,000' in output
    assert post.call_count == 3
    assert ocr.extract_windows(str(source), 'ocrtest1', 'test-model') == output
    assert post.call_count == 3
    assert not list((config.OUTPUT_FOLDER/'ocrtest1').rglob('*.mp4'))


def test_clean_video_real_media_single_encode_and_roi(monkeypatch, tmp_path):
    import numpy as np
    from services.video import clean_pipeline as clean
    source, output = tmp_path/'source.mp4', tmp_path/'clean.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=blue:s=160x120:r=10:d=2',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-shortest', str(source)], check=True, timeout=20)
    shapes = []
    def mask(crop, **kwargs):
        shapes.append(crop.shape[:2])
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    monkeypatch.setattr(clean, 'generate_text_mask', mask)
    monkeypatch.setattr(clean, 'encoder_args', lambda: ['-c:v', 'libx264', '-preset', 'ultrafast'])
    config.create_job('render01', status='processing')
    result = clean.clean_video_pipeline(str(source), {'x_ratio': .25, 'y_ratio': .5, 'w_ratio': .25, 'h_ratio': .2},
        SRT, str(output), 'render01')
    assert result['encode_passes'] == 1 and result['method'] == 'clean_opencv'
    assert shapes and set(shapes) == {(24, 40)}
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)],
                            capture_output=True, text=True, check=True, timeout=10)
    data = json.loads(probe.stdout)
    assert {s['codec_type'] for s in data['streams']} == {'audio', 'video'}
    assert abs(float(data['format']['duration']) - 2) < .15


def test_ai_partial_response_preserves_good_lines(monkeypatch):
    import openai
    import deep_translator
    from services import translation as tr
    source = '1\n00:00:00,000 --> 00:00:03,000\n你好\n\n2\n00:00:03,100 --> 00:00:06,000\n再见'
    config.create_job('translate1')
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content='1. [F1] Xin chào'))]
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: client)
    monkeypatch.setitem(tr.AI_TRANSLATE_CONFIG, 'api_key', 'test')
    translator = MagicMock()
    translator.translate.return_value = 'Tạm biệt'
    monkeypatch.setattr(deep_translator, 'GoogleTranslator', lambda **kw: translator)
    result = tr.translate_srt_ai(source, 'vi', 'translate1')
    assert 'Xin chào' in result and 'Tạm biệt' in result
    translator.translate.assert_called_once_with('再见')
    assert config.jobs['translate1']['segment_speakers'] == ['F1', 'M1']


def test_failed_monitor_pipeline_never_enters_history(monkeypatch, tmp_path):
    from services.douyin_monitor import daemon as d
    monkeypatch.setattr(d, 'get_channels', lambda: [{'channel_id': 'test', 'sec_uid': 'x'*40, 'enabled': True}])
    monkeypatch.setattr(d, 'get_downloaded_history', lambda: set())
    monkeypatch.setattr(d, 'update_channel', lambda *args: None)
    monkeypatch.setattr(d, 'fetch_channel_videos', lambda *a, **kw: [{'aweme_id': '123', 'create_time': __import__('time').time()}])
    video = tmp_path/'movie.mp4'
    video.write_bytes(b'video')
    monkeypatch.setattr(d, 'download_master_video', lambda *a: str(video))
    monkeypatch.setattr(d, 'send_telegram_message', MagicMock())
    monkeypatch.setattr(d, 'execute_auto_pipeline', MagicMock(side_effect=RuntimeError('TTS failed')))
    monkeypatch.setattr(d.time, 'sleep', lambda _: None)
    mark = MagicMock()
    monkeypatch.setattr(d, 'mark_as_downloaded', mark)
    d._stop_event.clear()
    result = d._perform_scan()
    assert result['errors'] and 'TTS failed' in result['errors'][0]
    mark.assert_not_called()


def test_delivery_retry_only_sends_existing_artifact(monkeypatch, tmp_path):
    from services.douyin_monitor import pipeline_bridge as bridge
    path = tmp_path/'result.mp4'
    path.write_bytes(b'rendered')
    config.create_job('dy_retry01', status='done', delivery_status='pending', delivery_path=str(path),
                      delivery_attempts=1, delivery_next_attempt=0)
    send = MagicMock(return_value=True)
    monkeypatch.setattr(bridge, 'send_telegram_document', send)
    bridge.retry_pending_deliveries()
    send.assert_called_once()
    assert config.jobs.store.load('dy_retry01')['delivery_status'] == 'delivered'


def test_cache_identity_changes_with_text_voice(monkeypatch, tmp_path):
    from services import srt_to_tts as tts
    paths = []
    async def synth(text, voice, pitch, out_path, cancelled_cb=None):
        paths.append(str(out_path))
        return False
    monkeypatch.setattr(tts, 'synthesize_edge_segment', synth)
    async def no_sleep(_): pass
    monkeypatch.setattr(tts.asyncio, 'sleep', no_sleep)
    seg = tts.parse_srt_data(SRT)
    tts.synthesize_all_segments(seg, 'vi', custom_voice='voiceA', temp_dir=tmp_path)
    tts.synthesize_all_segments(seg, 'vi', custom_voice='voiceA', temp_dir=tmp_path)
    tts.synthesize_all_segments(seg, 'vi', custom_voice='voiceB', temp_dir=tmp_path)
    seg[0]['text'] = 'Khác'
    tts.synthesize_all_segments(seg, 'vi', custom_voice='voiceB', temp_dir=tmp_path)
    assert paths[0] == paths[1] and len(set(paths)) == 3


def test_retry_uses_saved_srt_and_tts_cache(monkeypatch):
    from services import pipeline_orchestrator as p
    folder = config.OUTPUT_FOLDER/'retry001'
    folder.mkdir()
    srt = folder/'vi.srt'
    srt.write_text(SRT, encoding='utf-8')
    config.create_job('retry001', status='partial', srt_files={'vi': {'path': str(srt)}},
        pipeline_config={'source_type':'srt_only', 'steps':{'tts':True}, 'sub_options': {'target_lang':'vi'}})
    dispatch = MagicMock()
    monkeypatch.setattr(p, 'submit', dispatch)
    response = app.test_client().post('/api/pipeline/retry/retry001', json={'tts_options': {'max_speed': 1.8}})
    assert response.status_code == 200
    passed = dispatch.call_args.args[2]
    assert passed['target_srt_content'] == SRT and passed['tts_options']['max_speed'] == 1.8


def test_rendered_frontend_scripts_parse(tmp_path):
    import re
    import shutil
    node = shutil.which('node') or r'C:\Program Files\nodejs\node.exe'
    if not Path(node).is_file():
        pytest.skip('Node unavailable for frontend syntax check')
    html = app.test_client().get('/').get_data(as_text=True)
    for index, body in enumerate(re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>', html, re.S)):
        if not body.strip():
            continue
        script = tmp_path / f'inline-{index}.js'
        script.write_text(body, encoding='utf-8')
        result = subprocess.run([node, '--check', str(script)], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr


def test_old_queued_invocation_cannot_run_after_retry(monkeypatch):
    from concurrent.futures import Future
    from services import job_runner
    queued = []
    class Executor:
        def submit(self, fn):
            queued.append(fn)
            return Future()
    monkeypatch.setattr(job_runner, '_executor', Executor())
    monkeypatch.setattr(job_runner, '_capacity', __import__('threading').BoundedSemaphore(2))
    config.create_job('replaced1', status='queued', _created_at=1)
    worker = MagicMock()
    job_runner.submit(worker, 'replaced1')
    config.create_job('replaced1', status='queued', _created_at=2)
    queued[0]()
    worker.assert_not_called()


def test_dead_owner_restores_active_child_as_interrupted():
    config.create_job('childdead', status='done', _owner_pid=99999999, tts_vi={'status':'generating'})
    restored = JobRegistry().get('childdead')
    assert restored['status'] == 'interrupted'
    assert restored['tts_vi']['status'] == 'interrupted'


def test_cleanup_preserves_review_and_generating_artifacts(monkeypatch):
    for jid, state in [('retain01', 'awaiting_review'), ('retain02', 'generating')]:
        config.create_job(jid, status=state)
        folder = config.OUTPUT_FOLDER/jid
        folder.mkdir()
        (folder/'draft.srt').write_text(SRT, encoding='utf-8')
    config.cleanup_old_files(force=True)
    assert all((config.OUTPUT_FOLDER/jid/'draft.srt').is_file() for jid in ('retain01', 'retain02'))


def test_cancel_upload_after_reload_releases_active_state():
    import routes.api as api
    client = app.test_client()
    jid = client.post('/api/upload/init', json={'filename':'x.mp4','total_size':4,'chunk_size':4}).json['job_id']
    api._chunk_uploads.clear()
    assert client.post('/api/upload/cancel', json={'job_id':jid}).status_code == 200
    assert config.jobs.store.load(jid)['status'] == 'cancelled'
    assert not (config.UPLOAD_FOLDER/jid).exists()


def test_clean_download_url_handles_douyin_share_text():
    from services.downloader import clean_download_url, validate_download_url
    raw = "7.46 J@i.pd :2pm 10/02 Qkc:/ test https://v.douyin.com/7drZYENBiQo/ copy"
    cleaned = clean_download_url(raw)
    assert cleaned == "https://v.douyin.com/7drZYENBiQo/"
    valid, err = validate_download_url(raw)
    assert valid and not err


def test_launcher_check_settles_cancelled_or_dead_jobs(monkeypatch):
    import services.launcher_check as lc
    config.create_job("stale_cancelled", status="processing", cancel=True)
    config.create_job("stale_dead", status="processing", _owner_pid=99999999)
    assert lc.main() == 0
    assert config.jobs.store.load("stale_cancelled")["status"] == "cancelled"
    assert config.jobs.store.load("stale_dead")["status"] == "interrupted"



def test_review_cannot_approve_known_untranslated_lines(monkeypatch):
    from services import pipeline_orchestrator as pipeline
    source = '1\n00:00:00,000 --> 00:00:01,000\n你好\n\n2\n00:00:01,100 --> 00:00:02,000\n再见'
    config.create_job('reviewbad', status='awaiting_review', review_revision=0,
        translation_quality={'en': {'unchanged_segments': [1, 2], 'total_segments': 2}},
        pending_pipeline={'video_path': '', 'target_lang': 'en', 'target_srt_content': source,
                          'target_srt_path': '', 'steps': {'tts': True}, 'clean_opts': {},
                          'tts_opts': {}, 'burn_opts': {}})
    client = app.test_client()
    response = client.post('/api/pipeline/continue/reviewbad', json={'updated_srt': source, 'revision': 0})
    assert response.status_code == 400
    assert 'Còn 2 câu chưa dịch' in response.json['error']
    edited = source.replace('你好', 'Hello').replace('再见', 'Goodbye')
    dispatch = MagicMock()
    monkeypatch.setattr(pipeline, 'submit', dispatch)
    response = client.post('/api/pipeline/continue/reviewbad', json={'updated_srt': edited, 'revision': 0})
    assert response.status_code == 200
    state = config.jobs.store.load('reviewbad')
    assert state['translation_quality']['en']['unchanged_segments'] == []
    assert state['translation_quality']['en']['reviewed_by_user'] is True


def test_status_downgrades_done_job_with_missing_artifact(tmp_path):
    missing = config.OUTPUT_FOLDER / 'artifact1' / 'result.srt'
    config.create_job('artifact1', status='done', progress=100,
                      step_results={'subtitles': {'status': 'done', 'path': str(missing)}})
    response = app.test_client().get('/api/status/artifact1')
    assert response.status_code == 200
    assert response.json['status'] == 'partial'
    assert response.json['step_results']['subtitles']['artifact_missing'] is True
    assert 'path' not in response.json['step_results']['subtitles']
    assert config.jobs.store.load('artifact1')['status'] == 'partial'


def test_channel_fetch_preserves_configured_uifid(monkeypatch):
    from types import SimpleNamespace
    from services.douyin_monitor import crawler
    from dy_apis.douyin_api import DouyinAPI
    auth = SimpleNamespace(cookie={'UIFID': 'configured-browser-identity'})
    monkeypatch.setattr(crawler, 'get_douyin_auth', lambda force_refresh_uifid=False: auth)
    monkeypatch.setattr(DouyinAPI, 'get_user_work_info', staticmethod(lambda *_: {
        'aweme_list': [{'aweme_id': '123', 'aweme_type': 0, 'create_time': 1}], 'has_more': 0
    }))
    works = crawler.fetch_channel_videos('x' * 40, max_count=10)
    assert [work['aweme_id'] for work in works] == ['123']
    assert auth.cookie['UIFID'] == 'configured-browser-identity'



def test_cleanup_uses_longer_retention_for_outputs(monkeypatch):
    import time
    upload = config.UPLOAD_FOLDER / 'old-upload.bin'
    output = config.OUTPUT_FOLDER / 'old-result.bin'
    upload.write_bytes(b'upload')
    output.write_bytes(b'output')
    old = time.time() - 20
    __import__('os').utime(upload, (old, old))
    __import__('os').utime(output, (old, old))
    monkeypatch.setattr(config, 'UPLOAD_FILE_MAX_AGE_SECONDS', 10)
    monkeypatch.setattr(config, 'OUTPUT_FILE_MAX_AGE_SECONDS', 100)
    monkeypatch.setattr(config, '_CLEANUP_INTERVAL', 0)
    monkeypatch.setattr(config, '_last_cleanup_time', 0)
    config.cleanup_old_files()
    assert not upload.exists()
    assert output.exists()



def test_douyin_network_candidates_choose_highest_video_bitrate():
    from services.downloader import _parse_douyin_media_responses, _select_best_douyin_media_url
    import json
    def event(url, mime='video/mp4', status=206, resource_type='Media'):
        return {'message': json.dumps({'message': {'method': 'Network.responseReceived', 'params': {
            'type': resource_type, 'response': {'url': url, 'mimeType': mime, 'status': status}
        }}})}
    logs = [
        event('https://lf.douyinstatic.com/uuu_265.mp4?br=9999'),
        event('https://v.zjcdn.com/video/tos/media-audio-und-mp4a/?br=46'),
        event('https://v.zjcdn.com/video/tos/media-video-avc1/?br=723', resource_type='Fetch'),
        event('https://v.zjcdn.com/video/tos/combined/?br=1861', resource_type='XHR'),
    ]
    candidates = _parse_douyin_media_responses(logs)
    assert len(candidates) == 2
    assert _select_best_douyin_media_url(candidates).endswith('br=1861')


def test_douyin_media_parser_ignores_failed_and_non_video_responses():
    from services.downloader import _parse_douyin_media_responses
    import json
    logs = [
        {'message': json.dumps({'message': {'method': 'Network.responseReceived', 'params': {
            'type': 'Fetch', 'response': {'url': 'https://v.zjcdn.com/video/tos/x?br=1000', 'mimeType': 'text/plain', 'status': 403}
        }}})},
        {'message': json.dumps({'message': {'method': 'Network.responseReceived', 'params': {
            'type': 'Image', 'response': {'url': 'https://p.douyinpic.com/a.jpg', 'mimeType': 'image/jpeg', 'status': 200}
        }}})},
    ]
    assert _parse_douyin_media_responses(logs) == []



def test_douyin_cookie_error_is_actionable_without_browser_fallback(monkeypatch):
    import config
    from services import downloader
    config.create_job('cookieerr1')
    monkeypatch.setenv('DOUYIN_BROWSER_FALLBACK', '0')
    monkeypatch.setattr(downloader, 'download_douyin_master', MagicMock(side_effect=RuntimeError('Blocked by ArgusSecurityPlugin Sign Invalid')))
    browser = MagicMock()
    monkeypatch.setattr(downloader, 'download_chinese_video', browser)
    with pytest.raises(RuntimeError, match='DY_COOKIES'):
        downloader.download_from_url('cookieerr1', 'https://v.douyin.com/test123/')
    browser.assert_not_called()


def test_douyin_browser_fallback_requires_explicit_opt_in(monkeypatch):
    import config
    from services import downloader
    config.create_job('cookieerr2')
    monkeypatch.setenv('DOUYIN_BROWSER_FALLBACK', '1')
    monkeypatch.setattr(downloader, 'download_douyin_master', MagicMock(side_effect=RuntimeError('Blocked by ArgusSecurityPlugin Sign Invalid')))
    monkeypatch.setattr(downloader, 'download_chinese_video', MagicMock(return_value='fallback.mp4'))
    assert downloader.download_from_url('cookieerr2', 'https://v.douyin.com/test123/') == 'fallback.mp4'
