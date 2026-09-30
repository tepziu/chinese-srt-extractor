"""Recover a pipeline that rendered its final video but lost terminal status."""
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from config import create_job, jobs


def _stale_render(tmp_path, job_id='repair_render_123'):
    folder = tmp_path / job_id
    folder.mkdir()
    video = folder / 'burned_en.mp4'
    video.write_bytes(b'encoded fixture')
    old = time.time() - 180
    os.utime(video, (old, old))
    job = create_job(
        job_id, status='processing', progress=85, active_step='step_5_burn',
        pipeline_config={'steps': {'extract_sub': True, 'tts': True, 'clean_video': False,
                                   'burn_sub': True}, 'sub_options': {'target_lang': 'en'}},
        step_results={'subtitles': {'status': 'done'}, 'tts': {'status': 'done'}},
        artifacts={}, burn_en={'status': 'done', 'progress': 100, 'path': str(video),
                               'size': video.stat().st_size},
    )
    return job, video


def _probe(*_args, **_kwargs):
    from subprocess import CompletedProcess
    return CompletedProcess([], 0, stdout='{"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "1.0"}}')


def test_recover_verified_stale_final_video(tmp_path):
    from services.pipeline_completion import reconcile_final_render
    job, video = _stale_render(tmp_path)
    try:
        with patch('services.pipeline_completion.subprocess.run', side_effect=_probe):
            assert reconcile_final_render(job, 'repair_render_123', tmp_path)
        assert job['status'] == 'done' and job['progress'] == 100
        assert job['artifacts']['final_video'] == str(video)
        assert job['step_results']['burn']['status'] == 'done'
        assert job['completion_recovered'] is True
        assert not reconcile_final_render(job, 'repair_render_123', tmp_path)
    finally:
        jobs.pop('repair_render_123', None)


@pytest.mark.parametrize('variant', ['recent', 'missing', 'outside', 'cancelled', 'no_subtitles'])
def test_do_not_recover_without_durable_evidence(tmp_path, variant):
    from services.pipeline_completion import reconcile_final_render
    job, video = _stale_render(tmp_path)
    try:
        if variant == 'recent':
            os.utime(video, None)
        elif variant == 'missing':
            video.unlink()
        elif variant == 'outside':
            job['burn_en']['path'] = str(tmp_path / 'elsewhere.mp4')
        elif variant == 'cancelled':
            job['cancel'] = True
        else:
            job['step_results'].pop('subtitles')
        with patch('services.pipeline_completion.subprocess.run', side_effect=_probe) as probe:
            assert not reconcile_final_render(job, 'repair_render_123', tmp_path)
            probe.assert_not_called()
        assert job['status'] == 'processing' and job['progress'] == 85
    finally:
        jobs.pop('repair_render_123', None)


def test_invalid_probe_keeps_status_and_output(tmp_path):
    from services.pipeline_completion import reconcile_final_render
    job, video = _stale_render(tmp_path)
    try:
        with patch('services.pipeline_completion.subprocess.run', side_effect=_probe) as probe:
            probe.side_effect = lambda *_a, **_k: type('Result', (), {
                'returncode': 1, 'stdout': '', 'stderr': 'invalid'})()
            assert not reconcile_final_render(job, 'repair_render_123', tmp_path)
        assert video.exists() and job['status'] == 'processing'
    finally:
        jobs.pop('repair_render_123', None)


def test_status_poll_repairs_only_verified_stale_pipeline(tmp_path, monkeypatch):
    from app import app
    import routes.api as api
    monkeypatch.setattr(api, 'OUTPUT_FOLDER', tmp_path)
    job, video = _stale_render(tmp_path)
    try:
        with patch('services.pipeline_completion.subprocess.run', side_effect=_probe):
            response = app.test_client().get('/api/status/repair_render_123')
        assert response.status_code == 200
        assert response.json['status'] == 'done'
        assert response.json['artifacts']['final_video'] == str(video)
    finally:
        jobs.pop('repair_render_123', None)


def test_status_poll_recovers_render_after_worker_process_exits(tmp_path, monkeypatch):
    from app import app
    import routes.api as api
    monkeypatch.setattr(api, 'OUTPUT_FOLDER', tmp_path)
    job, video = _stale_render(tmp_path, 'repair_dead_owner_123')
    try:
        job['_owner_pid'] = 99999999
        with patch('services.pipeline_completion.subprocess.run', side_effect=_probe):
            response = app.test_client().get('/api/status/repair_dead_owner_123')
        assert response.status_code == 200
        assert response.json['status'] == 'done'
        assert response.json['completion_recovered'] is True
        assert response.json['artifacts']['final_video'] == str(video)
    finally:
        jobs.pop('repair_dead_owner_123', None)


def test_launcher_preflight_reconciles_stale_render(tmp_path, monkeypatch):
    from services import launcher_check
    from services.runtime_state import JobStore
    job, _ = _stale_render(tmp_path, 'repair_preflight_123')
    monkeypatch.setattr(launcher_check, 'OUTPUT_FOLDER', tmp_path)
    try:
        with patch('services.pipeline_completion.subprocess.run', side_effect=_probe):
            assert launcher_check.main() == 0
        assert JobStore().load('repair_preflight_123')['status'] == 'done'
    finally:
        jobs.pop('repair_preflight_123', None)
