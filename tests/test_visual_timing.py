"""Regression tests for conservative visual boundary refinement (no OCR/model calls)."""
import cv2
import numpy as np

from services.srt_utils import parse_srt_timing
from services.visual_timing import refine_visual_timing


def _video(path, visible, frames=40, fps=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 180))
    assert writer.isOpened()
    for frame_id in range(frames):
        image = np.full((180, 320, 3), 35, dtype=np.uint8)
        image[:60] = (frame_id * 11) % 130  # Changing scenery outside subtitle ROI.
        if frame_id in visible:
            image[120:142, 80:240] = (255, 255, 255)
        writer.write(image)
    writer.release()


def _recognize(frame):
    return ['你好'] if np.mean(frame[120:142, 80:240]) > 180 else []


def test_refines_start_and_first_absent_frame(tmp_path):
    video = tmp_path / 'source.mp4'
    _video(video, set(range(7, 21)))
    source = '1\n00:00:00,400 --> 00:00:00,680\n你好\n'
    result, report = refine_visual_timing(str(video), source, recognizer=_recognize)
    assert parse_srt_timing(result) == [(280, 840, '你好')]
    assert report['refined'] == 1
    assert report['cues'][0]['start_frame'] == 7
    assert report['cues'][0]['end_frame_exclusive'] == 21


def test_refuses_to_change_when_text_is_not_visual_or_too_short(tmp_path):
    video = tmp_path / 'source.mp4'
    _video(video, set(range(7, 21)))
    source = '1\n00:00:00,400 --> 00:00:00,680\n完全不同\n'
    result, report = refine_visual_timing(str(video), source, recognizer=_recognize)
    assert result == source
    assert report['refined'] == 0
    assert report['cues'][0]['status'] == 'unchanged_no_anchor'


def test_no_visual_transition_preserves_original(tmp_path):
    video = tmp_path / 'source.mp4'
    _video(video, set(range(40)))
    source = '1\n00:00:00,400 --> 00:00:00,680\n你好\n'
    result, report = refine_visual_timing(str(video), source, recognizer=_recognize)
    assert result == source
    assert report['cues'][0]['status'] == 'unchanged_unverified_boundary'


def test_cancellation_is_not_misreported_as_a_fallback(tmp_path):
    import pytest
    video = tmp_path / 'source.mp4'
    _video(video, set(range(7, 21)))
    source = '1\n00:00:00,400 --> 00:00:00,680\n你好\n'
    with pytest.raises(RuntimeError, match='hủy'):
        refine_visual_timing(str(video), source, recognizer=_recognize, cancelled=lambda: True)


def test_refine_and_store_preserves_original_and_writes_evidence(tmp_path, monkeypatch):
    from services import visual_timing
    before = '1\n00:00:00,400 --> 00:00:00,680\n你好\n'
    after = '1\n00:00:00,280 --> 00:00:00,840\n你好\n'
    report = {'method': 'test', 'refined': 1, 'unchanged': 0,
              'timestamp_basis': 'decoder_msec', 'cues': []}
    monkeypatch.setattr(visual_timing, 'refine_visual_timing', lambda *args, **kwargs: (after, report))
    job = {'artifacts': {}}
    result = visual_timing.refine_and_store(job, 'unused.mp4', before, tmp_path)
    assert result == after
    assert (tmp_path / 'hardsub_zh_original_timing.srt').read_text(encoding='utf-8') == before
    assert (tmp_path / 'hardsub_visual_timing.json').is_file()
    assert job['visual_timing']['refined'] == 1


def test_refinement_error_keeps_existing_pipeline(tmp_path, monkeypatch):
    from services import visual_timing
    before = '1\n00:00:00,400 --> 00:00:00,680\n你好\n'

    def failure(*args, **kwargs):
        raise RuntimeError('OCR offline')

    monkeypatch.setattr(visual_timing, 'refine_visual_timing', failure)
    job = {}
    assert visual_timing.refine_and_store(job, 'unused.mp4', before, tmp_path) == before
    assert 'OCR offline' in job['subtitle_source_warnings']['visual_timing']


def test_pipeline_clean_uses_visual_source_instead_of_translation(tmp_path, monkeypatch):
    from config import create_job, jobs
    from services import pipeline_orchestrator as pipeline
    from services.video import clean_pipeline
    source = '1\n00:00:00,280 --> 00:00:00,840\n你好\n'
    display = '1\n00:00:00,400 --> 00:00:00,680\nXin chào\n'
    job_id = 'test_visual_clean_123'
    create_job(job_id, source_visual_srt=source, artifacts={}, step_results={})
    monkeypatch.setattr(pipeline, 'OUTPUT_FOLDER', tmp_path)
    seen = {}

    def fake_region(*args, **kwargs):
        seen['region_srt'] = kwargs['srt_content']
        return {'x_ratio': .1, 'y_ratio': .7, 'w_ratio': .8, 'h_ratio': .2}

    def fake_clean(**kwargs):
        seen['clean_srt'] = kwargs['srt_content']
        from pathlib import Path
        Path(kwargs['output_path']).write_bytes(b'encoded video placeholder')
        return {'status': 'done'}

    monkeypatch.setattr('services.burn_sub.detect_hardsub_region', fake_region)
    monkeypatch.setattr(clean_pipeline, 'clean_video_pipeline', fake_clean)
    try:
        pipeline._execute_remaining_steps(
            job_id, str(tmp_path / 'source.mp4'), 'vi', display, '',
            {'clean_video': True, 'tts': False, 'burn_sub': False},
            {'engine': 'opencv'}, {}, {})
        assert seen == {'region_srt': source, 'clean_srt': source}
    finally:
        jobs.pop(job_id, None)


def test_burn_keeps_translation_timing_separate_from_clean_timing(monkeypatch):
    from unittest.mock import patch
    from services.burn_sub import burn_sub_video
    from config import create_job, jobs
    rendered = '1\n00:00:00,400 --> 00:00:00,680\nXin chào\n'
    original = '1\n00:00:00,280 --> 00:00:00,840\n你好\n'
    job_id = 'test_burn_source_timing_123'
    create_job(job_id, video_path='test_nvenc.mp4', video_file={'path': 'test_nvenc.mp4'},
               original_name='test_nvenc.mp4')
    try:
        with patch('services.burn_sub._srt_to_ass', return_value='[Script Info]') as ass, \
             patch('services.burn_sub.detect_hardsub_region', return_value={
                 'x_ratio': .1, 'y_ratio': .7, 'w_ratio': .8, 'h_ratio': .2}), \
             patch('services.video.clean_pipeline.clean_video_pipeline', return_value={'status': 'done'}) as clean:
            burn_sub_video(job_id=job_id, lang='vi', srt_content=rendered,
                           clean_timing_srt=original, render_mode='inpaint_burn')
            assert ass.call_args.args[0] == rendered
            assert clean.call_args.kwargs['srt_content'] == original
    finally:
        jobs.pop(job_id, None)


def test_pipeline_opt_in_refines_source_before_translation(tmp_path, monkeypatch):
    from unittest.mock import patch
    from config import create_job, jobs
    from services import pipeline_orchestrator as pipeline
    source = '1\n00:00:00,400 --> 00:00:00,680\n你好\n'
    adjusted = '1\n00:00:00,280 --> 00:00:00,840\n你好\n'
    translation = '1\n00:00:00,280 --> 00:00:00,840\nXin chào\n'
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'not decoded: refinement is mocked')
    job_id = 'test_refine_pipeline_123'
    create_job(job_id, status='queued', video_path=str(video), srt_files={},
               artifacts={}, step_results={}, original_name=video.name)
    monkeypatch.setattr(pipeline, 'OUTPUT_FOLDER', tmp_path)
    options = {'source_type': 'upload', 'trim_intro': 'off',
               'steps': {'extract_sub': True, 'clean_video': False, 'tts': False, 'burn_sub': False},
               'sub_options': {'engine': 'gemini', 'target_lang': 'vi',
                               'translate_method': 'google', 'visual_timing': True}}
    try:
        with patch.object(pipeline, '_extract_gemini_source', return_value=source), \
             patch('services.visual_timing.refine_and_store', return_value=adjusted) as refine, \
             patch('services.translation.translate_srt', return_value=translation) as translate, \
             patch.object(pipeline, '_execute_remaining_steps') as rest:
            pipeline._pipeline_worker(job_id, options)
        assert jobs[job_id]['source_visual_srt'] == adjusted
        assert (tmp_path / job_id / 'gemini_zh.srt').read_text(encoding='utf-8') == adjusted
        assert refine.call_args.args[1:3] == (str(video), source)
        assert translate.call_args.args[0] == adjusted
        assert rest.call_count == 1
    finally:
        jobs.pop(job_id, None)


def test_visual_timing_option_defaults_off():
    from services.pipeline_options import validate_pipeline_options
    options = {'steps': {'extract_sub': True}, 'sub_options': {'engine': 'gemini'}}
    assert validate_pipeline_options(options)['sub_options']['visual_timing'] is False
    assert validate_pipeline_options(options)['clean_options']['visual_guard'] is True
    options['sub_options']['visual_timing'] = 'true'
    assert validate_pipeline_options(options)['sub_options']['visual_timing'] is True


def test_low_confidence_may_only_anchor_exact_long_text():
    from services.visual_timing import _match_rows
    assert _match_rows('你好世界', [{'text':'你好世界','confidence':.07}])[0] is True
    assert _match_rows('你好世界', [{'text':'你好世间','confidence':.07}])[0] is False
    assert _match_rows('你好', [{'text':'你好','confidence':.07}])[0] is False


def test_verified_start_is_not_discarded_when_end_is_unverified(tmp_path):
    video=tmp_path/'partial.mp4'
    _video(video,set(range(5,40)))
    source='1\n00:00:00,400 --> 00:00:00,680\n你好\n'
    result,report=refine_visual_timing(str(video),source,recognizer=_recognize,max_shift=.3)
    assert parse_srt_timing(result)==[(200,680,'你好')]
    assert report['cues'][0]['start_verified'] is True
    assert report['cues'][0]['end_verified'] is False
    assert report['cues'][0]['status']=='refined_start_only'


def test_source_timeline_is_kept_even_when_refinement_is_off(tmp_path, monkeypatch):
    from unittest.mock import patch
    from config import create_job,jobs
    from services import pipeline_orchestrator as pipeline
    source='1\n00:00:00,400 --> 00:00:00,680\n你好\n'
    translated='1\n00:00:00,800 --> 00:00:01,200\nHello\n'
    video=tmp_path/'source.mp4';video.write_bytes(b'not decoded')
    jid='test_keep_source_timing_123'
    create_job(jid,status='queued',video_path=str(video),srt_files={},artifacts={},step_results={},original_name='source.mp4')
    monkeypatch.setattr(pipeline,'OUTPUT_FOLDER',tmp_path)
    options={'source_type':'upload','trim_intro':'off',
             'steps':{'extract_sub':True,'tts':False,'burn_sub':False},
             'sub_options':{'engine':'gemini','target_lang':'en','translate_method':'google','visual_timing':False}}
    try:
        with patch.object(pipeline,'_extract_gemini_source',return_value=source), \
             patch('services.visual_timing.refine_and_store') as refine, \
             patch('services.translation.translate_srt',return_value=translated), \
             patch.object(pipeline,'_execute_remaining_steps'):
            pipeline._pipeline_worker(jid,options)
        refine.assert_not_called()
        assert jobs[jid]['source_visual_srt']==source
        assert jobs[jid]['source_visual_srt'] != translated
    finally:jobs.pop(jid,None)


def test_numpy_boolean_matcher_does_not_reverse_boundary_search(tmp_path, monkeypatch):
    from services import visual_timing as timing
    video=tmp_path/'numpy_match.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25,(320,240))
    for i in range(40):
        image=np.full((240,320,3),75,np.uint8)
        if 5<=i<=27:
            for j,char in enumerate('OLD'):
                cv2.putText(image,char,(80+j*26,184),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),6)
                cv2.putText(image,char,(80+j*26,184),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        writer.write(image)
    writer.release()
    real=timing.reference_matches
    monkeypatch.setattr(timing,'reference_matches',lambda ref,img:np.bool_(real(ref,img)))
    def recognize(image):
        return ['你好世界'] if np.count_nonzero(image[160:190,70:165]>180)>50 else []
    source='1\n00:00:00,800 --> 00:00:01,000\n你好世界\n'
    result,report=timing.refine_visual_timing(str(video),source,recognizer=recognize)
    assert parse_srt_timing(result)==[(200,1120,'你好世界')]
    assert report['cues'][0]['start_verified'] and report['cues'][0]['end_verified']


def test_retry_keeps_original_timing_without_resubmitting_recognition(tmp_path):
    from unittest.mock import patch
    from app import app
    from config import create_job,jobs
    source='1\n00:00:00,200 --> 00:00:00,800\n你好世界\n'
    target='1\n00:00:00,500 --> 00:00:01,000\nHello\n'
    target_path=tmp_path/'target_en.srt';target_path.write_text(target,encoding='utf-8')
    video=tmp_path/'source.mp4';video.write_bytes(b'not decoded')
    jid='test_retry_source_timing_123'
    create_job(jid,status='error',source_visual_srt=source,video_path=str(video),
               pipeline_config={'steps':{'extract_sub':True,'burn_sub':True},'sub_options':{'target_lang':'en'}},
               srt_files={'en':{'path':str(target_path)}})
    try:
        with patch('services.pipeline_orchestrator.start_pipeline_job',return_value={'status':'queued'}) as start:
            response=app.test_client().post('/api/pipeline/retry/'+jid,json={})
        assert response.status_code==200
        opts=start.call_args.args[1]
        assert opts['steps']['extract_sub'] is False
        assert opts['zh_srt_content']==source and opts['target_srt_content']==target
    finally:jobs.pop(jid,None)
