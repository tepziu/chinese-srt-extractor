import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SOURCE = """1
00:00:00,000 --> 00:00:01,000
那刘玥还在

2
00:00:01,000 --> 00:00:02,100
沈总办公室呢

3
00:00:03,000 --> 00:00:04,000
可惜

4
00:00:04,000 --> 00:00:05,500
你们要失望了
"""


def test_contract_builds_deterministic_display_timestamps():
    from services.hybrid_subtitles import cues_from_srt, validate_fusion_groups, groups_to_srt
    cues = cues_from_srt(SOURCE)
    groups = [
        {'source_ids': [1, 2], 'translation': "Liu Yue is still in CEO Shen's office.", 'speaker': 'F1'},
        {'source_ids': [3, 4], 'translation': "Too bad. You're going to be disappointed.", 'speaker': 'F1'},
    ]
    normalized = validate_fusion_groups(groups, cues)
    display = groups_to_srt(normalized)
    assert '00:00:00,000 --> 00:00:02,100' in display
    assert '00:00:03,000 --> 00:00:05,500' in display
    assert "Liu Yue is still in CEO Shen's office." in display


@pytest.mark.parametrize('groups,match', [
    ([{'source_ids': [1, 3], 'translation': 'bad'}], 'liên tiếp'),
    ([{'source_ids': [1, 2], 'translation': 'a'}, {'source_ids': [2, 3, 4], 'translation': 'b'}], 'đúng một lần'),
    ([{'source_ids': [1, 2], 'translation': 'a'}], 'thiếu'),
])
def test_contract_rejects_invalid_coverage(groups, match):
    from services.hybrid_subtitles import cues_from_srt, validate_fusion_groups
    with pytest.raises(ValueError, match=match):
        validate_fusion_groups(groups, cues_from_srt(SOURCE))


def test_parse_fenced_json_response():
    from services.hybrid_subtitles import parse_fusion_response
    raw = '```json\n' + json.dumps({'groups': [{'source_ids': [1], 'translation': 'Hello', 'speaker': 'F1'}]}) + '\n```'
    assert parse_fusion_response(raw)[0]['translation'] == 'Hello'


def test_tts_groups_merge_adjacent_display_groups_only_when_safe():
    from services.hybrid_subtitles import cues_from_srt, validate_fusion_groups, build_tts_groups
    cues = cues_from_srt(SOURCE)
    display = validate_fusion_groups([
        {'source_ids': [1], 'translation': 'Liu Yue is still', 'speaker': 'F1'},
        {'source_ids': [2], 'translation': "in CEO Shen's office.", 'speaker': 'F1'},
        {'source_ids': [3], 'translation': 'Too bad.', 'speaker': 'M1'},
        {'source_ids': [4], 'translation': "You're going to be disappointed.", 'speaker': 'M1'},
    ], cues)
    tts = build_tts_groups(display, max_duration_ms=6000, max_gap_ms=250)
    assert len(tts) == 3
    assert tts[0]['translation'] == "Liu Yue is still in CEO Shen's office."
    assert tts[1]['source_ids'] == [3]
    assert tts[2]['source_ids'] == [4]



def test_sentence_split_keeps_one_complete_sentence_per_subtitle():
    from services.hybrid_subtitles import split_text_sentences

    assert split_text_sentences("First sentence. Second sentence!") == [
        "First sentence.", "Second sentence!"
    ]
    assert split_text_sentences("A complete thought without punctuation") == [
        "A complete thought without punctuation"
    ]


def test_tts_groups_do_not_merge_two_complete_sentences():
    from services.hybrid_subtitles import cues_from_srt, validate_fusion_groups, build_tts_groups

    display = validate_fusion_groups([
        {'source_ids': [1, 2], 'translation': 'Too bad.', 'speaker': 'F1'},
        {'source_ids': [3, 4], 'translation': "You're going to be disappointed.", 'speaker': 'F1'},
    ], cues_from_srt(SOURCE))
    tts = build_tts_groups(display, max_duration_ms=6000, max_gap_ms=250)
    assert len(tts) == 2


def test_semantic_fusion_uses_model_to_split_long_sentence_into_meaningful_cues(monkeypatch):
    import config
    from services import hybrid_subtitles as hybrid

    config.create_job('hybridsegment')
    long_text = 'If you often struggle to time your lane changes, this video will clearly explain the core rules of safe lane changing.'
    response = {'groups': [
        {'source_ids': [1, 2], 'translation': long_text, 'speaker': 'M1'},
        {'source_ids': [3, 4], 'translation': 'One complete thought.', 'speaker': 'M1'},
    ]}
    monkeypatch.setattr(hybrid, '_call_fusion_model', lambda *args, **kwargs: json.dumps(response))
    monkeypatch.setattr(
        hybrid,
        '_call_segmentation_model',
        lambda *args, **kwargs: json.dumps({'items': [
            {'index': 1, 'segments': [
                {'text': 'If you often struggle to time your lane changes'},
                {'text': 'this video will clearly explain the core rules of safe lane changing.'},
            ]}
        ]}),
    )

    result = hybrid.semantic_fuse_translate(SOURCE, 'en', 'hybridsegment', translation_mode='driving')

    assert [group['translation'] for group in result['groups']] == [
        'If you often struggle to time your lane changes',
        'this video will clearly explain the core rules of safe lane changing.',
        'One complete thought.',
    ]
    assert result['groups'][0]['semantic_split'] is True
    assert len(result['tts_groups']) == 2
    assert result['tts_groups'][0]['translation'] == long_text


def test_semantic_fusion_splits_model_output_with_multiple_sentences(monkeypatch):
    import config
    from services import hybrid_subtitles as hybrid

    config.create_job('hybridsplit')
    response = {'groups': [
        {'source_ids': [1, 2], 'translation': 'First sentence. Second sentence!', 'speaker': 'F1'},
        {'source_ids': [3, 4], 'translation': 'One complete thought.', 'speaker': 'F1'},
    ]}
    monkeypatch.setattr(hybrid, '_call_fusion_model', lambda *args, **kwargs: json.dumps(response))

    result = hybrid.semantic_fuse_translate(SOURCE, 'en', 'hybridsplit', translation_mode='movie')

    assert [group['translation'] for group in result['groups']] == [
        'First sentence.', 'Second sentence!', 'One complete thought.'
    ]
    assert len(result['tts_groups']) == 3


def test_semantic_fusion_falls_back_when_provider_contract_invalid(monkeypatch):
    import config
    from services import hybrid_subtitles as hybrid
    config.create_job('hybrid01')
    monkeypatch.setattr(hybrid, '_call_fusion_model', lambda *args, **kwargs: '{"groups": [{"source_ids": [1], "translation": "Only one"}]}')
    fallback = MagicMock(return_value="""1
00:00:00,000 --> 00:00:02,100
Liu Yue is still in CEO Shen's office.

2
00:00:03,000 --> 00:00:05,500
Too bad. You'll be disappointed.
""")
    monkeypatch.setattr(hybrid, '_translate_grouped_fallback', fallback)
    result = hybrid.semantic_fuse_translate(SOURCE, 'en', 'hybrid01', translation_mode='movie')
    assert result['used_fallback'] is True
    assert result['display_srt']
    fallback.assert_called_once()


def test_semantic_fusion_records_display_and_tts_contract(monkeypatch):
    import config
    from services import hybrid_subtitles as hybrid
    config.create_job('hybrid02')
    response = {'groups': [
        {'source_ids': [1, 2], 'translation': "Liu Yue is still in CEO Shen's office.", 'speaker': 'F1'},
        {'source_ids': [3, 4], 'translation': "Too bad. You're going to be disappointed.", 'speaker': 'F1'},
    ]}
    monkeypatch.setattr(hybrid, '_call_fusion_model', lambda *args, **kwargs: json.dumps(response))
    result = hybrid.semantic_fuse_translate(SOURCE, 'en', 'hybrid02', translation_mode='movie')
    assert result['used_fallback'] is False
    assert len(result['groups']) == 3
    assert result['display_srt']
    assert result['tts_srt']
    assert config.jobs['hybrid02']['semantic_fusion']['source_segments'] == 4
    assert config.jobs['hybrid02']['semantic_fusion']['display_segments'] == 3



def test_display_srt_can_regenerate_tts_track_after_review():
    from services.hybrid_subtitles import display_srt_to_tts_srt
    display = """1
00:00:00,000 --> 00:00:01,000
Liu Yue is still

2
00:00:01,000 --> 00:00:02,100
in CEO Shen's office.
"""
    tts_srt, speakers = display_srt_to_tts_srt(display, ['F1', 'F1'])
    assert "Liu Yue is still in CEO Shen's office." in tts_srt
    assert speakers == ['F1']



def test_pipeline_options_accept_hybrid_engine():
    from services.pipeline_options import validate_pipeline_options
    options = validate_pipeline_options({'source_type': 'upload', 'steps': {'extract_sub': True},
        'sub_options': {'engine': 'hybrid', 'target_lang': 'en'}})
    assert options['sub_options']['engine'] == 'hybrid'


def test_pipeline_hybrid_uses_vision_primary_and_whisper_context(monkeypatch, tmp_path):
    import config
    from services import pipeline_orchestrator as pipeline
    from services import hybrid_subtitles as hybrid
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'video')
    jid = 'hybridpipe'
    config.create_job(jid, status='processing', original_name='episode.mp4', video_path=str(video),
                      artifacts={}, step_results={}, srt_files={})
    whisper = SOURCE.replace('那刘玥还在', 'audio mistake')
    monkeypatch.setattr(pipeline, '_extract_whisper_source', lambda *args: (whisper, str(tmp_path/'whisper.srt')))
    monkeypatch.setattr(pipeline, '_extract_gemini_source', lambda *args: SOURCE)
    captured = {}
    def fuse(source, target, job_id, **kwargs):
        captured.update(source=source, context=kwargs.get('whisper_context'))
        config.jobs[job_id]['semantic_fusion'] = {'used_fallback': False}
        return {'display_srt': SOURCE.replace('那刘玥还在', 'Vision translation'),
                'tts_srt': SOURCE.replace('那刘玥还在', 'Vision translation'),
                'groups': [], 'tts_groups': [{}, {}], 'used_fallback': False, 'warning': None}
    monkeypatch.setattr(hybrid, 'semantic_fuse_translate', fuse)
    pipeline._pipeline_worker.__wrapped__(jid, {'source_type': 'upload', 'video_path': str(video),
        'steps': {'download': False, 'extract_sub': True, 'clean_video': False, 'tts': False, 'burn_sub': False},
        'sub_options': {'engine': 'hybrid', 'target_lang': 'en', 'style': 'movie', 'translate_method': 'ai'},
        'clean_options': {}, 'tts_options': {}, 'burn_options': {}})
    state = config.jobs[jid]
    assert captured['source'] == SOURCE
    assert captured['context'] == whisper
    assert state['status'] == 'done'
    assert state['tts_script']['segments'] == 2
    assert Path(state['srt_files']['en']['path']).is_file()


def test_pipeline_hybrid_falls_back_to_whisper_when_vision_fails(monkeypatch, tmp_path):
    import config
    from services import pipeline_orchestrator as pipeline
    from services import hybrid_subtitles as hybrid
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'video')
    jid = 'hybridfall'
    config.create_job(jid, status='processing', original_name='episode.mp4', video_path=str(video),
                      artifacts={}, step_results={}, srt_files={})
    monkeypatch.setattr(pipeline, '_extract_whisper_source', lambda *args: (SOURCE, str(tmp_path/'whisper.srt')))
    monkeypatch.setattr(pipeline, '_extract_gemini_source', MagicMock(side_effect=RuntimeError('vision unavailable')))
    captured = {}
    def fuse(source, target, job_id, **kwargs):
        captured['source'] = source
        config.jobs[job_id]['semantic_fusion'] = {'used_fallback': False}
        return {'display_srt': SOURCE, 'tts_srt': SOURCE, 'groups': [], 'tts_groups': [],
                'used_fallback': False, 'warning': None}
    monkeypatch.setattr(hybrid, 'semantic_fuse_translate', fuse)
    pipeline._pipeline_worker.__wrapped__(jid, {'source_type': 'upload', 'video_path': str(video),
        'steps': {'download': False, 'extract_sub': True, 'clean_video': False, 'tts': False, 'burn_sub': False},
        'sub_options': {'engine': 'hybrid', 'target_lang': 'en', 'style': 'movie', 'translate_method': 'ai'},
        'clean_options': {}, 'tts_options': {}, 'burn_options': {}})
    assert captured['source'] == SOURCE
    assert config.jobs[jid]['semantic_fusion']['vision_fallback'] is True
    assert 'vision' in config.jobs[jid]['subtitle_source_warnings']



def test_contract_rejects_merge_across_large_pause():
    from services.hybrid_subtitles import cues_from_srt, validate_fusion_groups
    with pytest.raises(ValueError, match='khoảng nghỉ lớn'):
        validate_fusion_groups([{'source_ids': [1,2,3,4], 'translation': 'One unsafe group'}], cues_from_srt(SOURCE))


def test_quality_gate_flags_han_residue_and_fragments():
    from services.hybrid_subtitles import evaluate_fusion_quality
    groups = [{'translation': 'Still 在办公室...', 'start_ms': 0, 'end_ms': 2000}]
    quality = evaluate_fusion_quality(groups, 'en')
    assert quality['needs_review'] is True
    assert {item['type'] for item in quality['severe']} == {'sentence_fragment', 'han_residue'}

