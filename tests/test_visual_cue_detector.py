import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_parse_contact_response_requires_all_frame_ids():
    from services.visual_cue_detector import parse_contact_response
    raw = '```json\n' + json.dumps({'frames': [
        {'frame_id': 'F0001', 'text': '你好', 'kind': 'dialogue'},
        {'frame_id': 'F0002', 'text': '', 'kind': 'none'},
    ]}, ensure_ascii=False) + '\n```'
    frames = parse_contact_response(raw, ['F0001', 'F0002'])
    assert frames['F0001']['text'] == '你好'
    with pytest.raises(ValueError, match='thiếu frame_id'):
        parse_contact_response(raw, ['F0001', 'F0002', 'F0003'])


def test_merge_ocr_states_deduplicates_same_subtitle():
    from services.visual_cue_detector import merge_ocr_states
    states = [
        {'start_ms': 0, 'end_ms': 500, 'text': '安总监真的来了', 'kind': 'dialogue'},
        {'start_ms': 500, 'end_ms': 1000, 'text': '安总监真的来了', 'kind': 'dialogue'},
        {'start_ms': 1000, 'end_ms': 1500, 'text': '穿成这样', 'kind': 'dialogue'},
        {'start_ms': 1500, 'end_ms': 2000, 'text': '', 'kind': 'none'},
    ]
    cues = merge_ocr_states(states)
    assert len(cues) == 2
    assert cues[0] == {'start': 0.0, 'end': 1.0, 'text': '安总监真的来了'}
    assert cues[1]['text'] == '穿成这样'


def test_merge_ocr_states_accepts_small_ocr_variation():
    from services.visual_cue_detector import merge_ocr_states
    states = [
        {'start_ms': 0, 'end_ms': 600, 'text': '安总监真的来了', 'kind': 'dialogue'},
        {'start_ms': 600, 'end_ms': 1200, 'text': '安总监真的来啦', 'kind': 'dialogue'},
    ]
    cues = merge_ocr_states(states, similarity_threshold=0.85)
    assert len(cues) == 1
    assert cues[0]['end'] == 1.2


def test_contact_sheet_builds_label_mapping(tmp_path):
    from services.visual_cue_detector import build_contact_sheet
    samples = []
    for i in range(3):
        image = np.zeros((80, 320, 3), dtype=np.uint8)
        cv2.putText(image, f'text{i}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        samples.append({'frame_id': f'F{i+1:04d}', 'image': image})
    sheet, ids = build_contact_sheet(samples, columns=2)
    assert ids == ['F0001', 'F0002', 'F0003']
    assert sheet.shape[0] > 80 and sheet.shape[1] >= 640


def test_candidate_intervals_add_guard_points():
    from services.visual_cue_detector import candidate_intervals
    intervals = candidate_intervals([0.8, 1.8, 5.0], duration=6.0, guard_gap=1.2)
    assert intervals[0]['start_ms'] == 0
    assert intervals[-1]['end_ms'] == 6000
    assert all(item['end_ms'] > item['start_ms'] for item in intervals)
    assert len(intervals) > 4
