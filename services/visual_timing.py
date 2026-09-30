"""Optional source-content anchoring + inexpensive per-frame visual timing.

Every cue/reference belongs to the current video. Unknown cues, unsupported
styles and unverified boundaries stay explicitly unverified; no speech timestamp
or low-confidence fuzzy OCR is promoted to visual proof.
"""
from __future__ import annotations

from contextlib import nullcontext
from difflib import SequenceMatcher
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

import cv2

from services.srt_utils import generate_srt, parse_srt_timing, validate_srt
from services.video.timing_guard import glyph_reference, reference_matches


def _normalized(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text or "").casefold()


def _same_text(expected: str, observed: str) -> bool:
    expected, observed = _normalized(expected), _normalized(observed)
    if len(expected) < 2 or len(observed) < 2:
        return False
    threshold = .90 if min(len(expected), len(observed)) <= 4 else .84
    return SequenceMatcher(None, expected, observed).ratio() >= threshold


def _match_rows(expected: str, rows: list) -> tuple[bool, list]:
    """Exact, sufficiently long low-score OCR may anchor; weak fuzzy OCR cannot."""
    if isinstance(rows, str):
        rows = [rows]
    observations = [row if isinstance(row, dict) else {'text': str(row), 'confidence': 1.0}
                    for row in rows or []]
    def valid(text, confidence):
        if confidence >= .2:
            return _same_text(expected, text)
        return confidence >= .03 and len(_normalized(expected)) >= 4 and _normalized(expected) == _normalized(text)
    matched = [row for row in observations if valid(row.get('text', ''), float(row.get('confidence', 0)))]
    if matched:
        return True, [row['box'] for row in matched if row.get('box')]
    if observations:
        joined = ''.join(row.get('text', '') for row in observations)
        score = min(float(row.get('confidence', 0)) for row in observations)
        if valid(joined, score):
            return True, [row['box'] for row in observations if row.get('box')]
    return False, []


def _roi(frame, region):
    height, width = frame.shape[:2]
    region = region or {'x_ratio': .05, 'y_ratio': .60, 'w_ratio': .90, 'h_ratio': .38}
    x = max(0, min(width-1, int(width*region.get('x_ratio', .05))))
    y = max(0, min(height-1, int(height*region.get('y_ratio', .60))))
    right = min(width, x+max(1,int(width*region.get('w_ratio', .90))))
    bottom = min(height, y+max(1,int(height*region.get('h_ratio', .38))))
    return x, y, right, bottom


def _default_recognizer(region=None):
    import easyocr
    from config import DEVICE
    reader = easyocr.Reader(['ch_sim'], gpu=DEVICE == 'cuda', download_enabled=False, verbose=False)
    def recognize(frame):
        x,y,right,bottom = _roi(frame,region)
        crop = frame[y:bottom,x:right]
        scale = min(1.0, 960/max(1,crop.shape[1]))
        if scale < 1:
            crop=cv2.resize(crop,(960,max(1,round(crop.shape[0]*scale))))
        return [{'text':text,'confidence':float(confidence),
                 'box':[(float(px)/scale+x,float(py)/scale+y) for px,py in box]}
                for box,text,confidence in reader.readtext(crop,detail=1,paragraph=False) if text.strip()]
    return recognize


def refine_visual_timing(video_path: str, srt_content: str, *, recognizer: Callable | None = None,
                         max_shift: float = 2.0, scan_fps: float = 6.0,
                         cancelled: Callable[[], bool] | None = None,
                         region: dict | None = None, progress: Callable | None = None) -> tuple[str,dict]:
    original = parse_srt_timing(srt_content)
    valid,errors = validate_srt(srt_content)
    if not valid:
        raise ValueError('SRT nguồn không hợp lệ: '+'; '.join(errors[:2]))
    if not 0 < max_shift <= 2 or not 1 <= scan_fps <= 15:
        raise ValueError('Tham số tinh chỉnh thời gian không hợp lệ')
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError('Không mở được video để tinh chỉnh mốc')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    if count <= 1 or not math.isfinite(fps) or fps <= 0:
        cap.release();raise RuntimeError('Video không có thông số frame hợp lệ')
    step=max(1,round(fps/scan_fps));window=max(1,round(fps*max_shift))
    timestamps={};ocr_cache={};pts_available=True;started=time.monotonic();refined=[]
    report={'method':'anchored_visual_timing_v2','cues':[],'refined':0,'unchanged':0,
            'verified':0,'ocr_calls':0,'visual_checks':0,'timestamp_basis':'decoder_msec',
            'video':str(Path(video_path))}
    try:
        if recognizer is None:
            from config import acquire_gpu_slot
            context=acquire_gpu_slot()
        else:
            context=nullcontext()
        with context:
            recognize=recognizer or _default_recognizer(region)
            def read_at(index):
                nonlocal pts_available
                if cancelled and cancelled():
                    raise RuntimeError('Đã hủy tinh chỉnh mốc phụ đề')
                if not 0 <= index < count:return None
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok or frame is None:return None
                ms=float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if not math.isfinite(ms) or index>0 and ms<=0:
                    pts_available=False;ms=index*1000/fps
                timestamps[index]=ms/1000
                return frame
            def ocr(index,text):
                key=(index,text)
                if key not in ocr_cache:
                    frame=read_at(index)
                    if frame is None:ocr_cache[key]=(None,[])
                    else:
                        try:
                            report['ocr_calls']+=1
                            ocr_cache[key]=_match_rows(text,recognize(frame))
                        except Exception:ocr_cache[key]=(None,[])
                return ocr_cache[key]
            for pos,(start_ms,end_ms,text) in enumerate(original):
                if cancelled and cancelled():raise RuntimeError('Đã hủy tinh chỉnh mốc phụ đề')
                a=max(0,min(count-1,round(start_ms*fps/1000)))
                b=max(a,min(count-1,round(end_ms*fps/1000)))
                lo=max(0,a-window);hi=min(count-1,b+window)
                if pos:
                    pa,pb,_=original[pos-1];lo=max(lo,round((pa+pb)*fps/2000))
                if pos+1<len(original):
                    na,nb,_=original[pos+1];hi=min(hi,round((na+nb)*fps/2000))
                middle=min(hi,max(lo,round((a+b)/2)))
                entry={'index':pos+1,'source_start_ms':start_ms,'source_end_ms':end_ms,
                       'status':'unchanged_no_anchor','start_verified':False,'end_verified':False}
                anchor=None
                if hi>lo and len(_normalized(text))>=2:
                    for index in sorted(set((a,middle,max(a,b-1))),key=lambda i:abs(i-middle)):
                        if lo<=index<=hi and ocr(index,text)[0] is True:
                            anchor=index;break
                if anchor is not None:
                    anchor_frame=read_at(anchor);reference=None;box=None
                    if anchor_frame is not None:
                        boxes=ocr(anchor,text)[1]
                        if boxes:
                            points=[point for polygon in boxes for point in polygon]
                            xs,ys=zip(*points);pad=max(5,round((max(ys)-min(ys))*.3))
                            height,width=anchor_frame.shape[:2]
                            box=(max(0,int(min(xs))-pad),max(0,int(min(ys))-pad),
                                 min(width,int(max(xs))+pad+1),min(height,int(max(ys))+pad+1))
                        else:box=_roi(anchor_frame,region)
                        x,y,right,bottom=box;reference=glyph_reference(anchor_frame[y:bottom,x:right])
                    checks={anchor:True}
                    def seen(index):
                        if index not in checks:
                            if reference is not None:
                                frame=read_at(index);report['visual_checks']+=1
                                checks[index]=None if frame is None else reference_matches(reference,frame[box[1]:box[3],box[0]:box[2]])
                            elif recognizer is not None:checks[index]=ocr(index,text)[0]
                            else:checks[index]=None
                        return None if checks[index] is None else bool(checks[index])
                    def bracket(center,low,high,entering):
                        center=max(low,min(high,center));state=seen(center)
                        if state is None:return None
                        backwards=state is entering;probe=center
                        for _ in range((high-low)//step+2):
                            nxt=max(low,probe-step) if backwards else min(high,probe+step)
                            if nxt==probe:
                                if entering and state is True and probe==0:return(-1,0)
                                if not entering and state is True and probe==count-1:return(count-1,count)
                                return None
                            next_state=seen(nxt)
                            if next_state is None:return None
                            if next_state is not state:return(nxt,probe) if backwards else(probe,nxt)
                            probe=nxt
                        return None
                    first=None;last=None
                    edge=bracket(a,lo,min(hi,a+window),True)
                    if edge:
                        negative,positive=edge
                        values=[seen(i) for i in range(negative+1,positive+1)]
                        if None not in values and True in values and positive<=anchor:
                            offset=values.index(True)
                            if all(values[offset:]):first=negative+1+offset
                    edge=bracket(b,max(lo,b-window),hi,False)
                    if edge:
                        positive,negative=edge
                        values=[seen(i) for i in range(positive,negative)]
                        if positive>=anchor and None not in values:
                            offset=values.index(False) if False in values else len(values)
                            if offset>0 and all(values[:offset]) and not any(values[offset:]):
                                last=positive+offset-1
                    # A constant full-video overlay is not evidence that a
                    # short dialogue cue should expand to the whole video.
                    if first == 0 and last == count - 1 and (a > 0 or b < count - 1):
                        first = last = None
                    new_start=start_ms;new_end=end_ms
                    prev_end=refined[-1]['end_ms'] if refined else 0
                    next_start=original[pos+1][0] if pos+1<len(original) else None
                    if first is not None:
                        candidate=round(timestamps[first]*1000)
                        if prev_end<=candidate<end_ms:
                            new_start=candidate;entry.update(start_verified=True,start_frame=first)
                    if last is not None:
                        if last+1<count:
                            next_frame=read_at(last+1)
                            candidate=round(timestamps[last+1]*1000) if next_frame is not None else None
                        else:candidate=round(count*1000/fps)
                        if candidate is not None and candidate>new_start and (next_start is None or candidate<=next_start):
                            new_end=candidate;entry.update(end_verified=True,end_frame_exclusive=last+1)
                    changed=(new_start,new_end)!=(start_ms,end_ms)
                    if changed:
                        entry.update(status='refined' if entry['start_verified'] and entry['end_verified'] else 'refined_start_only' if entry['start_verified'] else 'refined_end_only',
                                     start_ms=new_start,end_ms=new_end)
                        refined.append({'start_ms':new_start,'end_ms':new_end,'text':text});report['refined']+=1
                    else:
                        entry['status']='verified_unchanged' if entry['start_verified'] and entry['end_verified'] else 'unchanged_unverified_boundary'
                        refined.append({'start_ms':start_ms,'end_ms':end_ms,'text':text});report['unchanged']+=1
                    if entry['start_verified'] and entry['end_verified']:report['verified']+=1
                else:
                    refined.append({'start_ms':start_ms,'end_ms':end_ms,'text':text});report['unchanged']+=1
                report['cues'].append(entry)
                if progress:progress(pos+1,len(original),report['refined'],report['unchanged'])
    finally:cap.release()
    report['elapsed_seconds']=round(time.monotonic()-started,3)
    if not pts_available:
        report.update(timestamp_basis='unavailable',status='unchanged_no_frame_timestamps',refined=0,unchanged=len(original))
        return srt_content,report
    if report['refined']==0:return srt_content,report
    result=generate_srt([{'start':row['start_ms']/1000,'end':row['end_ms']/1000,'text':row['text']} for row in refined])
    if not validate_srt(result)[0]:
        report.update(status='unchanged_invalid_result',refined=0,unchanged=len(original))
        return srt_content,report
    return result,report


def refine_and_store(job: dict, video_path: str, content: str, output_dir: Path) -> str:
    try:
        cfg=job.get('pipeline_config') or {}
        region=(cfg.get('clean_options') or {}).get('sub_region') or (cfg.get('burn_options') or {}).get('sub_region') or job.get('sub_region')
        def progress(done,total,changed,unchanged):
            job['visual_timing_progress']={'done':done,'total':total,'refined':changed,'unchanged':unchanged}
            job['message']=f'🔎 Đối chiếu mốc bằng frame gốc: {done}/{total} câu, {changed} câu đã chỉnh.'
        result,report=refine_visual_timing(video_path,content,region=region,progress=progress,
                                         cancelled=lambda:bool(job.get('cancel')))
        report_path=output_dir/'hardsub_visual_timing.json'
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        job.setdefault('artifacts',{})['visual_timing_report']=str(report_path)
        job['visual_timing']={key:report[key] for key in ('method','refined','unchanged','timestamp_basis')}
        for key in ('verified','ocr_calls','visual_checks','elapsed_seconds'):
            if key in report:job['visual_timing'][key]=report[key]
        if result!=content:
            raw_path=output_dir/'hardsub_zh_original_timing.srt';raw_path.write_text(content,encoding='utf-8')
            job['artifacts']['srt_zh_original_timing']=str(raw_path)
        return result
    except Exception as exc:
        if job.get('cancel'):raise
        job.setdefault('subtitle_source_warnings',{})['visual_timing']=str(exc)[:500]
        return content
