import os
from pathlib import Path
from unittest.mock import patch

from config import create_job,jobs
from services.burn_sub import _srt_to_ass
from services import pipeline_orchestrator as pipeline


def test_ass_places_wrapped_subtitle_at_original_region_center():
    region={'x_ratio':.1,'y_ratio':.7,'w_ratio':.8,'h_ratio':.08}
    srt='1\n00:00:00,000 --> 00:00:02,000\nĐây là một câu tiếng Việt dài cần tự xuống dòng nhưng vẫn nằm ở vị trí phụ đề gốc.\n'
    ass=_srt_to_ass(srt,720,1280,region)
    assert r'\an5\pos(360,947)' in ass
    assert r'\q2' in ass
    assert r'\N' in ass


def test_clean_then_burn_keeps_original_geometry(tmp_path,monkeypatch):
    jid='test_source_layout_123'
    original=tmp_path/'original.mp4';original.write_bytes(b'video fixture')
    region={'x_ratio':.1,'y_ratio':.7,'w_ratio':.8,'h_ratio':.08,'method':'source_ocr'}
    create_job(jid,status='processing',artifacts={},step_results={},video_path=str(original),
               video_file={'path':str(original)},source_visual_srt='1\n00:00:00,000 --> 00:00:01,000\n你好\n')
    monkeypatch.setattr(pipeline,'OUTPUT_FOLDER',tmp_path)
    seen={}
    def clean(**kwargs):
        Path(kwargs['output_path']).write_bytes(b'cleaned video fixture')
        return {'status':'done','path':kwargs['output_path'],'sub_region':region}
    def burn(**kwargs):
        seen.update(kwargs);dest=tmp_path/jid/'final.mp4';dest.write_bytes(b'final video fixture')
        return {'status':'done','path':str(dest),'sub_region':kwargs.get('sub_region')}
    try:
        with patch('services.burn_sub.detect_hardsub_region',return_value=region) as detect, \
             patch('services.video.clean_pipeline.clean_video_pipeline',side_effect=clean), \
             patch('services.burn_sub.burn_sub_video',side_effect=burn):
            pipeline._execute_remaining_steps(jid,str(original),'vi','1\n00:00:00,000 --> 00:00:01,000\nXin chào\n','',
                                             {'clean_video':True,'burn_sub':True,'tts':False},{'region_mode':'auto'}, {}, {})
        assert seen['render_mode']=='pure_burn'
        assert seen['sub_region']==region
        assert jobs[jid]['source_sub_region']==region
        detect.assert_called_once()
    finally:jobs.pop(jid,None)


def test_rendered_caption_is_centered_and_fits_source_band(tmp_path):
    import cv2,numpy as np,subprocess
    region={'x_ratio':.1,'y_ratio':.7,'w_ratio':.8,'h_ratio':.08}
    text='Đây là một câu tiếng Việt dài cần xuống dòng nhưng vẫn nằm trong vùng phụ đề gốc.'
    ass=tmp_path/'source_position.ass';ass.write_text(_srt_to_ass('1\n00:00:00,000 --> 00:00:01,000\n'+text+'\n',720,1280,region),encoding='utf-8')
    escaped=str(ass).replace('\\','/').replace(':','\\:')
    png=tmp_path/'frame.png'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=black:s=720x1280:r=1:d=1',
                    '-vf',"ass='"+escaped+"'",'-frames:v','1',str(png)],check=True,capture_output=True)
    frame=cv2.imread(str(png));ys,xs=np.where(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)>180)
    assert len(xs)>20
    assert xs.min()>=72-3 and xs.max()<=648+3
    assert ys.min()>=896-4 and ys.max()<=999+4
    assert abs((float(ys.min())+float(ys.max()))/2-947)<=10


def test_literal_ass_commands_cannot_override_original_position():
    region={'x_ratio':.1,'y_ratio':.7,'w_ratio':.8,'h_ratio':.08}
    ass=_srt_to_ass('1\n00:00:00,000 --> 00:00:01,000\nText {\\pos(0,0)}\n',720,1280,region)
    dialogue=next(line for line in ass.splitlines() if line.startswith('Dialogue:'))
    assert dialogue.count(r'\pos(')==1
    assert r'\pos(360,947)' in dialogue


def test_cjk_and_long_tokens_are_wrapped_without_losing_content():
    from services.video.subtitle_layout import fit_subtitle
    text='这是一个比较长的中文字幕需要保持原来的字幕位置'
    layout=fit_subtitle(text,560,100,49)
    assert ''.join(layout['lines'])==text
    assert layout['width']<=548 and layout['height']<=92
    assert not layout['overflow']
