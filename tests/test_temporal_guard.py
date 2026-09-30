"""Visual timing guard must recover early text, not paint clean clothing/background."""
import cv2
import numpy as np

from services.video.timing_guard import SubtitleTimingGuard


def _strip(text='OLD SUB', value=70):
    image = np.full((90, 430, 3), value, np.uint8)
    if text:
        cv2.putText(image, text, (35, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 7)
        cv2.putText(image, text, (35, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2)
    return image


def _guard():
    return SubtitleTimingGuard.from_references([
        {'start': 30.0, 'end': 32.0, 'image': _strip()},
    ])


def test_recovers_same_text_before_unverified_srt_start():
    guard = _guard()
    mask = guard.mask_if_visible(_strip(value=100), 29.5)
    assert mask is not None and np.count_nonzero(mask) > 100
    assert mask.shape == (90, 430)


def test_does_not_inpaint_white_background_without_matching_glyphs():
    guard = _guard()
    background = _strip(text='', value=80)
    background[15:65, 20:300] = 245
    assert guard.mask_if_visible(background, 29.5) is None
    assert guard.mask_if_visible(_strip(text=''), 32.5) is None


def test_different_caption_or_distant_frame_is_not_the_same_text():
    guard = _guard()
    assert guard.mask_if_visible(_strip(text='NEW LINE'), 29.5) is None
    assert guard.mask_if_visible(_strip(), 27.5) is None


def test_no_template_means_no_unsafe_extra_inpainting():
    guard = SubtitleTimingGuard.from_references([
        {'start': 30.0, 'end': 32.0, 'image': _strip(text='')},
    ])
    assert guard.mask_if_visible(_strip(), 29.5) is None
    assert guard.template_count == 0


def test_each_video_builds_its_own_font_size_and_two_line_reference():
    for scale in (.8, 1.45):
        image = np.full((150, 470, 3), 65, np.uint8)
        for text, y in (('FIRST', 55), ('SECOND', 112)):
            cv2.putText(image, text, (35, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 7)
            cv2.putText(image, text, (35, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2)
        guard = SubtitleTimingGuard.from_references([{'start':4.0, 'end':6.0, 'image':image}])
        assert guard.template_count == 1
        assert guard.mask_if_visible(image, 3.5) is not None


def test_pipeline_erases_early_frames_without_erasing_blank_background(tmp_path):
    from services.video.clean_pipeline import clean_video_pipeline
    video = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (320,240))
    assert writer.isOpened()
    for i in range(40):
        image = np.full((240,320,3),75,np.uint8)
        if 5 <= i <= 27:
            for j, char in enumerate('OLD'):
                cv2.putText(image,char,(80+j*26,184),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),6)
                cv2.putText(image,char,(80+j*26,184),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        writer.write(image)
    writer.release()
    region={'x_ratio':.1,'y_ratio':.65,'w_ratio':.8,'h_ratio':.2}
    srt='1\n00:00:00,800 --> 00:00:01,000\nOLD\n'
    outputs=[]
    for enabled in (False, True):
        dest=tmp_path / f'clean_{enabled}.mp4'
        result=clean_video_pipeline(str(video),region,srt,str(dest),'test_temporal_guard',timing_guard=enabled)
        outputs.append((dest,result))
    assert outputs[1][1]['timing_guard']['recovered_frames'] > 0
    def frame(path,t):
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
        ok,image=cap.read();cap.release();assert ok;return image
    old,new=frame(outputs[0][0],.4),frame(outputs[1][0],.4)
    area=(slice(160,190),slice(70,165))
    old_white=np.count_nonzero(cv2.cvtColor(old[area],cv2.COLOR_BGR2GRAY)>180)
    new_white=np.count_nonzero(cv2.cvtColor(new[area],cv2.COLOR_BGR2GRAY)>180)
    assert old_white > 50 and new_white < old_white*.25
    blank=frame(outputs[1][0],.08)
    assert abs(float(blank[160:190,70:165].mean())-75) < 10


def test_guard_mask_includes_disconnected_short_strokes_between_glyphs():
    image=np.full((90,430,3),65,np.uint8)
    for char,x in [('O',35),('D',110),('L',200)]:
        cv2.putText(image,char,(x,56),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),7)
        cv2.putText(image,char,(x,56),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2)
    image[35:40,160:173]=255
    image[45:50,160:173]=255
    guard=SubtitleTimingGuard.from_references([{'start':10.0,'end':12.0,'image':image}])
    mask=guard.mask_if_visible(image,9.5)
    assert mask is not None
    assert mask[37,166]==255 and mask[47,166]==255


def test_blur_prepass_produces_only_visually_confirmed_extra_intervals(tmp_path):
    path=tmp_path/'blur_source.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(430,90))
    for i in range(40):writer.write(_strip() if 5<=i<=27 else _strip(text=''))
    writer.release()
    guard=SubtitleTimingGuard.from_video(str(path),[(800,1000,'OLD SUB')],(0,0,430,90))
    extra,recovered=guard.scan_extra_intervals(str(path),[(.7,1.15)],(0,0,430,90))
    assert recovered>=10
    assert extra and abs(extra[0][0]-.2)<.001
    assert extra[0][1]<=.7+1/25


def test_yellow_outlined_subtitle_uses_the_current_video_reference():
    yellow=_strip()
    yellow[np.all(yellow==255,axis=2)]=(0,230,255)
    guard=SubtitleTimingGuard.from_references([{'start':7.0,'end':9.0,'image':yellow}])
    assert guard.template_count==1
    assert guard.mask_if_visible(yellow,6.5) is not None


def test_presence_scan_finds_styled_subtitle_missing_from_srt(tmp_path):
    path=tmp_path/'missing_cue.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(430,90))
    for i in range(100):
        if 5<=i<=25:image=_strip('OLD SUB')
        elif 60<=i<=85:image=_strip('NEW LINE')
        else:image=_strip(text='')
        writer.write(image)
    writer.release()
    guard=SubtitleTimingGuard.from_video(str(path),[(200,1000,'OLD SUB')],(0,0,430,90))
    extra,count=guard.scan_extra_intervals(str(path),[(.1,1.15)],(0,0,430,90))
    assert count>10
    assert any(a<=2.4 and b>=3.4 for a,b in extra)


def test_presence_scan_does_not_erase_a_brief_or_bright_background(tmp_path):
    path=tmp_path/'background.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(430,90))
    for i in range(100):
        image=_strip('OLD SUB') if 5<=i<=25 else _strip(text='')
        if 60<=i<=85:image[20:65,20:300]=245
        if i==90:image=_strip('NEW LINE')
        writer.write(image)
    writer.release()
    guard=SubtitleTimingGuard.from_video(str(path),[(200,1000,'OLD SUB')],(0,0,430,90))
    extra,_=guard.scan_extra_intervals(str(path),[(.1,1.15)],(0,0,430,90))
    assert not any(b>2 for a,b in extra)


def test_clean_pipeline_removes_stable_caption_missing_from_source_srt(tmp_path):
    from services.video.clean_pipeline import clean_video_pipeline
    path=tmp_path/'missing_source.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(430,90))
    for i in range(100):
        if 5<=i<=25:image=_strip('OLD SUB')
        elif 60<=i<=85:image=_strip('NEW LINE')
        else:image=_strip(text='')
        writer.write(image)
    writer.release()
    dest=tmp_path/'missing_clean.mp4'
    result=clean_video_pipeline(str(path),{'x_ratio':0,'y_ratio':0,'w_ratio':1,'h_ratio':1},
                               '1\n00:00:00,200 --> 00:00:01,000\nOLD SUB\n',str(dest),'test_missing_cleanup')
    cap=cv2.VideoCapture(str(dest));cap.set(cv2.CAP_PROP_POS_MSEC,2800);ok,im=cap.read();cap.release()
    assert ok and np.count_nonzero(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)>180)<40
    assert result['timing_guard']['presence_gap_intervals']
