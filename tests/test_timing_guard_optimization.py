"""Exact-equivalence gates for the local component-rim optimization."""
import cv2
import numpy as np
import pytest

from services.video.timing_guard import _component_outline_fraction


@pytest.mark.parametrize("edge", [False, True])
def test_component_rim_score_equals_full_frame_score(edge, monkeypatch):
    image = np.zeros((120, 960), np.uint8)
    x = 0 if edge else 450
    image[30:58, x:x + 18] = 255
    image[36:51, x + 5:x + 13] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image)
    dark = np.random.default_rng(17).integers(0, 2, image.shape, dtype=np.uint8) * 255
    component = (labels == 1).astype(np.uint8) * 255
    rim = cv2.subtract(cv2.dilate(component, np.ones((5, 5), np.uint8)), component)
    expected = np.count_nonzero(cv2.bitwise_and(rim, dark)) / np.count_nonzero(rim)
    seen = []
    real = cv2.dilate
    def observe(mask, *args, **kwargs):
        seen.append(mask.shape)
        return real(mask, *args, **kwargs)
    monkeypatch.setattr(cv2, "dilate", observe)
    actual = _component_outline_fraction(labels, dark, 1, stats[1])
    assert actual == expected
    assert seen and all(height <= 32 and width <= 22 for height, width in seen)


def _caption(value=70, text="OLD SUB"):
    image = np.full((90, 430, 3), value, np.uint8)
    if text:
        cv2.putText(image, text, (35, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 7)
        cv2.putText(image, text, (35, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2)
    return image


def _guard():
    from services.video.timing_guard import SubtitleTimingGuard
    return SubtitleTimingGuard.from_references([{"start": .2, "end": .32, "image": _caption()}])


def test_mask_reuses_one_hsv_conversion_per_frame(monkeypatch):
    guard = _guard()
    calls = []
    real = cv2.cvtColor
    def observe(image, code, *args, **kwargs):
        calls.append(code)
        return real(image, code, *args, **kwargs)
    monkeypatch.setattr(cv2, "cvtColor", observe)
    assert guard.mask_if_visible(_caption(), .1) is not None
    assert calls.count(cv2.COLOR_BGR2HSV) == 1


def test_presence_scan_reuses_edges_instead_of_running_canny_twice(tmp_path, monkeypatch):
    video = tmp_path / "same_caption.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (430, 90))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(_caption())
    writer.release()
    guard = _guard()
    calls = []
    real = cv2.Canny
    def observe(*args, **kwargs):
        calls.append(True)
        return real(*args, **kwargs)
    monkeypatch.setattr(cv2, "Canny", observe)
    extra, recovered = guard.scan_extra_intervals(str(video), [], (0, 0, 430, 90))
    assert extra and recovered == 12
    assert len(calls) == 12


def test_bright_background_known_caption_is_recovered_by_blur_prepass(tmp_path):
    video = tmp_path / "bright_caption.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (430, 90))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(_caption(value=100))
    writer.release()
    guard = _guard()
    extra, recovered = guard.scan_extra_intervals(str(video), [(.2, .32)], (0, 0, 430, 90))
    assert recovered > 0 and extra[0][0] == 0
    assert extra[-1][1] >= .44


@pytest.mark.parametrize("value", [96, 100, 160, 240])
def test_known_reference_fallback_never_accepts_blank_or_other_text(value):
    guard = _guard()
    assert guard.mask_if_visible(_caption(value=value, text=""), .1) is None
    assert guard.mask_if_visible(_caption(value=value, text="NEW LINE"), .1) is None
    assert guard.mask_if_visible(_caption(value=value), 10.0) is None


def test_invalid_video_does_not_look_like_successful_empty_scan(tmp_path):
    with pytest.raises(RuntimeError, match="video"):
        _guard().scan_extra_intervals(str(tmp_path / "missing.mp4"), [], (0, 0, 430, 90))


def test_local_bright_panel_cannot_turn_other_caption_into_source_evidence():
    image = _caption(text="")
    image[20:70, 25:310] = 240
    for color, width in (((0, 0, 0), 7), ((255, 255, 255), 2)):
        cv2.putText(image, "NEW LINE", (35, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, width)
    assert _guard().mask_if_visible(image, .1) is None
