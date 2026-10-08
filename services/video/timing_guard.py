"""Per-video visual guard for subtitle frames missed by an approximate SRT.

References are rebuilt from each source video inside the selected subtitle ROI.
No cue text, timestamp, font size, or video ID is hardcoded. A guard only repairs
bounded cue neighborhoods when the source glyph pattern is still visible; it is
not an OCR engine or a guarantee that missing whole cues have been discovered.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _Features:
    core: np.ndarray
    edge: np.ndarray


class _FrameEvidence:
    """One frame-local color/edge extraction, shared by detection and masking.

    Only the current ROI is retained. Learned references still store their two
    small masks; no full-frame evidence is cached across videos or cues.
    """
    def __init__(self, image: np.ndarray):
        self.image = image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
        yellow = cv2.inRange(hsv, (16, 65, 110), (38, 255, 255))
        self.bright = cv2.bitwise_or(white, yellow)
        self.dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 95))
        self._core = None
        self._edge = None

    @property
    def core(self):
        if self._core is None:
            self._core = cv2.morphologyEx(self.bright, cv2.MORPH_CLOSE,
                                          np.ones((3, 3), np.uint8))
        return self._core

    @property
    def edge(self):
        if self._edge is None:
            self._edge = cv2.Canny(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), 60, 160)
        return self._edge

    def features(self) -> _Features:
        return _Features(self.core, self.edge)


def _component_outline_fraction(labels: np.ndarray, dark: np.ndarray,
                                label: int, stats: np.ndarray) -> float:
    """Measure exactly the same 5x5 rim, without allocating a full-ROI mask.

    A 5x5 dilation reaches two pixels beyond the component bounding box. Keep
    that halo (clipped at the image boundary) so rim area and dark overlap are
    identical to the original whole-image calculation.
    """
    x, y, width, height = (int(value) for value in stats[:4])
    left, top = max(0, x - 2), max(0, y - 2)
    right = min(labels.shape[1], x + width + 2)
    bottom = min(labels.shape[0], y + height + 2)
    component = (labels[top:bottom, left:right] == label).astype(np.uint8) * 255
    rim = cv2.subtract(cv2.dilate(component, np.ones((5, 5), np.uint8)), component)
    area = np.count_nonzero(rim)
    if not area:
        return 0.0
    return np.count_nonzero(cv2.bitwise_and(rim, dark[top:bottom, left:right])) / area


def _outlined_glyphs(image: np.ndarray, *, evidence: _FrameEvidence | None = None) -> _Features | None:
    """Keep a coherent line of bright glyphs with nearby dark outlines.

    Background brightness alone is never sufficient evidence for extra erasing.
    Sizes and line alignment are measured relative to this video/ROI.
    """
    height, width = image.shape[:2]
    if height < 12 or width < 24:
        return None
    evidence = evidence if evidence is not None else _FrameEvidence(image)
    bright, dark = evidence.core, evidence.dark
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if (y <= 1 or y + h >= height - 1 or h < max(7, height * .08) or
                h > height * .8 or w < 2 or w > h * 2.8 or area < 12):
            continue
        if area / max(1, w*h) > .85:
            continue  # Solid bright patches/windows are not character strokes.
        if _component_outline_fraction(labels, dark, label, stats[label]) < .18:
            continue
        candidates.append((label, float(y + h / 2), int(h)))
    if len(candidates) < 3:
        return None
    median_height = float(np.median([h for _, _, h in candidates]))
    rows = []
    for label, center, h in sorted(candidates, key=lambda item: item[1]):
        if not .6 * median_height <= h <= 1.5 * median_height:
            continue
        if not rows or abs(center - np.mean([entry[1] for entry in rows[-1]])) > median_height * .35:
            rows.append([])
        rows[-1].append((label, center))
    selected = [label for row in rows if len(row) >= 3 for label, _ in row]
    if len(selected) < 3:
        return None
    selected_labels = np.zeros(count, np.uint8)
    selected_labels[selected] = 255
    core = selected_labels[labels]
    coverage = np.count_nonzero(core) / (height * width)
    if not .002 <= coverage <= .45:
        return None
    edges = evidence.edge
    edges = cv2.bitwise_and(edges, cv2.dilate(core, np.ones((5, 5), np.uint8)))
    if np.count_nonzero(edges) < 40:
        return None
    return _Features(core, edges)


def _mask_from_core(image: np.ndarray, core: np.ndarray, *,
                    evidence: _FrameEvidence | None = None) -> np.ndarray:
    evidence = evidence if evidence is not None else _FrameEvidence(image)
    # The matching cores prove the line exists, but Chinese radicals can be
    # disconnected/shorter than the components used to identify that line.
    # Recover *all* bright strokes within each confirmed line envelope rather
    # than leaving excluded radicals in the final video.
    envelope = np.zeros_like(core)
    occupied = np.any(core > 0, axis=1)
    edges = np.diff(np.r_[False, occupied, False].astype(np.int8))
    for top, bottom in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        xs = np.flatnonzero(np.any(core[top:bottom] > 0, axis=0))
        if xs.size:
            envelope[max(0, top-4):min(core.shape[0], bottom+4), max(0, xs[0]-4):min(core.shape[1], xs[-1]+5)] = 255
    core = cv2.bitwise_or(core, cv2.bitwise_and(evidence.bright, envelope))
    dark = evidence.dark
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    outline = cv2.bitwise_and(dark, cv2.dilate(core, kernel, iterations=2))
    full = cv2.morphologyEx(cv2.bitwise_or(core, outline), cv2.MORPH_CLOSE, kernel)
    return cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def glyph_reference(image: np.ndarray):
    """Build a lightweight spatial reference; None means insufficient evidence."""
    return _outlined_glyphs(image)


def reference_matches(expected: _Features, image: np.ndarray) -> bool:
    if expected.core.shape != image.shape[:2]:
        return False
    # The reference already established which pixels belong to text. Re-running
    # component/line selection on every frame lets moving clothing/background
    # change the inferred line and can falsely move an otherwise stable cue.
    return _features_match(expected, _FrameEvidence(image).features())


def _features_match(expected: _Features, current: _Features) -> bool:
    core_near = cv2.dilate(current.core, np.ones((3, 3), np.uint8))
    edge_near = cv2.dilate(current.edge, np.ones((3, 3), np.uint8))
    core_match = np.count_nonzero(cv2.bitwise_and(expected.core, core_near)) / np.count_nonzero(expected.core)
    edge_match = np.count_nonzero(cv2.bitwise_and(expected.edge, edge_near)) / np.count_nonzero(expected.edge)
    return bool(core_match >= .82 and edge_match >= .72)


def _line_styles(features: _Features) -> list[tuple]:
    occupied = np.any(features.core > 0, axis=1)
    edges = np.diff(np.r_[False, occupied, False].astype(np.int8))
    styles = []
    for top, bottom in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        xs = np.flatnonzero(np.any(features.core[top:bottom] > 0, axis=0))
        if xs.size and bottom-top >= 7:
            styles.append((float(bottom-top), float((top+bottom)/2), float((xs[0]+xs[-1])/2)))
    return styles


class SubtitleTimingGuard:
    def __init__(self, references: list[dict], window: float = 1.0):
        self.references = sorted(references, key=lambda item: item['start'])
        self.starts = [row['start'] for row in self.references]
        self.ends = [row['end'] for row in self.references]
        self.window = window
        self.template_count = len(self.references)
        self.sampled_frames = 0
        self.missing_templates = 0
        self.styles = [style for row in self.references for style in _line_styles(row['features'])]
        self.presence_gap_intervals = []

    @classmethod
    def from_references(cls, cues: list[dict], window: float = 1.0):
        references = []
        for cue in cues:
            features = _outlined_glyphs(cue['image'])
            if features is not None:
                references.append({'start': cue['start'], 'end': cue['end'], 'features': features})
        instance = cls(references, window)
        instance.sampled_frames = len(cues)
        instance.missing_templates = len(cues) - len(references)
        return instance

    @classmethod
    def from_video(cls, path: str, cues: list[tuple], box: tuple, window: float = 1.0, cancelled=None):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError('Không mở được video để tạo mẫu chữ')
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        x, y, w, h = box
        references, sampled, missing = [], 0, 0
        try:
            for start_ms, end_ms, _text in cues:
                found = None
                for fraction in (.5, .25, .75):
                    if cancelled and cancelled():
                        raise RuntimeError('Đã hủy tạo mẫu chữ')
                    t = (start_ms + (end_ms - start_ms) * fraction) / 1000
                    index = max(0, min(frames - 1, round(t * fps)))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    sampled += 1
                    if ok and frame is not None:
                        found = _outlined_glyphs(frame[y:y + h, x:x + w])
                        if found is not None:
                            break
                if found is not None:
                    references.append({'start': start_ms / 1000, 'end': end_ms / 1000,
                                       'features': found})
                else:
                    missing += 1
        finally:
            cap.release()
        instance = cls(references, window)
        instance.sampled_frames = sampled
        instance.missing_templates = missing
        return instance

    def _matches_style(self, current: _Features) -> bool:
        lines = _line_styles(current)
        if not lines or not self.styles:
            return False
        return all(any(.65*h <= ch <= 1.45*h and abs(cy-y) <= max(4, h*.5)
                       and abs(cx-x) <= max(20, current.core.shape[1]*.22)
                       for h,y,x in self.styles) for ch,cy,cx in lines)

    def _matched_reference(self, evidence: _FrameEvidence, lo: int, hi: int):
        if lo >= hi:
            return None
        # A nearly solid white panel can cover every expected stroke and make a
        # one-way template overlap look convincing. Preserve the original
        # coverage bound, and demand reverse spatial overlap as well.
        coverage = np.count_nonzero(evidence.core) / evidence.core.size
        if not .002 <= coverage <= .45:
            return None
        raw = None
        for row in self.references[lo:hi]:
            expected = row['features']
            if expected.core.shape != evidence.image.shape[:2]:
                continue
            x, y, width, height = cv2.boundingRect(expected.core)
            left, top = max(0, x-1), max(0, y-1)
            right = min(expected.core.shape[1], x+width+1)
            bottom = min(expected.core.shape[0], y+height+1)
            current_core = evidence.core[top:bottom, left:right]
            current_pixels = np.count_nonzero(current_core)
            if not current_pixels:
                continue
            near_expected = cv2.dilate(expected.core[top:bottom, left:right],
                                       np.ones((3, 3), np.uint8))
            precision = np.count_nonzero(cv2.bitwise_and(current_core, near_expected)) / current_pixels
            if precision < .72:
                continue
            if raw is None:
                raw = evidence.features()
            if _features_match(expected, raw):
                return expected
        return None

    def scan_extra_intervals(self, path: str, base_intervals: list[tuple], box: tuple, cancelled=None):
        """Find old-text presence outside SRT, including wholly omitted cues.

        Unknown content needs a learned source line style and three consecutive
        matching glyph-pattern frames. Its initial frames are then included too.
        No OCR/provider call is used in this cleanup prepass.
        """
        self.presence_gap_intervals = []
        if not self.template_count:
            return [], 0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError('Không mở được video để kiểm tra vùng biên phụ đề')
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        x,y,w,h = box
        index = interval_index = recovered = 0
        extra = []
        run_start = None
        run_count = 0
        run_confirmed = False
        run_has_unknown = False
        previous = None
        def flush(timestamp):
            nonlocal recovered,run_start,run_count,run_confirmed,run_has_unknown,previous
            if run_start is not None and run_confirmed:
                extra.append((run_start,timestamp))
                recovered += run_count
                if run_has_unknown:
                    self.presence_gap_intervals.append((run_start,timestamp))
            run_start = None;run_count = 0;run_confirmed = False
            run_has_unknown = False;previous = None
        try:
            while cap.grab():
                if cancelled and cancelled():
                    raise RuntimeError('Đã hủy kiểm tra vùng biên phụ đề')
                timestamp = index/fps
                while interval_index<len(base_intervals) and base_intervals[interval_index][1]<timestamp:
                    interval_index+=1
                active=interval_index<len(base_intervals) and base_intervals[interval_index][0]<=timestamp
                candidate = None
                evidence = None
                known = False
                if not active:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        strip = frame[y:y+h, x:x+w]
                        if strip.shape[0] >= 12 and strip.shape[1] >= 24:
                            evidence = _FrameEvidence(strip)
                            candidate = _outlined_glyphs(strip, evidence=evidence)
                            lo = bisect_left(self.ends, timestamp-self.window)
                            hi = bisect_right(self.starts, timestamp+self.window)
                            if candidate is not None:
                                known = any(row['features'].core.shape == candidate.core.shape and
                                            _features_match(row['features'], candidate)
                                            for row in self.references[lo:hi])
                            else:
                                # Scene brightness can defeat fresh component selection
                                # without changing the already learned text/outline.
                                candidate = self._matched_reference(evidence, lo, hi)
                                known = candidate is not None
                            if candidate is not None and not known and not self._matches_style(candidate):
                                candidate = None
                if candidate is None:
                    flush(timestamp)
                else:
                    assert evidence is not None
                    if previous is None or not _features_match(previous, evidence.features()):
                        flush(timestamp)
                        run_start=timestamp
                    run_count+=1
                    run_has_unknown=run_has_unknown or not known
                    run_confirmed=run_confirmed or known or run_count>=3
                    previous=candidate
                index+=1
            flush(index/fps)
        finally:
            cap.release()
        return extra,recovered

    def mask_if_visible(self, strip: np.ndarray, timestamp: float, *, allow_style: bool = False) -> np.ndarray | None:
        lo = bisect_left(self.ends, timestamp - self.window)
        hi = bisect_right(self.starts, timestamp + self.window)
        if lo >= hi and not allow_style:
            return None
        if strip.shape[0] < 12 or strip.shape[1] < 24:
            return None
        evidence = _FrameEvidence(strip)
        current = _outlined_glyphs(strip, evidence=evidence)
        if current is None:
            expected = self._matched_reference(evidence, lo, hi)
            if expected is not None:
                return _mask_from_core(strip, expected.core, evidence=evidence)
            return None
        for row in self.references[lo:hi]:
            expected = row['features']
            if expected.core.shape == current.core.shape and _features_match(expected, current):
                return _mask_from_core(strip, current.core, evidence=evidence)
        if allow_style and self._matches_style(current):
            return _mask_from_core(strip, current.core, evidence=evidence)
        return None
