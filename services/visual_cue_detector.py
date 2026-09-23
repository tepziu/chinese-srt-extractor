"""Visual subtitle cue timing primitives for contact-sheet OCR."""
from __future__ import annotations
import json, math, re
from difflib import SequenceMatcher
from typing import Any
import cv2
import numpy as np

DEFAULT_ROI = (0.10, 0.675, 0.90, 0.785)


def scan_visual_candidates(video_path, roi=DEFAULT_ROI, target_fps=8.0, quantile=0.70, nms_seconds=0.22):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps / target_fps))
    metrics = {name: [] for name in ("raw", "edge", "white", "tophat")}
    times, previous, frame_no = [], None, 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % step:
                frame_no += 1
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = roi
            crop = frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
            if crop.size == 0:
                raise ValueError("ROI phụ đề không hợp lệ")
            crop = cv2.resize(crop, (288, 44), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 80, 160) > 0
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            white = (hsv[:,:,1] < 95) & (hsv[:,:,2] > 175)
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel) > 35
            current = (gray, edge, white, tophat)
            if previous is not None:
                times.append(frame_no / fps)
                metrics["raw"].append(float(np.mean(cv2.absdiff(gray, previous[0])) / 255.0))
                for name, cur, prev in zip(("edge","white","tophat"), current[1:], previous[1:]):
                    metrics[name].append(float(np.mean(cur != prev)))
            previous = current
            frame_no += 1
    finally:
        cap.release()
    raw = []
    for name, vals in metrics.items():
        values = np.asarray(vals, dtype=np.float32)
        if len(values) < 3: continue
        threshold = float(np.quantile(values, quantile))
        for i in range(1, len(values)-1):
            value = float(values[i])
            if value >= threshold and value >= values[i-1] and value > values[i+1]:
                raw.append({"time": times[i], "signal": name, "value": value, "threshold": threshold})
    raw.sort(key=lambda x: x["time"])
    clusters = []
    for item in raw:
        if not clusters or item["time"] - clusters[-1][-1]["time"] > nms_seconds:
            clusters.append([item])
        else: clusters[-1].append(item)
    candidates = []
    for cluster in clusters:
        weights = [max(x["value"] / max(x["threshold"], 1e-9), 1.0) for x in cluster]
        t = sum(x["time"]*w for x,w in zip(cluster, weights))/sum(weights)
        candidates.append({"time": round(t,3), "signals": sorted({x["signal"] for x in cluster})})
    return {"source_fps": fps, "duration": frame_count/fps if frame_count else 0,
            "frame_step": step, "candidates": candidates, "roi": list(roi)}


def candidate_intervals(candidate_times, duration, guard_gap=1.2, min_interval=0.16):
    points = sorted(set([0.0] + [round(float(t),3) for t in candidate_times if 0 < float(t) < duration] + [round(float(duration),3)]))
    guarded = [points[0]]
    for right in points[1:]:
        left = guarded[-1]
        while right-left > guard_gap:
            left = round(min(right, left+guard_gap),3); guarded.append(left)
        if right > guarded[-1]: guarded.append(right)
    return [{"start_ms":round(a*1000), "end_ms":round(b*1000), "sample_ms":round((a+b)*500)}
            for a,b in zip(guarded,guarded[1:]) if b-a >= min_interval]


def build_contact_sheet(samples, columns=4):
    if not samples: raise ValueError("Không có frame để tạo contact sheet")
    columns = max(1, int(columns)); label_h = 30
    cell_w = max(x["image"].shape[1] for x in samples)
    image_h = max(x["image"].shape[0] for x in samples)
    rows = math.ceil(len(samples)/columns)
    sheet = np.zeros((rows*(image_h+label_h), columns*cell_w, 3), dtype=np.uint8)
    ids = []
    for pos,item in enumerate(samples):
        row,col = divmod(pos,columns); x,y=col*cell_w,row*(image_h+label_h); image=item["image"]
        sheet[y+label_h:y+label_h+image.shape[0],x:x+image.shape[1]]=image
        cv2.putText(sheet,item["frame_id"],(x+8,y+22),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,255,255),2,cv2.LINE_AA)
        ids.append(item["frame_id"])
    return sheet, ids

def parse_contact_response(raw, expected_ids):
    text = re.sub(r"^```(?:json)?\s*", "", str(raw or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start,end=text.find("{"),text.rfind("}")
    if start < 0 or end < start: raise ValueError("Gemini không trả JSON contact sheet")
    data=json.loads(text[start:end+1]); rows=data.get("frames") if isinstance(data,dict) else None
    if not isinstance(rows,list): raise ValueError("Gemini không trả frames")
    result={}
    for row in rows:
        if not isinstance(row,dict) or row.get("frame_id") not in expected_ids: continue
        fid=row["frame_id"]
        if fid in result: raise ValueError(f"Gemini trả trùng frame_id {fid}")
        kind=str(row.get("kind") or "none").lower()
        if kind not in {"dialogue","title","watermark","ui","none"}: kind="none"
        result[fid]={"text":" ".join(str(row.get("text") or "").split()),"kind":kind}
    missing=[fid for fid in expected_ids if fid not in result]
    if missing: raise ValueError(f"Gemini thiếu frame_id: {missing[:8]}")
    return result


def _norm(value):
    return "".join(str(value or "").split()).strip("，。！？,.!?;；:：")


def merge_ocr_states(states, similarity_threshold=.90):
    cues=[]; current=None
    for state in states:
        text=_norm(state.get("text","")); kind=state.get("kind","none")
        if kind != "dialogue" or not text:
            if current: cues.append(current); current=None
            continue
        if current:
            ratio=SequenceMatcher(None,_norm(current["text"]),text).ratio()
            if ratio >= similarity_threshold and state["start_ms"] <= current["end_ms"]+250:
                current["end_ms"]=state["end_ms"]
                if len(text)>len(current["text"]): current["text"]=text
                continue
            cues.append(current)
        current={"start_ms":state["start_ms"],"end_ms":state["end_ms"],"text":text}
    if current: cues.append(current)
    return [{"start":x["start_ms"]/1000,"end":x["end_ms"]/1000,"text":x["text"]}
            for x in cues if x["end_ms"]-x["start_ms"] >= 250]
