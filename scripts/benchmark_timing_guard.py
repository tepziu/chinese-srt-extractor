"""CPU-only, paired timing-guard benchmark; never contacts a provider or outbox.

Compare the worktree against a Git version on identical synthetic frames/video.
Run from any directory: py -3.12 scripts/benchmark_timing_guard.py --render
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.video import timing_guard as current


def load_baseline(ref):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:services/video/timing_guard.py"], cwd=ROOT,
    )
    name = "_benchmark_timing_guard_baseline"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"{ref}:timing_guard.py", "exec"), module.__dict__)
    engine_source = subprocess.check_output(
        ["git", "show", f"{ref}:services/video/inpainters/opencv_inpaint.py"], cwd=ROOT,
    )
    engine_name = "_benchmark_opencv_inpaint_baseline"
    engine = types.ModuleType(engine_name)
    sys.modules[engine_name] = engine
    exec(compile(engine_source, f"{ref}:opencv_inpaint.py", "exec"), engine.__dict__)
    module.baseline_inpainter = engine.OpenCVInpainter
    return module


def strip(width, height, text="SOURCE CAPTION LINE", value=65):
    image = np.full((height, width, 3), value, np.uint8)
    scale = max(.7, min(2.0, height / 75))
    y = min(height - 12, round(height * .65))
    x = max(12, round(width * .04))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), max(5, round(scale * 6)))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), max(2, round(scale * 2)))
    return image


def same_features(first, second):
    if first is None or second is None:
        return first is None and second is None
    return np.array_equal(first.core, second.core) and np.array_equal(first.edge, second.edge)


def paired_measure(first, second, repeats):
    timings = [[], []]
    outputs = [None, None]
    for repeat in range(repeats):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            started = time.perf_counter()
            outputs[index] = (first, second)[index]()
            timings[index].append(time.perf_counter() - started)
    old, new = (statistics.median(values) for values in timings)
    return {
        "baseline_seconds": round(old, 6), "worktree_seconds": round(new, 6),
        "speedup": round(old / max(new, 1e-9), 3),
        "baseline_runs": timings[0], "worktree_runs": timings[1],
    }, outputs


def make_video(path, frames, moving=False):
    width, height, roi_height, fps = 960, 540, 144, 25
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot create benchmark video")
    total = frames / fps
    random = np.random.default_rng(20260930)
    cues = [(round(total * .20 * 1000), round(total * .40 * 1000), "SOURCE CAPTION LINE"),
            (round(total * .58 * 1000), round(total * .74 * 1000), "ANOTHER SOURCE LINE")]
    try:
        for index in range(frames):
            value = 65 + index % 15 if moving else 65
            image = np.full((height, width, 3), value, np.uint8)
            image[:height - roi_height] = 35 + index % 30
            timestamp = index / fps
            text = ""
            if total * .10 <= timestamp < total * .46:
                text = cues[0][2]
            elif total * .52 <= timestamp < total * .80:
                text = cues[1][2]
            elif total * .88 <= timestamp < total * .96:
                text = "MISSED SOURCE LINE"
            roi = strip(width, roi_height, text, value=value) if text else image[-roi_height:].copy()
            if moving:
                # Change source pixels, not the caption geometry. No frame can
                # reuse a previous inpaint result merely because text matches.
                background = np.all(roi == value, axis=2)
                values = value + random.integers(-8, 9, (roi_height, width), dtype=np.int16)
                roi[background] = np.repeat(values[background, None], 3, axis=1).astype(np.uint8)
            image[-roi_height:] = roi
            writer.write(image)
    finally:
        writer.release()
    return cues, [(a / 1000, b / 1000) for a, b, _ in cues], (0, height - roi_height, width, roi_height)


def decoded_digest(path):
    digest = hashlib.sha256()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot decode benchmark output: {path}")
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            digest.update(frame.tobytes())
            frames += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Empty benchmark video: {path}")
    return {"sha256": digest.hexdigest(), "frames": frames}


def render_pair(baseline, video, cues, box, folder, repeats):
    # Isolate only this benchmark process before importing application config.
    os.environ["STUDIO_RUNTIME_DIR"] = str(folder / "runtime")
    os.environ["DOUYIN_TIKTOK_OUTBOX_DIR"] = ""
    from services.video import clean_pipeline
    from services.srt_utils import generate_srt
    srt = generate_srt([{"start": a / 1000, "end": b / 1000, "text": text} for a, b, text in cues])
    cap = cv2.VideoCapture(str(video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    x, y, w, h = box
    region = {"x_ratio": x / width, "y_ratio": y / height, "w_ratio": w / width, "h_ratio": h / height}
    paths = [folder / "baseline.mp4", folder / "worktree.mp4"]
    engines = (baseline.baseline_inpainter, clean_pipeline.OpenCVInpainter)
    def run(index):
        module = (baseline, current)[index]
        np.random.seed(20260930)  # Same grain sequence; only the base operation is cached.
        with patch.object(clean_pipeline, "SubtitleTimingGuard", module.SubtitleTimingGuard), \
             patch.object(clean_pipeline, "OpenCVInpainter", engines[index]), \
             patch.object(clean_pipeline, "encoder_args", return_value=["-c:v", "libx264", "-preset", "fast", "-crf", "20"]), \
             redirect_stdout(io.StringIO()):
            return clean_pipeline.clean_video_pipeline(str(video), region, srt, str(paths[index]), None)
    metrics, outputs = paired_measure(lambda: run(0), lambda: run(1), repeats)
    metrics["decoded_outputs"] = [decoded_digest(path) for path in paths]
    metrics["equivalent"] = (metrics["decoded_outputs"][0] == metrics["decoded_outputs"][1] and
                             all(output["frames"] == source_frames for output in metrics["decoded_outputs"]))
    metrics["source_frames"] = source_frames
    metrics["recovered_frames"] = [output["timing_guard"]["recovered_frames"] for output in outputs]
    metrics["inpaint_cache"] = [output.get("inpaint_cache") for output in outputs]
    metrics["encoder"] = "libx264 fast CRF 20; no GPU/provider calls"
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="50891eb")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--video-frames", type=int, default=125)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--moving", action="store_true", help="Change every ROI frame; exercises cache misses")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not (1 <= args.repeats <= 10 and 1 <= args.calls <= 1000 and 50 <= args.video_frames <= 500):
        parser.error("Bounded benchmark: repeats 1..10, calls 1..1000, video-frames 50..500")
    baseline = load_baseline(args.baseline_ref)
    report = {"baseline_ref": args.baseline_ref, "opencv": cv2.__version__, "numpy": np.__version__,
              "opencv_threads": cv2.getNumThreads(), "repeats": args.repeats,
              "calls_per_round": args.calls, "synthetic_only": True,
              "moving_roi": args.moving, "features": []}
    for width, height in ((430, 90), (960, 144), (1920, 216)):
        image = strip(width, height)
        old = baseline._outlined_glyphs(image)
        new = current._outlined_glyphs(image)
        for _ in range(3):
            baseline._outlined_glyphs(image)
            current._outlined_glyphs(image)
        def repeat(module):
            for _ in range(args.calls):
                module._outlined_glyphs(image)
        metrics, _ = paired_measure(lambda: repeat(baseline), lambda: repeat(current), args.repeats)
        metrics.update(roi=f"{width}x{height}", equivalent=same_features(old, new), template_detected=old is not None)
        report["features"].append(metrics)
    with tempfile.TemporaryDirectory(prefix="timing_guard_benchmark_") as temporary:
        folder = Path(temporary)
        video = folder / "source.mp4"
        cues, intervals, box = make_video(video, args.video_frames, moving=args.moving)
        guards = [module.SubtitleTimingGuard.from_video(str(video), cues, box) for module in (baseline, current)]
        metrics, outputs = paired_measure(
            lambda: guards[0].scan_extra_intervals(str(video), intervals, box),
            lambda: guards[1].scan_extra_intervals(str(video), intervals, box), args.repeats,
        )
        templates = [guard.template_count for guard in guards]
        metrics.update(equivalent=outputs[0] == outputs[1] and all(templates), results=outputs,
                       templates=templates,
                       video_frames=args.video_frames, video_size="960x540")
        report["presence_scan"] = metrics
        if args.render:
            report["clean_render"] = render_pair(baseline, video, cues, box, folder, args.repeats)
    report["correctness_passed"] = all(row["equivalent"] and row["template_detected"] for row in report["features"]) and report["presence_scan"]["equivalent"]
    if args.render:
        report["correctness_passed"] &= report["clean_render"]["equivalent"]
    content = json.dumps(report, ensure_ascii=False, indent=2)
    print(content)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content + "\n", encoding="utf-8")
    return 0 if report["correctness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
