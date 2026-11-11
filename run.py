#!/usr/bin/env python3
"""Read a left MCAP stereo view and predict both hands without the source repo."""
import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from mcap_video import HEAD_TOPIC, FrameSampler, inspect_recording, iter_left_frames
from render import render_comparison, contact_sheet
from WiLoR.inference import WiLoRPoseExtractor
from WiLoR.detectors.yolo_world import YOLOWorldDetector
from sequence import TrackingSequence, camera_signature, recording_inputs
from pose_filter import EKFConfig, MANOPoseEKF, regenerate_geometry


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def choose_detections(rows, width, height, min_area_fraction=.001):
    """Select up to two detections, suppressing merged/duplicate glove boxes.

    Score all pairs using the smaller-box intersection penalty: ordinary IoU
    alone often keeps a large box enclosing both hands alongside one real hand.
    """
    candidates = []
    for row in rows:
        box = np.asarray(row[:4], np.float32).copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        area = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
        if area < width*height*min_area_fraction:
            continue
        candidates.append({"bbox": box, "detector_score": float(row[4]), "area": float(area)})
    if not candidates:
        return []
    best = [max(candidates, key=lambda d: d["detector_score"])]
    best_score = best[0]["detector_score"]
    for a, b in itertools.combinations(candidates, 2):
        lo = np.maximum(a["bbox"][:2], b["bbox"][:2])
        hi = np.minimum(a["bbox"][2:], b["bbox"][2:])
        intersection = float(np.prod(np.maximum(0, hi-lo)))
        overlap = intersection / min(a["area"], b["area"])
        score = a["detector_score"] + b["detector_score"] - .8*overlap
        if score > best_score:
            best, best_score = [a, b], score
    return sorted(best, key=lambda d: float((d["bbox"][0]+d["bbox"][2])/2))


def detect_hands(detector, image, *, conf=.12, use_tiles=True):
    h, w = image.shape[:2]
    rows = detector.predict(image, conf=conf, iou=.5)
    selected = choose_detections(rows, w, h)
    used_tiles = use_tiles and len(selected) < 2
    if used_tiles:
        extra = []
        for lo, hi in [(0, w//2), (w//2, w)]:
            boxes = detector.predict(image[:, lo:hi], conf=conf, iou=.5)
            boxes[:, [0, 2]] += lo
            extra.append(boxes)
        selected = choose_detections(np.concatenate([rows, *extra]), w, h)
    return selected, used_tiles


def assign_sides(detections, width, previous=None):
    """Explicitly heuristic labels; never claim anatomical verification."""
    if not detections:
        return []
    centers = [float((d["bbox"][0]+d["bbox"][2])/2)/width for d in detections]
    priors = previous or {0: .25, 1: .75}
    assignments = list(itertools.permutations([0, 1], len(detections)))
    flags = min(assignments, key=lambda flags: sum(abs(c-priors[s]) for c, s in zip(centers, flags)))
    return list(flags)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mcap", type=Path, nargs="+", help="MCAP paths or a dataset/session directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=float, default=None, help="Seconds from MCAP origin; default 3 for independent files, 0 for sequences")
    p.add_argument("--sequence", action="store_true", help="Process numbered files in order and retain tracking across continuous boundaries; implied for directories")
    p.add_argument("--first-recording", type=int, default=1, help="First recording number in each sequence (default 1, excluding recording_000)")
    p.add_argument("--duration", type=float, default=None)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--sample-hz", type=float, default=1.)
    selection.add_argument("--all-frames", action="store_true", help="Export a pose row for every native decoded frame in the requested window")
    p.add_argument("--times", type=float, nargs="+", help="Explicit offsets; overrides start/duration sampling")
    p.add_argument("--max-frames", type=int, default=None, help="Per recording")
    p.add_argument("--topic", default=HEAD_TOPIC)
    p.add_argument("--layout", choices=["side-by-side", "left-only"], default="side-by-side")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--conf", type=float, default=.12)
    p.add_argument("--no-tracking", action="store_true", help="Use the original frame-independent baseline")
    p.add_argument("--detect-every", type=int, default=5, help="Native frames between full-view detections")
    p.add_argument("--track-max-gap", type=float, default=.6, help="Maximum seconds without detector confirmation")
    p.add_argument("--tracking-warmup", type=float, default=.5, help="Track this many seconds before the first exported pose")
    p.add_argument("--pose-filter", choices=["none", "ekf"], default="none", help="Optional causal MANO error-state EKF; also retains predictions_raw.jsonl")
    p.add_argument("--ekf-observation-noise-deg", type=float, default=12.)
    p.add_argument("--ekf-angular-accel-noise-deg", type=float, default=300.)
    p.add_argument("--ekf-max-innovation-deg", type=float, default=75.)
    p.add_argument("--no-detector-tiles", action="store_true", help="Disable half-image fallback in --no-tracking baseline")
    p.add_argument("--mesh", action="store_true", help="Add hand-mesh wireframe to previews")
    p.add_argument("--save-frames", action="store_true", help="Write per-frame input and overlay JPEGs to frames/")
    p.add_argument("--video", action="store_true", help="Write MP4 at sample-hz; use evenly spaced samples")
    p.add_argument("--rectify", action="store_true", help="Apply the recorded left K,D,R,P calibration")
    args = p.parse_args(argv)
    args.sequence = args.sequence or any(path.is_dir() for path in args.mcap)
    if args.start is None:
        args.start = 0. if args.sequence else 3.
    if args.first_recording < 0:
        p.error("first-recording must be nonnegative")
    if args.sequence and (args.no_tracking or args.start != 0 or args.duration is not None or args.times is not None or args.max_frames is not None):
        p.error("Sequence mode requires tracking and complete files from --start 0; omit --duration, --times, and --max-frames")
    if args.sample_hz <= 0 or args.threads < 1 or args.start < 0 or (args.duration is not None and args.duration <= 0):
        p.error("Require sample-hz > 0, threads >= 1, start >= 0, duration > 0")
    if args.max_frames is not None and args.max_frames < 1:
        p.error("max-frames must be positive")
    if args.times is not None and (not all(np.isfinite(args.times)) or min(args.times) < 0):
        p.error("times must be finite and nonnegative")
    if args.video and args.times is not None:
        p.error("Use --start/--duration/--sample-hz for timestamp-consistent video")
    if args.all_frames and args.times is not None:
        p.error("--all-frames cannot be combined with --times")
    if args.all_frames and args.video:
        p.error("--video requires uniform --sample-hz; --all-frames preserves native timestamps and still writes inspection JPEGs")
    if not 0 < args.conf < 1:
        p.error("conf must be between 0 and 1")
    if args.detect_every < 1 or args.track_max_gap <= 0 or args.tracking_warmup < 0:
        p.error("Require detect-every >= 1, track-max-gap > 0, tracking-warmup >= 0")
    if args.no_detector_tiles and not args.no_tracking:
        p.error("--no-detector-tiles applies only to --no-tracking baseline")
    if args.pose_filter == "ekf" and args.no_tracking:
        p.error("--pose-filter ekf requires persistent tracking; omit --no-tracking")
    if args.pose_filter == "ekf" and not args.all_frames and args.times is None and args.sample_hz * args.track_max_gap < 1:
        p.error("EKF samples would exceed the reset gap; use --all-frames or increase --sample-hz")
    try:
        EKFConfig(observation_noise_deg=args.ekf_observation_noise_deg,
                  angular_accel_noise_deg=args.ekf_angular_accel_noise_deg,
                  max_innovation_deg=args.ekf_max_innovation_deg)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    recordings = recording_inputs(args.mcap, sequence=args.sequence, first_recording=args.first_recording)
    cv2.setNumThreads(2)
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.fp16 and device != "cuda":
        raise ValueError("fp16 requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading AnyHand-WiLoR + YOLO-World on {device}", flush=True)
    detector = YOLOWorldDetector(device=device)
    extractor = WiLoRPoseExtractor(device=device, fp16=args.fp16)
    faces = extractor.model.mano.faces.detach().cpu().numpy() if args.mesh else None
    sequence = TrackingSequence(detector, conf=args.conf, detect_every=args.detect_every,
                                max_gap_s=args.track_max_gap) if args.sequence else None
    pose_filter = MANOPoseEKF(EKFConfig(observation_noise_deg=args.ekf_observation_noise_deg,
                                      angular_accel_noise_deg=args.ekf_angular_accel_noise_deg,
                                      max_innovation_deg=args.ekf_max_innovation_deg,
                                      max_gap_s=args.track_max_gap)) if args.pose_filter == "ekf" else None
    all_summaries, overview = [], []
    for recording in recordings:
        path = recording.path
        if pose_filter is not None and sequence is None:
            pose_filter.clear()
        started = time.perf_counter()
        info = inspect_recording(path, args.topic)
        token = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        case = args.output / f"{path.parent.parent.name}_{path.stem}_{token}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "frames").mkdir(exist_ok=True)
        for stale in ("metadata.json", "examples.jpg", "preview.mp4", "predictions_raw.jsonl"):
            (case / stale).unlink(missing_ok=True)
        # Each run owns these outputs; avoid mixing old frame selections.
        for old in (case / "frames").glob("*.jpg"):
            old.unlink()
        info.update({
            "schema_version": 2, "model": "AnyHand-WiLoR", "detector": "YOLO-World (white glove, hand)",
            "handedness": "egocentric position at initialization, then persistent track identity; unverified" if not args.no_tracking else "egocentric position + previous sampled-frame center; heuristic, unverified",
            "detection_strategy": "native-frame visual tracking, gated association and per-track crop recovery" if not args.no_tracking else "full view + optional half-image fallback",
            "tracking_enabled": not args.no_tracking,
            "detect_every_native_frames": args.detect_every,
            "track_max_gap_s": args.track_max_gap,
            "tracking_warmup_s": args.tracking_warmup,
            "pose_temporal_fitting": False,
            "pose_filter": pose_filter.config.metadata() if pose_filter is not None else "none",
            "camera_view": "left", "layout": args.layout, "rectified": args.rectify,
            "coordinate_system": "image x right, y down, camera z forward",
            "geometry": "canonical MANO metric hand scale; monocular root translation is estimated, not stereo-measured",
            "mano_parameter_convention": "canonical right MANO rotations for flipped crops; left geometry is mirrored in x",
            "focal_length": "WiLoR configuration heuristic, not measured intrinsics",
            "timestamps": "integer nanoseconds preserved; offset_s uses MCAP log-time origin",
            "sample_hz": None if args.all_frames else args.sample_hz, "requested_times_s": args.times,
            "frame_selection": "all_native_frames" if args.all_frames else ("requested_times" if args.times is not None else "sample_hz"),
            "device": device, "torch": torch.__version__, "checkpoint_report": extractor.model.checkpoint_report,
            "outputs": {"predictions": "predictions.jsonl", "previews": "frames/*.jpg"},
        })
        if pose_filter is not None:
            info["schema_version"] = 3
            info["outputs"]["raw_predictions"] = "predictions_raw.jsonl"
        frames_meta, examples = [], []
        previous, previous_time = None, None
        writer = None
        remap = None
        tracker = None
        if sequence is not None:
            sequence.begin_recording(recording, camera_signature(info, topic=args.topic, layout=args.layout, rectify=args.rectify))
            tracker = sequence
        elif not args.no_tracking:
            from tracking import HandTracker
            tracker = HandTracker(detector, conf=args.conf, detect_every=args.detect_every,
                                  max_gap_s=args.track_max_gap)
        sampler = FrameSampler(start_s=args.start, sample_hz=args.sample_hz,
                               times=args.times, max_frames=args.max_frames)
        first_target = min(args.times) if args.times else args.start
        tracking_start = max(0., first_target-args.tracking_warmup)
        tracking_end = max(args.times)+.15 if args.times else (args.start+args.duration if args.duration is not None else None)
        native_updates = 0
        tracking_totals = {}
        with ExitStack() as stack:
            sink = stack.enter_context((case / "predictions.jsonl").open("w"))
            raw_sink = stack.enter_context((case / "predictions_raw.jsonl").open("w")) if pose_filter is not None else None
            try:
                if tracker is not None:
                    stream = iter_left_frames(path, start_s=tracking_start,
                        duration_s=tracking_end-tracking_start if tracking_end is not None else None,
                        topic=args.topic, layout=args.layout, all_frames=True)
                else:
                    stream = iter_left_frames(path, start_s=0 if args.times else args.start,
                        duration_s=None if args.times else args.duration, sample_hz=args.sample_hz,
                        times=args.times, max_frames=args.max_frames, topic=args.topic, layout=args.layout,
                        all_frames=args.all_frames)
                for frame in stream:
                    image = frame.image
                    h, w = image.shape[:2]
                    if args.rectify:
                        if remap is None:
                            c = info["left_calibration"]
                            if not c or (c["width"], c["height"]) != (w, h):
                                raise ValueError("Recorded left calibration does not match decoded left view")
                            if c.get("distortionModel") not in {"rational_polynomial", "plumb_bob"}:
                                raise ValueError("Unsupported distortion model")
                            remap = cv2.initUndistortRectifyMap(np.array(c["K"]).reshape(3,3), np.array(c["D"]),
                                np.array(c["R"]).reshape(3,3), np.array(c["P"]).reshape(3,4)[:,:3], (w,h), cv2.CV_32FC1)
                        image = cv2.remap(image, *remap, cv2.INTER_LINEAR)
                    if tracker is not None:
                        if sequence is not None:
                            detections, track_diagnostics, sequence_context = sequence.update(image, frame.log_time_ns)
                        else:
                            detections, track_diagnostics = tracker.update(image, frame.offset_s)
                        native_updates += 1
                        for key, value in track_diagnostics.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                tracking_totals[key] = tracking_totals.get(key, 0)+value
                        if args.all_frames and frame.offset_s + 1e-9 < args.start:
                            continue
                        if not args.all_frames and not sampler.select(frame.offset_s):
                            continue
                        right = [d["right"] for d in detections]
                        tiled = False
                    else:
                        detections, tiled = detect_hands(detector, image, conf=args.conf, use_tiles=not args.no_detector_tiles)
                        if previous_time is not None and frame.offset_s - previous_time > 1.:
                            previous = None
                        right = assign_sides(detections, w, previous)
                        track_diagnostics = {}
                    i = len(frames_meta)
                    hands = extractor.predict_bboxes(image, np.array([d["bbox"] for d in detections]).reshape(-1,4),
                        np.array(right), batch_size=2 if device == "cuda" else 1, rescale_factor=2.2)
                    for hand, detection in zip(hands, detections):
                        hand["detector_score"] = detection["detector_score"]
                        hand["handedness_verified"] = False
                        for field in ("track_id", "track_source", "seconds_since_detection", "raw_bbox", "crop_bbox"):
                            if field in detection:
                                hand[field] = detection[field]
                        hand["joints_camera_3d"] = hand["joints_3d"] + hand["cam_t"][None, :]
                        hand["vertices_camera_3d"] = hand["vertices"] + hand["cam_t"][None, :]
                        if not all(np.isfinite(np.asarray(v)).all() for v in hand.values() if isinstance(v, (np.ndarray, float))):
                            raise ValueError("Nonfinite model output")
                    raw_hands = hands
                    if pose_filter is not None:
                        hands = pose_filter.filter_hands(hands, frame.log_time_ns)
                        regenerate_geometry(hands, extractor.model.mano, w, h)
                    previous = {int(hand["right"]): float((hand["bbox"][0]+hand["bbox"][2])/2)/w for hand in hands}
                    previous = {0: previous.get(0, .25), 1: previous.get(1, .75)}
                    previous_time = frame.offset_s
                    stem = f"{i:06d}_{frame.offset_s:09.3f}s"
                    no_preview = not args.save_frames
                    if not no_preview:
                        cv2.imwrite(str(case / "frames" / f"{stem}_left.jpg"), image)
                    title = f"{path.parent.parent.name} | t={frame.offset_s:.3f}s | packet {frame.packet_index}"
                    if pose_filter is not None:
                        title += " | causal EKF"
                    preview = None
                    if not no_preview or i < 3 or args.video:
                        preview = render_comparison(image, hands, title, mesh=args.mesh, faces=faces)
                    if not no_preview:
                        cv2.imwrite(str(case / "frames" / f"{stem}_overlay.jpg"), preview)
                    if i < 3:
                        examples.append(preview)
                    if i == 0:
                        overview.append(preview)
                    if args.video:
                        if writer is None:
                            writer = cv2.VideoWriter(str(case / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                args.sample_hz, (preview.shape[1], preview.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError("Could not open MP4 writer")
                        writer.write(preview)
                    row = {"frame_index": i, "packet_index": frame.packet_index, "offset_s": frame.offset_s,
                        "log_time_ns": frame.log_time_ns, "publish_time_ns": frame.publish_time_ns,
                        "capture_time_ns": frame.capture_time_ns, "image_width": w, "image_height": h,
                        "stereo_width": frame.stereo_width, "stereo_height": frame.stereo_height,
                        "detector_half_view_fallback": tiled,
                        "tracking_diagnostics": track_diagnostics,
                        "image": f"frames/{stem}_left.jpg", "overlay": f"frames/{stem}_overlay.jpg",
                        "num_hands": len(hands), "hands": hands}
                    if sequence is not None:
                        row.update(sequence_context)
                    if raw_sink is not None:
                        raw_sink.write(json.dumps({**row, "hands": raw_hands}, default=serializable, allow_nan=False) + "\n")
                        raw_sink.flush()
                        row["pose_filter"] = "ekf"
                    sink.write(json.dumps(row, default=serializable, allow_nan=False) + "\n")
                    sink.flush()
                    frames_meta.append({k:v for k,v in row.items() if k != "hands"})
                    print(f"{path.parent.parent.name} {frame.offset_s:.3f}s: {len(hands)} hands", flush=True)
                    if args.max_frames is not None and len(frames_meta) >= args.max_frames:
                        break
                    if tracker is not None and not args.all_frames and sampler.done:
                        break
            finally:
                if writer is not None:
                    writer.release()
        if not frames_meta:
            raise ValueError(f"No frames matched the requested window in {path}")
        if args.video and shutil.which("ffmpeg"):
            temporary = case / "preview.h264.tmp.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(case / "preview.mp4"), "-c:v", "libx264", "-preset", "fast",
                "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], check=True)
            os.replace(temporary, case / "preview.mp4")
            info["preview_codec"] = "H.264/yuv420p"
        elif args.video:
            info["preview_codec"] = "MPEG-4 Part 2 (install ffmpeg for browser-compatible H.264)"
        info.update(frames=frames_meta, frames_processed=len(frames_meta), hands_predicted=sum(f["num_hands"] for f in frames_meta), elapsed_s=time.perf_counter()-started,
                    native_tracking_updates=native_updates, tracking_counters=tracking_totals)
        if sequence is not None:
            info["sequence"] = {**sequence.boundary, "last_video_log_time_ns": sequence.last_timestamp_ns}
        (case / "metadata.json").write_text(json.dumps(info, indent=2, default=serializable, allow_nan=False)+"\n")
        if examples:
            cv2.imwrite(str(case / "examples.jpg"), contact_sheet(examples))
        all_summaries.append({"directory": str(case.resolve()), "frames": len(frames_meta), "hands": info["hands_predicted"]})
    if overview:
        cv2.imwrite(str(args.output / "overview.jpg"), contact_sheet(overview))
    (args.output / "summary.json").write_text(json.dumps(all_summaries, indent=2)+"\n")
    print(json.dumps(all_summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
