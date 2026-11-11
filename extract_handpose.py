#!/usr/bin/env python3
"""End-to-end hand keypoints for one EgoForce recording.

Input is an EgoForce recording directory (``head.mp4`` + ``timestamps.json``),
a raw ``.mcap`` file, or (batch mode) any folder containing recording
directories, e.g. a date folder; batch outputs mirror the input tree as
``<output>/<task>/<collector>/recording_NNN/handpose.npz`` and existing
outputs are skipped. The pipeline is:

  YOLO-World glove/hand detection + CSRT tracking on every frame
  -> AnyHand-WiLoR (MANO) pose at ``--pose-hz``
  -> zero-phase stabilization (stabilize.py)
  -> one npz with a row per video frame.

Output npz (N = number of video frames):
  frame_t     float32 (N,)        frame time in seconds (timestamps.json "head")
  L_xy, R_xy  float16 (N,21,2)    2D joints in pixels of the EgoForce head.mp4 frame
  L_mm, R_mm  float16 (N,21,3)    3D joints in millimeters, relative to the wrist root
  L_ok, R_ok  bool    (N,)        hand present and estimated on that frame
  scale       float32 ()          factor applied to inference pixels to get *_xy
  resized_wh  int32   (2,)        width, height of the frame *_xy refers to

For MP4 input the joints lie in the left half of the 1920x600 side-by-side
frame (x in [0, 960), y in [0, 600)). Unestimated frames are zero with *_ok False.
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="Recording directory (head.mp4 + timestamps.json), an .mcap file, or any folder "
                        "containing recording directories (batch mode)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output .npz (single recording) or output root directory (batch mode)")
    p.add_argument("--weights", type=Path, default=None, help="Model asset directory (default: weights/ or $HANDPOSE_WEIGHTS)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--fp16", action="store_true", help="Half precision pose model (CUDA only)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.12, help="Detector confidence threshold")
    p.add_argument("--detect-every", type=int, default=5, help="Frames between full-view detections")
    p.add_argument("--track-max-gap", type=float, default=0.6, help="Seconds a track may go without detector confirmation")
    p.add_argument("--reacquire", type=float, default=2.0, help="Seconds after which an unconfirmed hand slot is released")
    p.add_argument("--pose-hz", type=float, default=10.0, help="Pose estimation rate; 0 = every frame")
    p.add_argument("--upscale", type=float, default=2.0, help="MP4 only: left-view upscale before inference")
    p.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames (testing)")
    p.add_argument("--save-predictions", type=Path, default=None, help="Single mode: also write raw per-pose predictions (.jsonl.gz)")
    p.add_argument("--overwrite", action="store_true", help="Batch mode: recompute recordings whose output exists")
    return p.parse_args(argv)


def find_recordings(root):
    """Recording directories (containing head.mp4 and timestamps.json) below root, sorted."""
    return sorted(m.parent for m in Path(root).rglob("head.mp4") if (m.parent / "timestamps.json").is_file())


class Pipeline:
    def __init__(self, args):
        if args.weights is not None:
            os.environ["HANDPOSE_WEIGHTS"] = str(args.weights.expanduser().resolve())
        import cv2
        import torch
        from WiLoR.inference import WiLoRPoseExtractor
        from WiLoR.detectors.yolo_world import YOLOWorldDetector

        cv2.setNumThreads(2)
        torch.set_num_threads(args.threads)
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if args.fp16 and device != "cuda":
            raise ValueError("--fp16 requires CUDA")
        print(f"Loading models on {device}", flush=True)
        self.args, self.device = args, device
        self.detector = YOLOWorldDetector(device=device)
        self.extractor = WiLoRPoseExtractor(device=device, fp16=args.fp16)

    def run(self, src, output, save_predictions=None):
        from mcap_video import FrameSampler, iter_left_frames
        from tracking import HandTracker
        from video_input import iter_mp4_frames, load_frame_times
        import stabilize

        args = self.args
        src = Path(src)
        if src.is_dir():
            mp4 = src / "head.mp4"
            if not mp4.is_file():
                raise FileNotFoundError(mp4)
            frame_times = load_frame_times(src)
            if args.max_frames is not None:
                frame_times = frame_times[:args.max_frames]
            frames = iter_mp4_frames(mp4, frame_times, upscale=args.upscale, max_frames=args.max_frames)
            scale = 1.0 / args.upscale
        elif src.suffix.lower() == ".mcap":
            frame_times = None
            frames = iter_left_frames(src, start_s=0.0, all_frames=True, max_frames=args.max_frames)
            scale = 1.0
        else:
            raise ValueError("Input must be a recording directory or an .mcap file")

        tracker = HandTracker(self.detector, conf=args.conf, detect_every=args.detect_every,
                              max_gap_s=args.track_max_gap, reacquire_s=args.reacquire)
        sampler = FrameSampler(start_s=0.0, sample_hz=args.pose_hz) if args.pose_hz > 0 else None
        preds, seen_times = [], []
        frame_wh, stereo_wh = None, None
        last_t = None
        started = time.perf_counter()
        for frame in frames:
            seen_times.append(frame.offset_s)
            frame_wh = (frame.image.shape[1], frame.image.shape[0])
            stereo_wh = (frame.stereo_width, frame.stereo_height)
            # Constant-rate MP4s repeat a picture (same timestamp) where the camera
            # dropped a frame; the tracker needs strictly increasing times.
            if last_t is not None and frame.offset_s <= last_t:
                continue
            last_t = frame.offset_s
            detections, _ = tracker.update(frame.image, frame.offset_s)
            if sampler is not None and not sampler.select(frame.offset_s):
                continue
            boxes = np.array([d["bbox"] for d in detections], np.float32).reshape(-1, 4)
            right = np.array([d["right"] for d in detections])
            hands = self.extractor.predict_bboxes(frame.image, boxes, right,
                                                  batch_size=2 if self.device == "cuda" else 1, rescale_factor=2.2)
            for hand, det in zip(hands, detections):
                hand["detector_score"] = det["detector_score"]
                hand["track_id"] = det["track_id"]
            preds.append((frame.offset_s, {"offset_s": frame.offset_s, "hands": hands}))
        if not seen_times:
            raise ValueError(f"No frames decoded from {src}")

        if frame_times is None:
            frame_times = np.asarray(seen_times, np.float64)
        frame_times = np.asarray(frame_times[:len(seen_times)], np.float64)
        stab, stats = stabilize.build(preds, frame_times)
        arrs = stabilize.to_arrays(stab, frame_times, scale=scale)
        resized_wh = stereo_wh if src.is_dir() else frame_wh
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(output.stem + ".partial.npz")
        np.savez_compressed(tmp, frame_t=frame_times.astype(np.float32), scale=np.float32(scale),
                            resized_wh=np.asarray(resized_wh, np.int32), **arrs)
        os.replace(tmp, output)

        if save_predictions is not None:
            def plain(v):
                return v.tolist() if isinstance(v, np.ndarray) else (v.item() if isinstance(v, np.generic) else v)
            keep = ("right", "track_id", "detector_score", "keypoints_2d", "joints_3d", "cam_t", "bbox")
            with gzip.open(save_predictions, "wt") as f:
                for _, row in preds:
                    hands = [{k: plain(v) for k, v in h.items() if k in keep} for h in row["hands"]]
                    f.write(json.dumps({"offset_s": row["offset_s"], "hands": hands}) + "\n")

        return {"input": str(src), "frames": len(frame_times), "poses": len(preds),
                "elapsed_s": round(time.perf_counter() - started, 1),
                "coverage_L": round(float(arrs["L_ok"].mean()), 4), "coverage_R": round(float(arrs["R_ok"].mean()), 4),
                "stabilizer": stats, "output": str(output)}


def main(argv=None):
    args = parse_args(argv)
    src = args.input
    single = src.suffix.lower() == ".mcap" or (src / "head.mp4").is_file()
    if single:
        summary = Pipeline(args).run(src, args.output, args.save_predictions)
        print(json.dumps(summary, indent=2), flush=True)
        return
    recordings = find_recordings(src)
    if not recordings:
        raise FileNotFoundError(f"No recording directories (head.mp4 + timestamps.json) under {src}")
    todo = []
    for rec in recordings:
        out = args.output / rec.relative_to(src) / "handpose.npz"
        if args.overwrite or not out.is_file():
            todo.append((rec, out))
    print(f"{len(recordings)} recordings found, {len(todo)} to process", flush=True)
    if not todo:
        return
    pipeline = Pipeline(args)
    failed = 0
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "summary.jsonl", "a") as log:
        for k, (rec, out) in enumerate(todo, 1):
            try:
                summary = pipeline.run(rec, out)
                log.write(json.dumps({k: v for k, v in summary.items() if k != "stabilizer"}) + "\n")
                log.flush()
                print(f"[{k}/{len(todo)}] {rec.relative_to(src)}  L {summary['coverage_L']:.1%}  R {summary['coverage_R']:.1%}"
                      f"  {summary['elapsed_s']:.0f} s", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[{k}/{len(todo)}] {rec.relative_to(src)}  FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"done: {len(todo) - failed} ok, {failed} failed; summaries in {args.output / 'summary.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
