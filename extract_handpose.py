#!/usr/bin/env python3
"""End-to-end hand keypoints for one EgoForce recording.

Input is an EgoForce recording directory (``head.mp4`` + ``timestamps.json``),
a raw ``.mcap`` file, or (batch mode) any folder containing recording
directories, e.g. a date folder; batch outputs mirror the input tree as
``<output>/<task>/<collector>/recording_NNN/handpose.npz`` and existing
outputs are skipped. The pipeline is:

  YOLO-World glove/hand detection on every pose frame
  -> offline left/right labeling of the whole recording (labeling.py)
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
    p.add_argument("--pose-hz", type=float, default=10.0, help="Pose estimation rate; 0 = every frame")
    p.add_argument("--upscale", type=float, default=2.0, help="MP4 only: left-view upscale before inference")
    p.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames (testing)")
    p.add_argument("--save-predictions", type=Path, default=None, help="Single mode: also write raw per-pose predictions (.jsonl.gz)")
    p.add_argument("--overwrite", action="store_true", help="Batch mode: recompute recordings whose output exists")
    return p.parse_args(argv)


def arm_check(preds, frame_wh):
    """Consistency of left/right labels with the forearm direction.

    The forearm continues the wrist beyond the hand (middle knuckle to wrist).
    Extended to below the image, the right forearm should end to the right of
    the left forearm. Returns the share of two-hand pose frames where it does
    not, by more than a tenth of the image width.
    """
    w, h = frame_wh
    checked = disagree = 0
    for _, row in preds:
        entry = {}
        for hand in row["hands"]:
            kp = np.asarray(hand["keypoints_2d"], np.float64)
            v = kp[0] - kp[9]
            if v[1] <= 0.3 * np.linalg.norm(v):
                continue
            entry[int(hand["right"])] = kp[0][0] + v[0] * (1.3 * h - kp[0][1]) / v[1]
        if len(entry) == 2:
            checked += 1
            disagree += int(entry[0] > entry[1] + 0.1 * w)
    return {"two_hand_frames": checked, "disagree": disagree,
            "disagree_rate": round(disagree / checked, 4) if checked else None}


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
        from video_input import iter_mp4_frames, load_frame_times
        import labeling
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
            scale = 1.0 / args.upscale

            def frames(upscale, keep=None):
                for f in iter_mp4_frames(mp4, frame_times, upscale=upscale, max_frames=args.max_frames, keep=keep):
                    yield f.packet_index, f
            det_scale = args.upscale
        elif src.suffix.lower() == ".mcap":
            frame_times = None
            scale = 1.0

            def frames(upscale, keep=None):
                it = iter_left_frames(src, start_s=0.0, all_frames=True, max_frames=args.max_frames)
                for k, f in enumerate(it):
                    if keep is None or k in keep:
                        yield k, f
            det_scale = 1.0
        else:
            raise ValueError("Input must be a recording directory or an .mcap file")

        started = time.perf_counter()
        # Pass 1: candidate boxes on every pose frame (native left view for MP4;
        # the detector letterboxes to 640 px either way).
        sampler = FrameSampler(start_s=0.0, sample_hz=args.pose_hz) if args.pose_hz > 0 else None
        keys, times, cand_boxes, cand_scores, cand_whites = [], [], [], [], []
        seen_times, frame_wh, stereo_wh, last_t = [], None, None, None
        for key, frame in frames(1.0):
            seen_times.append(frame.offset_s)
            frame_wh = (int(round(frame.image.shape[1] * det_scale)), int(round(frame.image.shape[0] * det_scale)))
            stereo_wh = (frame.stereo_width, frame.stereo_height)
            # Constant-rate MP4s repeat a picture (same timestamp) where the
            # camera dropped a frame; keep strictly increasing times.
            if last_t is not None and frame.offset_s <= last_t:
                continue
            last_t = frame.offset_s
            if sampler is not None and not sampler.select(frame.offset_s):
                continue
            boxes, scores, whites = labeling.detect_candidates(self.detector, frame.image, scale=det_scale)
            keys.append(key); times.append(frame.offset_s)
            cand_boxes.append(boxes); cand_scores.append(scores); cand_whites.append(whites)
        if not seen_times:
            raise ValueError(f"No frames decoded from {src}")
        lab = labeling.label_hands(np.asarray(times), cand_boxes, cand_scores, width=frame_wh[0], whites=cand_whites)

        # Pass 2: pose for the labeled boxes.
        wanted = {key: k for k, key in enumerate(keys)}
        preds = []
        for key, frame in frames(args.upscale if src.is_dir() else 1.0, keep=set(wanted)):
            k = wanted.get(key)
            if k is None:
                continue
            boxes, right, score = [], [], []
            for side, box, sc in ((0, lab.left[k], lab.left_score[k]), (1, lab.right[k], lab.right_score[k])):
                if box is not None:
                    boxes.append(box); right.append(side); score.append(sc)
            hands = self.extractor.predict_bboxes(frame.image, np.array(boxes, np.float32).reshape(-1, 4), np.array(right),
                                                  batch_size=2 if self.device == "cuda" else 1, rescale_factor=2.2)
            for hand, sc in zip(hands, score):
                hand["detector_score"] = sc
            preds.append((frame.offset_s, {"offset_s": frame.offset_s, "hands": hands}))

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
                "labeling": lab.stats, "arm_check": arm_check(preds, frame_wh), "stabilizer": stats, "output": str(output)}


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
