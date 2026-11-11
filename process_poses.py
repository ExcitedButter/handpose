#!/usr/bin/env python3
"""Apply a causal MANO EKF to existing exports without rerunning image inference."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from pose_filter import EKFConfig, MANOPoseEKF, regenerate_geometry
from render import render_comparison
from WiLoR.models.mano_wrapper import MANO


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def discover_cases(inputs):
    cases = {}
    for source in inputs:
        source = Path(source).resolve()
        paths = [source / "predictions.jsonl"] if (source / "predictions.jsonl").is_file() else sorted(source.rglob("predictions.jsonl"))
        for predictions in paths:
            if predictions.parent in cases:
                continue
            metadata = json.loads(predictions.with_name("metadata.json").read_text())
            if metadata.get("pose_filter") not in (None, "none"):
                raise ValueError(f"Input is already pose-filtered: {predictions}")
            with predictions.open() as stream:
                first = json.loads(next(stream))
            cases[predictions.parent] = (predictions.parent, metadata, first["log_time_ns"])
    if not cases:
        raise ValueError("No raw predictions.jsonl + metadata.json cases found")
    return sorted(cases.values(), key=lambda case: (case[2], str(case[0])))


def process_cases(inputs, output, *, config=None, render=True, mano_dir=None, threads=4):
    cases = discover_cases(inputs)
    output = Path(output).resolve()
    if any(output == case or output in case.parents or case in output.parents for case, _, _ in cases):
        raise ValueError("Use a separate output directory outside the input cases")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if len({case.name for case, _, _ in cases}) != len(cases):
        raise ValueError("Input case names collide; process them in separate output directories")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    cv2.setNumThreads(2)
    from WiLoR.assets import weights_dir
    model_path = mano_dir or weights_dir() / "mano"
    mano = MANO(model_path=str(model_path), hand_side="right").eval()
    pose_filter = MANOPoseEKF(config)
    summaries, previous_source = [], None
    previous_metadata = None
    for case, metadata, _ in cases:
        boundary = metadata.get("sequence") or {}
        # Only explicitly continuous, matching sequence boundaries share state.
        continuous = (previous_source is not None and boundary.get("continued_from_previous_file") is True
                      and boundary.get("previous_source_mcap") == previous_source
                      and boundary.get("tracking_segment_id") == (previous_metadata.get("sequence") or {}).get("tracking_segment_id")
                      and all(metadata.get(k) == previous_metadata.get(k) for k in ("layout", "rectified", "camera_view")))
        if not continuous:
            pose_filter.clear()
        destination = output / case.name
        destination.mkdir()
        (destination / "frames").mkdir()
        shutil.copy2(case / "predictions.jsonl", destination / "predictions_raw.jsonl")
        frames, counts, initialized = [], [], 0
        started = time.perf_counter()
        with ExitStack() as stack:
            source = stack.enter_context((case / "predictions.jsonl").open())
            sink = stack.enter_context((destination / "predictions.jsonl").open("w"))
            for line in source:
                row = json.loads(line)
                row["hands"] = pose_filter.filter_hands(row["hands"], row["log_time_ns"])
                regenerate_geometry(row["hands"], mano, row["image_width"], row["image_height"])
                for hand in row["hands"]:
                    counts.append(len(hand["pose_filter"]["rejected_rotation_indices"]))
                    initialized += int(hand["pose_filter"]["initialized"])
                row["pose_filter"] = "ekf"
                if render:
                    image_path = case / row["image"]
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise FileNotFoundError(image_path)
                    image_target = destination / row["image"]
                    image_target.parent.mkdir(parents=True, exist_ok=True)
                    link_or_copy(image_path, image_target)
                    preview = render_comparison(image, row["hands"], f"Causal MANO EKF | t={row['offset_s']:.3f}s")
                    overlay_target = destination / row["overlay"]
                    overlay_target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(overlay_target), preview):
                        raise OSError(f"Could not write {overlay_target}")
                else:
                    # Do not retain image references relative to another case.
                    row["image"] = str((case / row["image"]).resolve())
                    row["overlay"] = None
                sink.write(json.dumps(row, default=serializable, allow_nan=False) + "\n")
                frames.append({k: v for k, v in row.items() if k != "hands"})
        filtered_metadata = dict(metadata)
        filtered_metadata.update(schema_version=3, pose_filter=pose_filter.config.metadata(),
                                 pose_temporal_fitting=False, frames=frames,
                                 raw_export=str(case), filtering_elapsed_s=time.perf_counter() - started,
                                 pose_filter_continued_from_previous_file=continuous,
                                 outputs={"predictions": "predictions.jsonl", "raw_predictions": "predictions_raw.jsonl",
                                          "previews": "frames/*_overlay.jpg" if render else None})
        (destination / "metadata.json").write_text(json.dumps(filtered_metadata, indent=2, allow_nan=False) + "\n")
        summaries.append({"directory": str(destination), "frames": len(frames), "hand_initializations": initialized,
                          "rejected_rotation_observations": sum(counts),
                          "filtering_elapsed_s": filtered_metadata["filtering_elapsed_s"]})
        print(json.dumps(summaries[-1]), flush=True)
        previous_source, previous_metadata = metadata.get("source_mcap"), metadata
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Raw result case directories or a result root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true", help="Write parameters/geometry without inspection JPEGs")
    parser.add_argument("--mano-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--observation-noise-deg", type=float, default=12.)
    parser.add_argument("--angular-accel-noise-deg", type=float, default=300.)
    parser.add_argument("--max-innovation-deg", type=float, default=75.)
    parser.add_argument("--max-gap", type=float, default=.6)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    config = EKFConfig(observation_noise_deg=args.observation_noise_deg,
                       angular_accel_noise_deg=args.angular_accel_noise_deg,
                       max_innovation_deg=args.max_innovation_deg, max_gap_s=args.max_gap)
    process_cases(args.inputs, args.output, config=config, render=not args.no_render,
                  mano_dir=args.mano_dir, threads=args.threads)


if __name__ == "__main__":
    main()
