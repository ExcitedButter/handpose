#!/usr/bin/env python3
"""Adapt WiLoR's hand detector to a set of videos, using the hands run_nwbench.py already found in them.

The detector shipped with WiLoR (weights/hand_detector/detector.pt, a YOLOv8 pose model with the
classes left / right and 21 keypoints) scores hands in thin plastic gloves, close to the camera or
smeared by motion, so low that many of them never become a candidate box. run_nwbench.py still
finds a good part of them: it follows a hand through detector misses and checks every pose. Those
verified hands are training data the detector has not seen. This script

  1. takes, from a result folder of run_nwbench.py, evenly spaced frames in which both hands were
     found, with the box and the 21 keypoints of each hand as labels. Frames with one hand are left
     out: the other hand may be visible but missed, and would be learned as background. Hands of
     bystanders carry no label, so the detector also learns to leave them alone;
  2. fine-tunes the detector on these frames for a few epochs;
  3. writes the new weights, to be used with ``run_nwbench.py --detector``.

    python train_detector.py data/NW_bench results --out weights/hand_detector/detector_nwbench.pt

The labels are the pipeline's own output, not ground truth: the detector cannot learn a hand the
pipeline never finds, and results on the videos it was trained on say little about new videos.
``--exclude`` keeps samples out of the training set for such a check. Needs ``pip install polars``
(ultralytics reads its training log with it).
"""
import argparse
import shutil
import sys
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hand_detector import DEFAULT_MODEL  # noqa: E402
from run_nwbench import find_samples  # noqa: E402

BOX_MARGIN = 0.15     # box = keypoint extent grown on every side by this share of its longer side
IMAGE_SIDE = 960      # longer side of the stored training images


def export(root, results, out, per_sample, exclude):
    """Write images/ and labels/ (ultralytics pose format) under ``out``; returns the data.yaml path."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    images = 0
    for name, mp4, _, _ in find_samples(root):
        tag = name.replace("/", "_")
        if name in exclude or not (results / tag / "handpose.npz").is_file():
            continue
        z = np.load(results / tag / "handpose.npz")
        width, height = (float(v) for v in z["image_wh"])
        both = np.flatnonzero(z["L_valid"] & z["R_valid"])
        take = set(both[np.linspace(0, len(both) - 1, min(per_sample, len(both))).round().astype(int)].tolist()) if len(both) else set()
        with av.open(str(mp4)) as c:
            for i, frame in enumerate(c.decode(c.streams.video[0])):
                if i not in take:
                    continue
                rows = []
                for cls, side in ((0, "L"), (1, "R")):
                    kp = z[f"{side}_kp2d"][i]
                    lo, hi = kp.min(0), kp.max(0)
                    m = BOX_MARGIN * (hi - lo).max()
                    x0, y0 = max(lo[0] - m, 0), max(lo[1] - m, 0)
                    x1, y1 = min(hi[0] + m, width - 1), min(hi[1] + m, height - 1)
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    kps = " ".join(f"{x / width:.5f} {y / height:.5f} 2" if 0 <= x < width and 0 <= y < height else "0 0 0"
                                   for x, y in kp)
                    rows.append(f"{cls} {(x0 + x1) / 2 / width:.5f} {(y0 + y1) / 2 / height:.5f} "
                                f"{(x1 - x0) / width:.5f} {(y1 - y0) / height:.5f} {kps}")
                if len(rows) < 2:
                    continue
                split = "val" if images % 10 == 0 else "train"
                img = frame.to_ndarray(format="bgr24")
                k = IMAGE_SIDE / max(img.shape[:2])
                if k < 1:
                    img = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / "images" / split / f"{tag}_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (out / "labels" / split / f"{tag}_{i:05d}.txt").write_text("\n".join(rows) + "\n")
                images += 1
        print(f"{name}: {len(both)} frames with both hands, {len(take)} taken", flush=True)
    if not images:
        raise SystemExit("No frame with both hands found: nothing to train on.")
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nkpt_shape: [21, 3]\n"
                    "names:\n  0: left\n  1: right\n")
    print(f"{images} images in {out}", flush=True)
    return yaml


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="NW_bench folder (the videos)")
    p.add_argument("results", type=Path, help="Result folder of run_nwbench.py for these videos")
    p.add_argument("--out", type=Path, required=True, help="Weights file to write")
    p.add_argument("--base", type=Path, default=Path(DEFAULT_MODEL), help="Detector to start from")
    p.add_argument("--work", type=Path, default=None, help="Folder for the training set and the training run "
                   "(default: <out>.work next to the weights; removed afterwards unless --keep)")
    p.add_argument("--exclude", nargs="*", default=[], help="Sample names left out of the training set")
    p.add_argument("--per-sample", type=int, default=200, help="Most frames taken from one sample")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--keep", action="store_true", help="Keep the work folder")
    a = p.parse_args()

    work = a.work or a.out.with_suffix(".work")
    yaml = export(a.root, a.results, work / "set", a.per_sample, set(a.exclude))
    from ultralytics import YOLO
    # No left-right flip: it would turn a left hand into a right one under the same label. A small learning
    # rate: the detector is adapted, not retrained. amp and plots are off because they fetch files from the web.
    YOLO(str(a.base)).train(data=str(yaml), epochs=a.epochs, imgsz=640, batch=24, device=a.device, workers=4, amp=False,
                            plots=False, optimizer="SGD", lr0=0.002, lrf=0.2, warmup_epochs=1, fliplr=0.0, mosaic=0.5,
                            close_mosaic=2, project=str((work / "run").resolve()), name="detector", exist_ok=True,
                            verbose=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "run" / "detector" / "weights" / "last.pt", a.out)
    if not a.keep:
        shutil.rmtree(work)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
