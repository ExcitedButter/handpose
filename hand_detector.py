"""Hand box detector: the YOLO hand detector shipped with WiLoR, run at several input sizes.

The YOLO-World detector of the EgoForce pipeline is prompted for white gloves
and scores bare hands seen from a head camera below 0.1. WiLoR's own detector
(``weights/hand_detector/detector.pt``) is trained on hands. It is sensitive
to the size of the hand in its input: a hand that fills a third of a close-up
frame is found at a small input size and missed at a large one, a distant hand
the other way round. The image is therefore passed through the network at
several sizes and the boxes are merged; the score of a merged box is the mean
of its confidences over all sizes (0 where a size does not find it), so a box
seen at one size only scores low.

The weights are an ultralytics checkpoint (a pickle that also needs ``dill``):
load only a file you trust. ``$HANDPOSE_HAND_DETECTOR`` selects another file.
"""
import os

import numpy as np
import torch

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "hand_detector", "detector.pt")
SIZES = (320, 416, 512, 640)   # network input sizes (longer image side, pixels)
FLOOR = 0.02                   # lowest per-size confidence that enters the merge
MATCH_IOU = 0.5                # boxes of different sizes overlapping this much are the same hand
VERIFY_SIZE = 384              # input size for score_at()
VERIFY_IOU = 0.2               # overlap with the queried box that makes a detection count for it


def _iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


class HandDetector:
    def __init__(self, model=None, device="cuda", sizes=SIZES):
        from ultralytics import YOLO
        self.model = YOLO(model or os.environ.get("HANDPOSE_HAND_DETECTOR", DEFAULT_MODEL))
        self.device = str(device)
        self.sizes = tuple(sizes)

    @torch.inference_mode()
    def predict(self, image_bgr, conf=0.05):
        """Rows ``x1, y1, x2, y2, score, p_right`` in image pixels, best first.

        ``score`` is the merged confidence described in the module docstring; ``p_right`` the
        confidence-weighted share of the detector's "right hand" votes for the box.
        """
        found = []
        for k, size in enumerate(self.sizes):
            b = self.model.predict(image_bgr, conf=FLOOR, imgsz=size, device=self.device, verbose=False)[0].boxes
            for box, s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
                found.append((float(s), k, box.astype(np.float64), float(c)))
        found.sort(key=lambda t: -t[0])
        clusters = []   # members {size index: (score, box, class)}, running mean box
        for s, k, box, c in found:
            for cl in clusters:
                if k not in cl[0] and _iou(box, cl[1]) >= MATCH_IOU:
                    cl[0][k] = (s, box, c)
                    w = np.array([m[0] for m in cl[0].values()])
                    cl[1] = (np.stack([m[1] for m in cl[0].values()]) * w[:, None]).sum(0) / w.sum()
                    break
            else:
                clusters.append([{k: (s, box, c)}, box])
        rows = []
        for members, box in clusters:
            w = np.array([m[0] for m in members.values()])
            score = float(w.sum()) / len(self.sizes)
            if score >= conf:
                rows.append([*box, score, float((w * np.array([m[2] for m in members.values()])).sum() / w.sum())])
        rows.sort(key=lambda r: -r[4])
        return np.asarray(rows, np.float32).reshape(-1, 6)

    @torch.inference_mode()
    def score_at(self, image_bgr, box):
        """Highest confidence of a detection on ``box`` in a single pass over ``image_bgr``.

        Meant for a view that shows one hypothesised hand at a fixed share of the picture (see
        fisheye.HandView): there the detector sees every hand at the size it handles best, so the
        result tells a hand from a plate or a bag much better than the full-frame score does.
        """
        b = self.model.predict(image_bgr, conf=FLOOR, imgsz=VERIFY_SIZE, device=self.device, verbose=False)[0].boxes
        return max([float(s) for q, s in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()) if _iou(q, box) >= VERIFY_IOU],
                   default=0.0)
