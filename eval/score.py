"""Recall and precision of run_nwbench.py results on the hand-labeled frames.

    python eval/score.py results [more result dirs ...] [-v]

Two kinds of labels, both made by eye:

* eval/hand_points.json (the videos without calibration): the position of every hand on evenly
  spaced frames, as [x, y, kind] or [x, y, kind, side]. kind "hand" is a clearly visible hand of
  the camera wearer and must be found. kind "ignore" (the wearer's hand cut by the image border,
  hidden or blurred beyond recognition) is neither required nor an error. kind "other" is
  somebody else's hand: a skeleton on it is an error.
  A predicted hand is correct when a labeled hand center lies inside the bounding box of its
  keypoints, grown on every side by MARGIN_CELL grid cells (accuracy of the labels) plus
  MARGIN_SIZE of the box size (the center of a large, partly covered hand is not well defined);
  each labeled hand can be matched once.
  recall = matched / labeled hands, precision = matched / counted predictions. Where the label
  gives the side, "left/right" is the share of matched hands output under the correct name.
* eval/visible_hands.json (the fisheye samples and H02): only the number c of visible hands.
  With o = hands output, recall = sum min(o, c) / sum c and precision = sum min(o, c) / sum o. This
  does not check where a skeleton lies, so precision is an upper bound; confirm on overlay.mp4.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MARGIN_CELL = 0.5    # grid cells
MARGIN_SIZE = 0.25   # of the predicted hand's size


def match_points(preds, labels, wh, grid):
    """preds: list of (x0, y0, x1, y1) keypoint bounding boxes, pixels; labels: list of [x, y, kind, ...]
    fractions. Returns (status per prediction: "tp", "fp" or "ignored"; for every label of kind "hand",
    in order, the index of its prediction or None)."""
    cell = float(np.mean([wh[0] / grid[0], wh[1] / grid[1]]))
    hands = [np.array(l[:2]) * wh for l in labels if l[2] == "hand"]
    ignore = [np.array(l[:2]) * wh for l in labels if l[2] == "ignore"]

    def on(p, q):
        m = MARGIN_CELL * cell + MARGIN_SIZE * float(np.sqrt(max((p[2] - p[0]) * (p[3] - p[1]), 0.0)))
        return p[0] - m <= q[0] <= p[2] + m and p[1] - m <= q[1] <= p[3] + m

    def dist(p, q):
        return float(np.hypot((p[0] + p[2]) / 2 - q[0], (p[1] + p[3]) / 2 - q[1]))
    pairs = sorted((dist(p, q), i, j) for i, p in enumerate(preds) for j, q in enumerate(hands) if on(p, q))
    used_p, used_h = {}, {}
    for _, i, j in pairs:
        if i not in used_p and j not in used_h:
            used_p[i], used_h[j] = j, i
    # unmatched: on an "ignore" label -> not counted; anything else (a second skeleton on a matched hand,
    # another person's hand, an object) -> false
    status = ["tp" if i in used_p else "ignored" if any(on(p, q) for q in ignore) else "fp" for i, p in enumerate(preds)]
    return status, [used_h.get(j) for j in range(len(hands))]


def predictions(z, f):
    """[(side, keypoint bounding box)] of the hands output on frame f."""
    out = []
    for s in "LR":
        if z[f"{s}_valid"][f]:
            kp = z[f"{s}_kp2d"][f]
            out.append((s, (*kp.min(0).tolist(), *kp.max(0).tolist())))
    return out


def score_points(z, lab):
    """Counts over the labeled frames of one sample: tp, fp, fn, sides known, sides correct, and
    {frame: (missed, false, wrong side)} for the frames with an error."""
    wh = z["image_wh"].astype(float)
    T, bad = np.zeros(5, int), {}
    for fr, labels in lab["frames"].items():
        preds = predictions(z, int(fr))
        status, match = match_points([b for _, b in preds], labels, wh, lab["grid"])
        hands = [l for l in labels if l[2] == "hand"]
        sided = [(preds[i][0], h[3]) for h, i in zip(hands, match) if i is not None and len(h) > 3 and h[3] in ("L", "R")]
        row = (match.count(None), status.count("fp"), sum(a != b for a, b in sided))
        T += (status.count("tp"), row[1], row[0], len(sided), len(sided) - row[2])
        if any(row):
            bad[int(fr)] = row
    return T, bad


def main():
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv
    points = json.loads((HERE / "hand_points.json").read_text())["samples"]
    counts = json.loads((HERE / "visible_hands.json").read_text())["labels"]
    for res in args:
        print(f"== {res}")
        total = np.zeros(5, int)
        for name in list(points) + [k for k in counts if k not in points]:
            f = Path(res) / name.replace("/", "_") / "handpose.npz"
            if not f.is_file():
                continue
            z = np.load(f)
            if name in points:
                T, bad = score_points(z, points[name])
                total += T
                tp, fp, fn, sn, sok = T
                side = f"  left/right {sok}/{sn}" if sn else ""
                print(f"  {name:22s} frames {len(points[name]['frames']):2d}  recall {tp / max(tp + fn, 1):6.1%}  precision "
                      f"{tp / max(tp + fp, 1):6.1%}  (labeled {tp + fn}, found {tp}, false {fp}){side}  [positions]")
                if verbose:
                    for fr in sorted(bad):
                        print(f"     frame {fr}: {bad[fr][0]} missed, {bad[fr][1]} false, {bad[fr][2]} wrong side")
            else:
                lab = counts[name]
                o = z["L_valid"].astype(int) + z["R_valid"].astype(int)
                fr = np.array([int(k) for k in lab])
                c = np.array(list(lab.values()))
                m = np.minimum(o[fr], c).sum()
                print(f"  {name:22s} frames {len(fr):2d}  recall {m / max(c.sum(), 1):6.1%}  precision "
                      f"{m / max(o[fr].sum(), 1):6.1%}  (visible {int(c.sum())}, output {int(o[fr].sum())})  [counts]")
        tp, fp, fn, sn, sok = total
        if tp + fn:
            print(f"  {'all position labels':22s}            recall {tp / max(tp + fn, 1):6.1%}  precision {tp / max(tp + fp, 1):6.1%}"
                  f"  (labeled {tp + fn}, found {tp}, false {fp})" + (f"  left/right {sok}/{sn}" if sn else ""))


if __name__ == "__main__":
    main()
