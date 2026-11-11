"""Offline two-hand detection and left/right labeling for a whole recording.

The detector runs on every pose frame. Instead of following each hand with a
visual tracker, all detections of the recording are labeled at once by a
Viterbi search over per-frame assignments ``(left box, right box)``, each box
optionally absent. The cost of a path combines:

* detection evidence: assigning a confident box is rewarded;
* motion continuity: a hand keeps moving smoothly between pose frames, so a
  label that jumps to a distant box or changes size abruptly is penalized;
* appearance: the EgoForce gloves are white, so a box with almost no
  white (low-saturation, bright) pixels, such as a black wristband or sleeve,
  is penalized;
* geometry: with both hands visible, the left hand is normally to the left of
  the right hand (crossing is allowed but costs), and a hand far on the
  opposite side of the image is mildly penalized;
* births and deaths of a hand cost a constant, so short detector misses are
  bridged by the stabilizer instead of producing label changes.

Because the search sees the future as well as the past, a hand that the
detector picks up late, or that re-enters the image, is labeled from the
whole sequence rather than from the frame on which it appears.
"""
from dataclasses import dataclass, field
import math

import numpy as np

# Detection
CONF = 0.05             # lowest detector score kept as a candidate
STRONG = 0.12           # score counted as a confident detection
MIN_AREA = 0.001        # minimum box area, fraction of the image
MAX_CANDIDATES = 5      # per frame, after merging and filtering
# Path costs
BIRTH = 0.8             # a hand appears
DEATH = 0.8             # a hand disappears
MOVE = 2.5              # weight on squared normalized center motion (per 0.1 s)
RESIZE = 1.0            # weight on |log area ratio|
CROSS = 1.5             # left box to the right of the right box
SIDE = 0.8              # hand on the far opposite side of the image
OVERLAP_MAX = 0.6       # two hands may not share more than this of the smaller box
WHITE_MIN = 0.2         # white-pixel share below which a box is penalized
WHITE_PEN = 3.0         # penalty for a box with no white pixels at all
WHITE_S, WHITE_V = 60, 110   # HSV thresholds of a white pixel (OpenCV scale)


def area(b):
    return float(max(0., b[2] - b[0]) * max(0., b[3] - b[1]))


def inter(a, b):
    return float(max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1])))


def iou(a, b):
    i = inter(a, b)
    return i / max(area(a) + area(b) - i, 1.)


def overlap_small(a, b):
    return inter(a, b) / max(1., min(area(a), area(b)))


def center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def diag(b):
    return math.hypot(b[2] - b[0], b[3] - b[1])


def whiteness(image, box):
    """Share of white (low-saturation, bright) pixels inside ``box``."""
    import cv2
    h, w = image.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return 0.
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] < WHITE_S) & (hsv[..., 2] > WHITE_V)).mean())


def _nms(rows, thr=0.5):
    rows = sorted(rows, key=lambda r: -r[4])
    keep = []
    for r in rows:
        if all(iou(r, k) < thr for k in keep):
            keep.append(r)
    return keep


def detect_candidates(detector, image, scale=1.0):
    """Candidate hand boxes on one frame, in coordinates multiplied by ``scale``.

    Returns ``(boxes, scores, whites)``; ``whites`` is the white-pixel share of
    each box (see :func:`whiteness`).

    The full view is queried first; when it yields fewer than two confident
    boxes, two overlapping halves are queried at higher effective resolution.
    Boxes that cover two other candidates (both hands merged in one box) are
    dropped.
    """
    h, w = image.shape[:2]
    rows = [list(map(float, r[:5])) for r in detector.predict(image, conf=CONF, iou=.5)]
    if sum(r[4] >= STRONG for r in rows) < 2:
        for x0, x1 in ((0, int(w * .6)), (int(w * .4), w)):
            for r in detector.predict(image[:, x0:x1], conf=CONF, iou=.5):
                rows.append([float(r[0]) + x0, float(r[1]), float(r[2]) + x0, float(r[3]), float(r[4])])
    rows = [r for r in rows if area(r) >= MIN_AREA * w * h and np.isfinite(r).all()]
    rows = _nms(rows)
    merged = []
    for r in rows:
        inside = [o for o in rows if o is not r and inter(r, o) / max(1., area(o)) > .6 and area(o) < .7 * area(r)]
        if len(inside) >= 2 and overlap_small(inside[0], inside[1]) < .5:
            merged.append(r)
    rows = [r for r in rows if r not in merged][:MAX_CANDIDATES]
    whites = [whiteness(image, r[:4]) for r in rows]
    return [np.array(r[:4], np.float32) * scale for r in rows], [r[4] for r in rows], whites


def _bonus(score):
    return min(2.0, 1.5 * math.sqrt(score / 0.3))


@dataclass
class Labeling:
    left: list = field(default_factory=list)      # per frame: box (4,) or None
    right: list = field(default_factory=list)
    left_score: list = field(default_factory=list)
    right_score: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _appearance(white):
    return WHITE_PEN * max(0., WHITE_MIN - white) / WHITE_MIN


def label_hands(times, boxes, scores, width, whites=None):
    """Viterbi labeling. ``boxes[k]``/``scores[k]``/``whites[k]`` are the candidates of frame k."""
    n = len(times)
    states, emit = [], []
    for k in range(n):
        b, s = boxes[k], scores[k]
        wk = whites[k] if whites is not None else [1.] * len(b)
        st, em = [], []
        idx = range(-1, len(b))
        for i in idx:
            for j in idx:
                if i >= 0 and i == j:
                    continue
                if i >= 0 and j >= 0 and overlap_small(b[i], b[j]) > OVERLAP_MAX:
                    continue
                cost = 0.
                if i >= 0:
                    x = center(b[i])[0] / width
                    cost += -_bonus(s[i]) + _appearance(wk[i]) + SIDE * max(0., x - .65) / .35
                if j >= 0:
                    x = center(b[j])[0] / width
                    cost += -_bonus(s[j]) + _appearance(wk[j]) + SIDE * max(0., .35 - x) / .35
                if i >= 0 and j >= 0 and center(b[i])[0] > center(b[j])[0] + .02 * width:
                    cost += CROSS
                st.append((i, j))
                em.append(cost)
        states.append(st)
        emit.append(np.array(em))

    def trans(bprev, iprev, bcur, icur, dt):
        if iprev < 0 and icur < 0:
            return 0.
        if iprev < 0:
            return BIRTH
        if icur < 0:
            return DEATH
        a, c = bprev[iprev], bcur[icur]
        d = np.linalg.norm(center(a) - center(c)) / max(diag(a), diag(c), 1.)
        d = d * 0.1 / max(dt, 0.1)
        return MOVE * d * d + RESIZE * abs(math.log(max(area(c), 1.) / max(area(a), 1.)))

    cost = emit[0].copy()
    back = [None]
    for k in range(1, n):
        dt = float(times[k] - times[k - 1])
        bp, bc = boxes[k - 1], boxes[k]
        tl = {}
        for ip in range(-1, len(bp)):
            for ic in range(-1, len(bc)):
                tl[ip, ic] = trans(bp, ip, bc, ic, dt)
        prev_states = states[k - 1]
        new_cost = np.empty(len(states[k]))
        ptr = np.empty(len(states[k]), np.int64)
        for si, (i, j) in enumerate(states[k]):
            tot = cost + np.array([tl[pi, i] + tl[pj, j] for pi, pj in prev_states])
            m = int(np.argmin(tot))
            new_cost[si], ptr[si] = tot[m] + emit[k][si], m
        cost = new_cost
        back.append(ptr)
    out = Labeling()
    if n == 0:
        return out
    si = int(np.argmin(cost))
    path = [0] * n
    for k in range(n - 1, -1, -1):
        path[k] = si
        if k > 0:
            si = int(back[k][si])
    crossings, missed_two = 0, 0
    for k in range(n):
        i, j = states[k][path[k]]
        b, s = boxes[k], scores[k]
        out.left.append(None if i < 0 else b[i])
        out.right.append(None if j < 0 else b[j])
        out.left_score.append(0. if i < 0 else s[i])
        out.right_score.append(0. if j < 0 else s[j])
        if i >= 0 and j >= 0 and center(b[i])[0] > center(b[j])[0]:
            crossings += 1
        if sum(x >= STRONG for x in s) >= 2 and (i < 0 or j < 0):
            missed_two += 1
    out.stats = {"frames": n, "frames_two_confident": int(sum(sum(x >= STRONG for x in s) >= 2 for s in scores)),
                 "frames_two_confident_one_unassigned": missed_two, "crossed_frames": crossings,
                 "left_frames": int(sum(b is not None for b in out.left)),
                 "right_frames": int(sum(b is not None for b in out.right))}
    _smooth_sizes(out.left)
    _smooth_sizes(out.right)
    return out


def _smooth_sizes(seq, radius=2):
    """Median-filter box width and height within each run; centers are kept."""
    k = 0
    n = len(seq)
    while k < n:
        if seq[k] is None:
            k += 1
            continue
        e = k
        while e < n and seq[e] is not None:
            e += 1
        wh = np.array([[b[2] - b[0], b[3] - b[1]] for b in seq[k:e]])
        for t in range(k, e):
            lo, hi = max(k, t - radius), min(e, t + radius + 1)
            m = np.median(wh[lo - k:hi - k], axis=0)
            c = center(seq[t])
            seq[t] = np.array([c[0] - m[0] / 2, c[1] - m[1] / 2, c[0] + m[0] / 2, c[1] + m[1] / 2], np.float32)
        k = e
