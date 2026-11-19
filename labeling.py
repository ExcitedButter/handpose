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
* shoe-likeness: white sneakers look like gloves to the glove/hand prompts.
  The same detector is also queried with "shoe"/"sneaker" prompts; a box that
  scores clearly higher as a shoe than as a glove is penalized;
* optional second detector: when WiLoR's YOLO hand detector is installed,
  its confident boxes that no YOLO-World candidate covers are added as weak
  candidates (they mostly recover a hand next to the other hand, or a glove
  holding a shoe); their handedness output is not used;
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
WHITE_MIN = 0.35        # white-pixel share below which a box is penalized (v7: 0.2 -> 0.35)
WHITE_PEN = 3.0         # penalty for a box with no white pixels at all
WHITE_S, WHITE_V = 60, 110   # HSV thresholds of a white pixel (OpenCV scale)
SHOE_PROMPTS = ("shoe", "sneaker")
HAND_DET_CONF = 0.3     # extra candidates: hand-detector boxes at least this confident ...
HAND_DET_SCALE = 0.3    # ... enter with this fraction of their confidence as score
HAND_DET_MAX = 3        # ... at most this many per frame
SHOE_MATCH = 0.5        # IoU for a shoe-query box to describe a candidate
SHOE_MARGIN = 0.15      # shoe score minus glove score where the penalty starts
SHOE_RANGE = 0.25       # ... and over which it ramps up to SHOE_PEN
SHOE_PEN = 3.0          # penalty for a box that is clearly a shoe (v10: 6 -> 3, so a glove holding a shoe is kept)
HD_EVID_CONF = 0.05     # hand-detector boxes at least this confident are kept as gap candidates
# Gap rescue (v11): inside a short gap of one hand's track, a weak box near the interpolated position
RESCUE_MAX_S = 2.0      # longest gap (seconds) that is searched
RESCUE_IOU = 0.3        # IoU with the interpolated box for a weak candidate to be used
RESCUE_WHITE = 0.2      # minimum white share of a weak candidate


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


def shoe_scores(shoe_detector, image, boxes):
    """Per box: shoe score minus glove score of the best-overlapping prediction of
    ``shoe_detector`` (YOLO-World prompted with "white glove", "hand", then SHOE_PROMPTS)
    on the full view; 0 where no prediction overlaps by SHOE_MATCH IoU."""
    out = np.zeros(len(boxes), np.float32)
    if shoe_detector is None or not len(boxes):
        return out
    pb, ps = shoe_detector.predict_classes(image, conf=0.03)
    for i, b in enumerate(boxes):
        best = SHOE_MATCH
        for q, s in zip(pb, ps):
            v = iou(b, q)
            if v > best:
                best, out[i] = v, float(s[2:].max() - s[:2].max())
    return out


def hand_detector_boxes(hand_detector, image, rows):
    """Boxes of the optional hand detector (WiLoR's YOLO hand detector).

    Returns ``(extra, dets)``: ``extra`` are up to HAND_DET_MAX boxes with confidence
    >= HAND_DET_CONF that no candidate in ``rows`` covers (IoU <= 0.5), as rows
    [x0, y0, x1, y1, HAND_DET_SCALE x confidence]; ``dets`` are all detections with
    confidence >= HD_EVID_CONF as rows [x0, y0, x1, y1, confidence]."""
    if hand_detector is None:
        return [], []
    r = hand_detector.predict(image, conf=HD_EVID_CONF, verbose=False)[0]
    if not len(r.boxes):
        return [], []
    dets = sorted(([float(v) for v in b] + [float(c)] for b, c in
                   zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist())), key=lambda d: -d[4])
    out = []
    for d in dets:
        if d[4] < HAND_DET_CONF or any(iou(d, o) > .5 for o in rows + out):
            continue
        out.append(d[:4] + [HAND_DET_SCALE * d[4]])
        if len(out) >= HAND_DET_MAX:
            break
    return out, dets



def detect_candidates(detector, image, scale=1.0, shoe_detector=None, hand_detector=None):
    """Candidate hand boxes on one frame, in coordinates multiplied by ``scale``.

    Returns ``(boxes, scores, whites, shoes, spare)``; ``whites`` is the white-pixel
    share of each box (see :func:`whiteness`), ``shoes`` its shoe-minus-glove
    score (see :func:`shoe_scores`; zeros without ``shoe_detector``).

    The full view is queried first; when it yields fewer than two confident
    boxes, two overlapping halves are queried at higher effective resolution.
    Boxes that cover two other candidates (both hands merged in one box) are
    dropped.
        With ``hand_detector`` (an ultralytics YOLO hand detector), up to
    ``HAND_DET_MAX`` of its boxes that no candidate covers are appended with
    score ``HAND_DET_SCALE`` x confidence (see :func:`hand_detector_boxes`).
    ``spare`` holds the remaining weak hand-detector boxes as arrays
    [x0, y0, x1, y1, confidence, white share], used only to fill gaps (:func:`rescue_gaps`).
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
    extra, dets = hand_detector_boxes(hand_detector, image, rows)
    whites = [whiteness(image, r[:4]) for r in rows + extra]
    # Hand-detector boxes get no shoe score: they mostly recover a glove holding a shoe.
    shoes = list(shoe_scores(shoe_detector, image, [r[:4] for r in rows])) + [0.] * len(extra)
    rows = rows + extra
    spare = [np.array([v * scale for v in d[:4]] + [d[4], whiteness(image, d[:4])], np.float32)
             for d in dets if all(iou(d, r) <= .5 for r in rows)]
    return [np.array(r[:4], np.float32) * scale for r in rows], [r[4] for r in rows], whites, shoes, spare


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


def _shoe(shoe):
    return SHOE_PEN * min(1., max(0., shoe - SHOE_MARGIN) / SHOE_RANGE)


def label_hands(times, boxes, scores, width, whites=None, shoes=None):
    """Viterbi labeling. ``boxes[k]``/``scores[k]``/``whites[k]`` are the candidates of frame k;
    ``shoes[k]`` their shoe-minus-glove scores (see :func:`shoe_scores`)."""
    n = len(times)
    states, emit = [], []
    for k in range(n):
        b, s = boxes[k], scores[k]
        wk = whites[k] if whites is not None else [1.] * len(b)
        sk = shoes[k] if shoes is not None else [0.] * len(b)
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
                    cost += -_bonus(s[i]) + _appearance(wk[i]) + _shoe(sk[i]) + SIDE * max(0., x - .65) / .35
                if j >= 0:
                    x = center(b[j])[0] / width
                    cost += -_bonus(s[j]) + _appearance(wk[j]) + _shoe(sk[j]) + SIDE * max(0., .35 - x) / .35
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
    crossings, missed_two, shoe_rejected = 0, 0, 0
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
        if shoes is not None:
            shoe_rejected += sum(1 for x, v in zip(s, shoes[k]) if x >= STRONG and _shoe(v) >= SHOE_PEN)
    out.stats = {"frames": n, "frames_two_confident": int(sum(sum(x >= STRONG for x in s) >= 2 for s in scores)),
                 "frames_two_confident_one_unassigned": missed_two, "crossed_frames": crossings,
                 "confident_shoe_like_boxes": shoe_rejected,
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


def rescue_gaps(lab, times, boxes, whites, spare=None):
    """Boxes for samples inside short gaps of a labeled hand track (v11).

    A gap of at most RESCUE_MAX_S seconds with the hand labeled on both sides is
    usually a partial occlusion: the hand is still there, the detector just gave
    it no usable box. For each sample of such a gap, the box interpolated
    between the two ends is compared with that frame's candidates and weak
    hand-detector boxes (``spare``); the best one with IoU >= RESCUE_IOU and
    white share >= RESCUE_WHITE is used, otherwise the interpolated box itself.
    The caller verifies every rescued pose against the image before keeping it.

    Returns ``{k: [(side, box, source), ...]}`` with source "candidate" or "interpolated".
    """
    out = {}
    n = len(times)
    for side, seq, other in ((0, lab.left, lab.right), (1, lab.right, lab.left)):
        k = 0
        while k < n:
            if seq[k] is not None:
                k += 1
                continue
            e = k
            while e < n and seq[e] is None:
                e += 1
            if k == 0 or e == n or times[e] - times[k - 1] > RESCUE_MAX_S:
                k = e
                continue
            a, b = np.asarray(seq[k - 1], np.float32), np.asarray(seq[e], np.float32)
            t0, t1 = times[k - 1], times[e]
            for t in range(k, e):
                w = (times[t] - t0) / max(t1 - t0, 1e-6)
                ib = (1 - w) * a + w * b
                cands = [(np.asarray(c, np.float32), wh) for c, wh in zip(boxes[t], whites[t])]
                if spare is not None:
                    cands += [(np.asarray(r[:4], np.float32), float(r[5])) for r in spare[t]]
                best, src, bv = ib, "interpolated", RESCUE_IOU
                for c, wh in cands:
                    if other[t] is not None and overlap_small(c, other[t]) > OVERLAP_MAX:
                        continue
                    v = iou(c, ib)
                    if v >= bv and wh >= RESCUE_WHITE:
                        best, src, bv = c, "candidate", v
                if other[t] is not None and overlap_small(best, other[t]) > OVERLAP_MAX:
                    continue
                out.setdefault(t, []).append((side, best, src))
            k = e
    return out
