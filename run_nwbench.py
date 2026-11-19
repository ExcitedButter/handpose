#!/usr/bin/env python3
"""Bare-hand 3D keypoints for NW_bench (head-mounted fisheye, mono or stereo, and plain MP4s).

This is the EgoForce pipeline adapted to NW_bench. What changes, and why:

  * Detection. NW_bench hands are bare; the EgoForce YOLO-World detector is
    prompted for white gloves and misses most of them. Boxes come from the
    hand detector shipped with WiLoR, run at several input sizes
    (hand_detector.py); train_detector.py adapts it to the videos, using the
    hands this pipeline found in them. Discarded are boxes the size of a person; on the
    fisheye views, boxes touching the bottom border (the wearer's own face);
    on videos without calibration, boxes far smaller than the nearest hands
    of that video (bystanders' hands, hand-like background); among the rest
    the labeling prefers the larger boxes, which are the wearer's hands.
  * Left/right labels: the offline Viterbi labeling of labeling.py, without
    its white-glove and shoe terms. It tells left from right by position;
    when both the detector's own left/right votes and the stability of the
    pose network say the opposite for the whole video (both arms reaching in
    from one side), the labels are swapped.
  * Pose. Each labeled hand is re-rendered through a virtual pinhole camera
    aimed at it (fisheye.py), so the pose network sees an undistorted,
    centered hand. The four pose hypotheses of pose_select.py are computed
    in that view, scored against skin color instead of glove white, and mapped
    back through the Double Sphere model of camera.json.
  * Verification. A box counts as a hand when its pose hypotheses agree, at
    least 8 of its 21 keypoints lie inside the picture and the hand is
    within arm's reach (1 m; 1.5 m at the assumed focal length of a video
    without calibration). The detector must confirm every run of poses at
    least once per second, in the full frame or in the hand's own view. A
    hand the detector loses is followed from the neighboring pose sample
    for up to 1 s. A left and a right pose that coincide are one hand.
  * Stabilization: stabilize.py with ``max_step``: a run ends where the label
    jumps to another hand, smoothing may not move a sample away from its
    measured pose, and only a single missing sample is interpolated. A frame
    between two pose samples that is not interpolated takes the pose of the
    nearest sample.
  * Metric 3D. The wrist-relative joints are placed in the camera frame by
    fitting them to the viewing rays of their 2D keypoints. Stereo cases use
    the rays of both cameras (cam2 and cam3), which fixes depth by
    triangulation and calibrates the hand size; mono cases rely on the MANO
    hand size. Videos without camera.json use an assumed pinhole focal length.

Usage:
  python run_nwbench.py /data/NW_bench --out results                 # everything
  python run_nwbench.py /data/NW_bench --out results --only stereo/case0 mono_video_only/H02
  python run_nwbench.py /data/NW_bench --out results --detector weights/hand_detector/detector_nwbench.pt

Per sample it writes into <out>/<group>_<name>/:
  handpose.pkl   {"frame_t" (N,), "left"/"right": {"kp3d" (N,21,3) meters in the
                  reference camera frame (cam3; +X right, +Y down, +Z forward),
                  "kp3d_wrist_rel_mm" (N,21,3), "kp2d" (N,21,2) pixels of the
                  reference video, "valid" (N,), "depth_source" (N,) 0 none /
                  1 mono / 2 stereo}, ...}
  handpose.npz   the same arrays, flat (L_*/R_*)
  summary.json   coverage and consistency statistics
  overlay.mp4    reference video with the predicted skeletons (unless --no-video)
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fisheye  # noqa: E402
import labeling  # noqa: E402
import pose_select  # noqa: E402
import stabilize  # noqa: E402
from mcap_video import FrameSampler  # noqa: E402
from render import COLORS, EDGES  # noqa: E402

# Detection
DET_CONF = 0.05          # lowest merged detector score (see hand_detector.py) kept as a candidate
MIN_AREA = 0.0012        # box area, fraction of the image
MAX_AREA_FISHEYE = 0.06
MAX_AREA_PINHOLE = 0.40
SIZE_REF_SCORE = 0.30    # videos without calibration: a box with this merged score is surely a hand. The largest
SIZE_REF_MIN = 20        # such box of a frame is the hand nearest to the camera; given this many frames with one,
SIZE_REF_PCT = 75        # this percentile of its side is the size of the wearer's hands in the video, and
SIZE_REL_MIN = 0.25      # a box below this share of it is a bystander's hand or background,
SIZE_NEAR = 0.5          # and one below this share is less likely the wearer's than a larger one: in the labeling
SIZE_PEN = 2.5           # it costs SIZE_PEN * (1 - share / SIZE_NEAR), so the nearest hands are labeled first
FACE_BOTTOM = 0.03       # fisheye: a box ending within this share of the height from the bottom border,
FACE_CENTER = 0.90       # or centered below this share of the height, is the wearer's own face
# Hand verification and tracking through detector misses
AGREE_MAX = 0.35         # largest hypothesis disagreement (see agreement()) accepted on a detected box
AGREE_TRACK = 0.20       # ... on a box propagated from the neighboring pose sample
MAX_REACH_M = 1.0        # a hand farther than this is somebody else's. Without calibration depth is proportional to
FOCAL_TOLERANCE = 1.5    # the assumed focal length, which may be off by this factor; the limit is widened by it
MIN_KP_INSIDE = 8        # keypoints (of 21) that must lie inside the image; fewer: the hand has left the view
MIRROR_MAX_SHIFT = 0.3   # hand sizes; the mirror-image hypothesis must lie on the hand the other three found,
MIRROR_MAX_RESIZE = 1.4  # with about their extent
CONFIRM_MIN = 0.45       # a run of pose samples is a hand only where the detector, in the full frame or in the hand's
CONFIRM_REACH_S = 1.0    # own view, calls it one with at least this score within this time (see confirmed())
TRACK_MAX_S = 1.0        # longest stretch bridged by propagation from one detection
TRACK_GROW = 1.25        # propagated box = keypoint extent of the neighbor times this
# Stabilization (pose samples at --pose-hz)
SG_WINDOW_S = 0.4        # Savitzky-Golay window; the EgoForce default of 0.7 s lags fast bare-hand motion
SPIKE_RATIO = 3.0        # a sample is a spike when it leaves its neighbors' midpoint by this many neighbor distances
SPIKE_MIN = 0.25         # ... and by at least this share of the hand length
RUN_MAX_STEP = 1.0       # hand sizes per 0.1 s; a larger step between two samples means the label moved to another hand
FILL_MAX_S = 0.15        # longest hole filled by interpolation: one missing pose sample. Across longer holes the
                         # two ends are too often different hands, or the hand was out of view in between
                         # (EgoForce fills 0.3 s).
HOLD_PERIODS = 0.5       # a frame that is not interpolated takes the pose of a sample up to this many sampling
                         # periods away, so the frames between two pose samples are not lost where a run ends
SAME_HAND = 0.15         # hand sizes; a left and a right pose whose joints lie this close on average are one hand
# Labeling (labeling.py)
MAX_CANDIDATES = 5
BIRTH_DEATH = 0.4        # cost of a hand appearing or disappearing. The EgoForce value of 0.8 drops a detection that
                         # stands alone between two misses; here such a box is kept and judged by its pose instead.
SWAP_SAMPLES = 40        # left/right check (see handedness_trial): pose samples tried with both handedness flags;
SWAP_GAIN = 0.8          # labels are swapped when the flipped flags lower the hypothesis disagreement below this share
                         # (19 NW_bench videos: 0.60 on the one with both arms from one side, 1.27 to 4.7 on the others)
# Videos without calibration
FOCAL_FRAC = 0.6         # assumed focal length / longer image side
# Stereo fusion
STEREO_RMS_M = 0.02      # largest ray-fit residual accepted for a two-view fit
SCALE_RANGE = (0.8, 1.25)
DEPTH_RANGE = (0.08, 1.5)


def skin_mask(image):
    """Skin-colored pixels (YCrCb box), dilated; replaces the white-glove mask of pose_select."""
    y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
    m = ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135) & (y > 60)).astype(np.uint8)
    return cv2.dilate(m, np.ones((9, 9), np.uint8))


def in_image(kp, cam):
    """Number of keypoints (n, 2) inside the picture of ``cam``."""
    return int(((kp[:, 0] >= 0) & (kp[:, 0] < cam.width) & (kp[:, 1] >= 0) & (kp[:, 1] < cam.height)).sum())


def agreement(kp2):
    """Disagreement of the box, refit and rotated pose hypotheses (H, 21, 2): largest mean joint
    distance between two of them, in hand lengths. Small on a hand, large on anything else."""
    a = kp2[:3]
    size = max(float(np.ptp(a[1], axis=0).max()), 1.0)
    return max(float(np.linalg.norm(a[i] - a[j], axis=1).mean()) for i, j in ((0, 1), (0, 2), (1, 2))) / size


def isolated_spikes(A, k=5, nsig=3.0):
    """Drop-in for stabilize.hampel that repairs only isolated outliers.

    The Hampel filter replaces every coordinate far from the local median, which also flattens the
    start of a fast motion after a still phase. Here a sample (21 joints) is replaced by the midpoint
    of its neighbors only when it jumps away from both while they agree with each other.
    """
    out, fixed = A.copy(), 0
    if len(A) < 3:
        return out, 0
    flat = A.reshape(len(A), -1)
    scale = np.linalg.norm(np.ptp(A.reshape(len(A), 21, -1), axis=1), axis=1)  # hand extent per sample
    for i in range(1, len(A) - 1):
        mid = (flat[i - 1] + flat[i + 1]) / 2
        dev = np.abs(flat[i] - mid).reshape(21, -1).max(1).mean()
        gap = np.abs(flat[i + 1] - flat[i - 1]).reshape(21, -1).max(1).mean()
        if dev > SPIKE_RATIO * gap and dev > SPIKE_MIN * scale[i]:
            out[i] = mid.reshape(A.shape[1:])
            fixed += 1
    return out, fixed


def frame_times(mp4):
    """Presentation times (s, from 0) of the pictures of ``mp4``, in display order."""
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        pts = sorted(float(p.pts * s.time_base) for p in c.demux(s) if p.pts is not None and p.size)
    t = np.asarray(pts, np.float64)
    return t - t[0]


def iter_frames(mp4, n, keep=None):
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, f in enumerate(c.decode(s)):
            if i >= n:
                break
            if keep is None or i in keep:
                yield i, f.to_ndarray(format="bgr24")


def label_score(score):
    """Merged detector score -> score scale of labeling.py and stabilize.py."""
    return 0.05 + 0.5 * (score - DET_CONF)


def drop_distant(boxes, scores, *more):
    """Videos without calibration: remove candidates far smaller than the wearer's hands in this video.

    Without a focal length a metric reach limit is not available, but hands of bystanders and hand-like
    background are several times smaller than the wearer's own, which are the hands nearest to the camera.
    The reference is therefore taken from the largest confident box of each frame, so that a crowd of
    bystanders cannot outvote the wearer. The small boxes must go before the labeling, where they would
    otherwise take the place of a real hand. Edits the per-sample lists ``boxes``, ``scores`` and ``more``
    in place. Returns the number of boxes removed and, per sample, the size of every remaining box as a
    share of the reference (None when the video has too few confident boxes for a reference).
    """
    side = [[float(np.sqrt(labeling.area(b))) for b in bs] for bs in boxes]
    nearest = [max(v for v, s in zip(vs, ss) if s >= SIZE_REF_SCORE) for vs, ss in zip(side, scores)
               if any(s >= SIZE_REF_SCORE for s in ss)]
    if len(nearest) < SIZE_REF_MIN:
        return 0, None
    ref, removed = float(np.percentile(nearest, SIZE_REF_PCT)), 0
    for k, vs in enumerate(side):
        keep = [i for i, v in enumerate(vs) if v >= SIZE_REL_MIN * ref]
        removed += len(vs) - len(keep)
        for seq in (boxes, scores) + more:
            seq[k] = [seq[k][i] for i in keep]
        side[k] = [vs[i] / ref for i in keep]
    return removed, side


def detector_right_share(lab, boxes, rights):
    """Mean "right hand" share the detector gives the boxes labeled left, and those labeled right
    (0.5 for a label that never occurs)."""
    out = []
    for seq in (lab.left, lab.right):
        v = []
        for k, b in enumerate(seq):
            if b is not None and len(boxes[k]):
                c = labeling.center(b)     # the labeling keeps box centers
                v.append(rights[k][int(np.argmin([np.abs(labeling.center(q) - c).sum() for q in boxes[k]]))])
        out.append(float(np.mean(v)) if v else 0.5)
    return out


def confirmed(samples, times):
    """Sample indices of one hand that the detector confirms.

    ``samples``: {sample index: entry of pose_at}. Samples that can be the same hand (stabilize.jumped)
    form a run; it may pause for as long as the stabilizer interpolates (FILL_MAX_S: one missing
    sample). Inside a run the detector may be unsure for a while (motion blur, a
    hand under a plastic bag), so a sample counts when any sample of its run within CONFIRM_REACH_S scores
    CONFIRM_MIN or more, in the full frame or in the hand's own view. A plate or a jar that looks like
    a hand to the pose network never gets such a score and is dropped as a whole.
    """
    keep, run = set(), []

    def close():
        strong = [k for k in run if max(samples[k]["det"], samples[k]["own_view"]) >= CONFIRM_MIN]
        keep.update(k for k in run if any(abs(times[k] - times[j]) <= CONFIRM_REACH_S for j in strong))

    for k in sorted(samples):
        if run and (times[k] - times[run[-1]] > FILL_MAX_S
                    or stabilize.jumped({"kp2": samples[run[-1]]["xy"][1], "t": times[run[-1]]},
                                        {"kp2": samples[k]["xy"][1], "t": times[k]}, RUN_MAX_STEP)):
            close()
            run = []
        run.append(k)
    close()
    return keep


def handedness_trial(mp4, cam, lab, keys, n_frames, extractor):
    """Hypothesis disagreement of the labeled hands with their handedness flag as labeled and flipped.

    Given the wrong handedness the pose network has to explain, say, the back of a left hand as a right
    hand: its box, refit and rotated hypotheses then disagree. Returns the median agreement() over
    SWAP_SAMPLES evenly spaced pose samples for (labels as they are, labels exchanged) and the number of
    boxes tried.
    """
    ks = [k for k in range(len(keys)) if lab.left[k] is not None or lab.right[k] is not None]
    ks = [ks[i] for i in sorted(set(np.linspace(0, len(ks) - 1, min(SWAP_SAMPLES, len(ks))).round().astype(int)))] if ks else []
    wanted = {keys[k]: k for k in ks}
    same, flip = [], []
    for i, img in iter_frames(mp4, n_frames, keep=set(wanted)):
        k = wanted[i]
        for side, box in ((0, lab.left[k]), (1, lab.right[k])):
            if box is None:
                continue
            try:
                view = fisheye.HandView(cam, box)
            except ValueError:
                continue
            picture, a = view.render(img), []
            for flag in (side, 1 - side):
                xy = view.to_image(pose_select.hypotheses(extractor, picture, view.box[None], [flag], batch_size=2)[0][0])
                a.append(agreement(xy) if np.isfinite(xy).all() else np.nan)
            if np.isfinite(a).all():
                same.append(a[0])
                flip.append(a[1])
    if not same:
        return 0.0, 0.0, 0
    return float(np.median(same)), float(np.median(flip)), len(same)


def candidates(detector, image, is_fisheye):
    """Hand box candidates of one frame, best first: (boxes, merged detector scores, detector's "right
    hand" shares)."""
    h, w = image.shape[:2]
    max_area = (MAX_AREA_FISHEYE if is_fisheye else MAX_AREA_PINHOLE) * w * h
    rows = []
    for r in detector.predict(image, conf=DET_CONF):
        r = [float(v) for v in r]
        if not (MIN_AREA * w * h <= labeling.area(r) <= max_area) or not np.isfinite(r).all():
            continue
        if is_fisheye and (r[3] >= h * (1 - FACE_BOTTOM) or (r[1] + r[3]) / 2 > h * FACE_CENTER):
            continue
        rows.append(r)
    rows = labeling._nms(rows)
    drop = []
    for r in rows:
        inside = [o for o in rows if o is not r and labeling.inter(r, o) / max(1., labeling.area(o)) > .7
                  and labeling.area(o) < .7 * labeling.area(r)]
        # a box around both hands, or around a hand and the object it holds
        if (len(inside) >= 2 and labeling.overlap_small(inside[0], inside[1]) < .5) or any(o[4] > r[4] for o in inside):
            drop.append(r)
    rows = [r for r in rows if r not in drop]
    return [np.array(r[:4], np.float32) for r in rows], [r[4] for r in rows], [r[5] for r in rows]


def track_view(mp4, cam, is_fisheye, models, args, swap=None):
    """Detection, labeling, pose and tracking for one video: every step that needs the networks.

    Returns the pose samples before any of them is judged (see finish_view): dict(frame_t (N,), keys and
    times of the pose samples, the box candidates, track {side: {sample: entry of pose_at}}, stats).
    ``swap``: left/right decision of the other view of a stereo pair (None: decide here).
    """
    detector, extractor = models
    ft = frame_times(mp4)
    if args.max_frames:
        ft = ft[:args.max_frames]
    n = len(ft)
    sampler = FrameSampler(start_s=0.0, sample_hz=args.pose_hz) if args.pose_hz > 0 else None
    keys, times, cb, cs, cr = [], [], [], [], []
    decoded = 0
    for i, img in iter_frames(mp4, n):
        decoded = i + 1
        if sampler is not None and not sampler.select(ft[i]):
            continue
        b, s, r = candidates(detector, img, is_fisheye)
        keys.append(i); times.append(ft[i]); cb.append(b); cs.append(s); cr.append(r)
    ft, n = ft[:decoded], decoded
    if not keys:
        raise ValueError(f"No frames decoded from {mp4}")
    distant, share = (0, None) if is_fisheye else drop_distant(cb, cs, cr)
    # The labeling takes the MAX_CANDIDATES best boxes of a sample. They are counted only now: in front of a
    # crowd the bystanders' hands, just removed, outscore the wearer's own and would use up the list.
    for seq in (cb, cs, cr) + ((share,) if share else ()):
        seq[:] = [v[:MAX_CANDIDATES] for v in seq]
    lab = labeling.label_hands(np.asarray(times), cb, [[label_score(v) for v in c] for c in cs], width=cam.width,
                               extra=share and [[SIZE_PEN * max(0.0, 1 - v / SIZE_NEAR) for v in vs] for vs in share])

    # labeling.py tells left from right by position (the left hand is normally on the left). When both arms
    # reach in from one side this is the wrong way round for the whole video, and the pose network, given
    # the wrong handedness, returns unstable poses. Two independent checks must both say so before the labels
    # are exchanged: the detector, which also classifies each box as left or right, must lean against the
    # labels over the whole video, and the pose network must agree with itself clearly better with the
    # handedness flags flipped.
    vote = detector_right_share(lab, cb, cr)
    trial, swapped = None, bool(swap)
    if swap is None and vote[0] > vote[1]:
        trial = handedness_trial(mp4, cam, lab, keys, n, extractor)
        swapped = trial[2] >= 10 and trial[1] < SWAP_GAIN * trial[0]
    if swapped:
        lab.left, lab.right, lab.left_score, lab.right_score = lab.right, lab.left, lab.right_score, lab.left_score
    reach = MAX_REACH_M if is_fisheye else MAX_REACH_M * FOCAL_TOLERANCE
    wanted = {key: k for k, key in enumerate(keys)}
    track, refused = {0: {}, 1: {}}, {0: {}, 1: {}}
    max_age = max(1, int(round(TRACK_MAX_S * (args.pose_hz if args.pose_hz > 0 else 30))))
    rejected, lost = {}, {}

    def pose_at(img, side, box, det, age):
        """(pose hypotheses for one box, None), or (None, why no pose can be placed in the picture).
        ``det``: merged detector score of the box, 0 for a box propagated from a neighboring sample."""
        try:
            view = fisheye.HandView(cam, box)
        except ValueError:
            return None, "outside the camera model"
        picture = view.render(img)
        kp2, kp3, ev, _ = pose_select.hypotheses(extractor, picture, view.box[None], [side], batch_size=2)
        xy = view.to_image(kp2[0])
        if not np.isfinite(xy).all():
            return None, "outside the camera model"
        p = xy[1]
        if in_image(p, cam) < MIN_KP_INSIDE:
            return None, "outside the image"
        t, _ = fisheye.solve_translation(view.to_camera(kp3[0][1]), cam.unproject(p))
        a = agreement(xy)
        # The mirror-image hypothesis comes from a pass with the other handedness and can land elsewhere;
        # then it is not an alternative reading of this hand and is replaced by the refit hypothesis.
        size, size_m = float(np.ptp(p, axis=0).max()), float(np.ptp(xy[3], axis=0).max())
        if (np.linalg.norm(xy[3].mean(0) - p.mean(0)) > MIRROR_MAX_SHIFT * size
                or max(size, size_m) > MIRROR_MAX_RESIZE * min(size, size_m)):
            xy[3], kp3[0][3], ev[0][3] = xy[1], kp3[0][1], ev[0][1]
        return {"xy": xy, "kp3": kp3[0], "cam3": view.to_camera(kp3[0]), "ev": ev[0], "box": np.asarray(box, np.float32),
                "det": det, "agree": a, "age": age, "depth": float(np.linalg.norm(t)),    # distance from the hand size
                "own_view": detector.score_at(picture, pose_select.kp_box(kp2[0][1]))}, None

    def refusal(e, limit):
        """Why the pose ``e`` is not taken for a hand of the wearer, or None."""
        if not (DEPTH_RANGE[0] < e["depth"] < reach):
            return "out of reach"
        if e["agree"] > limit:
            return "hypotheses disagree"
        return None

    def next_box(entry):
        """Box for the neighboring sample: extent of the refit hypothesis' keypoints, enlarged."""
        kp = entry["xy"][1]
        c, half = (kp.min(0) + kp.max(0)) / 2, np.ptp(kp, axis=0).max() * TRACK_GROW / 2
        return np.array([c[0] - half, c[1] - half, c[0] + half, c[1] + half], np.float32)

    def propagate(img, side, k, src):
        """Pose at sample k from the neighbor's pose ``src``; refused when it lands on the other hand."""
        box = next_box(src)
        if labeling.area(box) < 0.25 * MIN_AREA * cam.width * cam.height:
            return None, "too small"
        e, why = pose_at(img, side, box, 0.0, src["age"] + 1)
        why = why or refusal(e, AGREE_TRACK)
        if why is not None:
            if e is not None:
                refused[side].setdefault(k, dict(e, why=why))
            return None, why
        if labeling.iou(next_box(e), box) < 0.3:
            return None, "moved away"
        o = track[1 - side].get(k)
        if o is not None and labeling.overlap_small(next_box(e), next_box(o)) > 0.5:
            return None, "on the other hand"
        return e, None

    def count(d, why):
        d[why] = d.get(why, 0) + 1

    from collections import deque
    recent = deque(maxlen=max_age)
    tracked = 0
    for i, img in iter_frames(mp4, n, keep=set(wanted)):
        k = wanted[i]
        for side, box, sc in ((0, lab.left[k], lab.left_score[k]), (1, lab.right[k], lab.right_score[k])):
            e = None
            if box is not None:
                e, why = pose_at(img, side, box, DET_CONF + (sc - label_score(DET_CONF)) / 0.5, 0)
                why = why or refusal(e, AGREE_MAX)
                if why is not None:
                    count(rejected, why)
                    if e is not None:
                        refused[side][k] = dict(e, why=why)
                    e = None
            prev = track[side].get(k - 1)
            if e is None and prev is not None and prev["age"] < max_age:      # forward through a detector miss
                e, why = propagate(img, side, k, prev)
                tracked += e is not None
                if e is None:
                    count(lost, why)
            if e is None:
                continue
            track[side][k] = e
            if e["age"] == 0:                                            # backward from a fresh detection
                cur, need = e, k - 1
                for j, old in reversed(recent):
                    if j != need or j in track[side] or cur["age"] >= max_age:
                        break
                    b, why = propagate(old, side, j, cur)
                    if b is None:
                        count(lost, why)
                        break
                    track[side][j] = cur = b
                    tracked += 1
                    need -= 1
        recent.append((k, img))
    return {"frame_t": ft, "keys": keys, "times": times, "cand_boxes": cb, "cand_scores": cs, "cand_right": cr,
            "left": lab.left, "right": lab.right, "track": track, "refused": refused,
            "stats": {"pose_samples": len(keys), "labeling": lab.stats, "candidates_dropped_too_small": int(distant),
                      "detector_right_share": {"left_label": round(vote[0], 3), "right_label": round(vote[1], 3)},
                      "handedness_trial": None if trial is None else {"disagreement_as_labeled": round(trial[0], 3),
                                                                      "disagreement_flipped": round(trial[1], 3),
                                                                      "boxes": trial[2]},
                      "left_right_swapped": bool(swapped), "boxes_rejected": rejected,
                      "samples_tracked_through_detector_miss": int(tracked), "tracking_stopped": lost}}


def finish_view(raw, cam):
    """Which pose samples of track_view are hands, which hypothesis each takes, and stabilization.

    Returns dict(frame_t (N,), xy {side: (N,21,2)}, rel {side: (N,21,3) m, camera frame, wrist-relative},
    ok {side: (N,)}, stats, debug).
    """
    ft, keys, times = raw["frame_t"], raw["keys"], raw["times"]
    n, track = len(ft), {side: dict(samples) for side, samples in raw["track"].items()}
    dbg = {s: {k: {"skin": float(e["ev"][1, 0]), "agree": float(e["agree"]), "age": int(e["age"]), "det": float(e["det"]),
                   "own_view": float(e["own_view"]), "depth": float(e["depth"]), "center": e["xy"][1].mean(0).tolist(),
                   "size": float(np.ptp(e["xy"][1], axis=0).max())} for k, e in track[i].items()} for i, s in enumerate("LR")}
    unconfirmed = 0
    for side in track:
        keep = confirmed(track[side], times)
        for k in [k for k in track[side] if k not in keep]:
            del track[side][k]
            unconfirmed += 1
    # Left and right pose on one hand (two boxes on it, or a propagated pose that slid over): the one the
    # detector supports less is dropped, a propagated pose before a detected one. Two hands that work
    # together overlap as well, but not joint by joint.
    duplicates = 0
    for k in sorted(set(track[0]) & set(track[1])):
        a, b = track[0][k], track[1][k]
        a, b = a["xy"][1], b["xy"][1]
        if np.linalg.norm(a - b, axis=1).mean() < SAME_HAND * min(np.ptp(a, axis=0).max(), np.ptp(b, axis=0).max()):
            weaker = min((0, 1), key=lambda side: (track[side][k]["age"] == 0, max(track[side][k]["det"], track[side][k]["own_view"])))
            del track[weaker][k]
            duplicates += 1
    preds = {k: [] for k in range(len(keys))}
    hyp = np.zeros(pose_select.N_HYP, int)
    for side, samples in track.items():
        runs = [[]]
        for k in sorted(samples):
            if runs[-1] and k - runs[-1][-1] > 1:
                runs.append([])
            runs[-1].append(k)
        for run in (r for r in runs if r):
            c = pose_select.select(np.stack([samples[k]["xy"] for k in run]), np.stack([samples[k]["kp3"] for k in run]),
                                   np.stack([samples[k]["ev"] for k in run]), bool(side))
            for k, h in zip(run, c):
                e = samples[k]
                hyp[h] += 1
                dbg["LR"[side]][k].update(chosen=int(h), hyp_xy=e["xy"].astype(np.float32))
                # "bbox" is the pose's own box: stabilize.py drops a pose whose wrist is far from the center of
                # its detection box, which the tight boxes of this detector do not allow (the wrist sits on the
                # box border or, for a hand entering the picture, outside it).
                preds[k].append({"right": side, "detector_score": label_score(max(e["det"], DET_CONF)), "keypoints_2d": e["xy"][h],
                                 "joints_3d": e["cam3"][h], "bbox": pose_select.kp_box(e["xy"][h])})
    preds = [(times[k], {"offset_s": times[k], "hands": preds[k]}) for k in range(len(keys))]
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    stab, stats = stabilize.build(preds, ft, min_score=0.0, win_s=SG_WINDOW_S, frame_wh=(cam.width, cam.height),
                                  max_step=RUN_MAX_STEP, max_fill=FILL_MAX_S, hold=HOLD_PERIODS * period + 1e-3)
    # How far smoothing moved the raw samples (wrist, pixels): a check against over-smoothing.
    for side, right in (("L", 0), ("R", 1)):
        measured = {t: h["keypoints_2d"] for t, row in preds for h in row["hands"] if h["right"] == right}
        d = [np.linalg.norm(e["kp2"] - measured[e["t"]], axis=1).mean() for e in stab[side] if e["t"] in measured]
        stats[side]["smoothing_shift_px_median"] = round(float(np.median(d)), 2) if d else None
        stats[side]["smoothing_shift_px_p95"] = round(float(np.percentile(d, 95)), 2) if d else None
    out = {"frame_t": ft, "xy": {}, "rel": {}, "ok": {}}
    for side in ("L", "R"):
        xy, rel, ok = np.zeros((n, 21, 2), np.float32), np.zeros((n, 21, 3), np.float32), np.zeros(n, bool)
        for e in stab[side]:
            if in_image(e["kp2"], cam) < MIN_KP_INSIDE:      # smoothing or interpolation carried it out of the view
                continue
            lo, hi = int(np.searchsorted(ft, e["t"] - 1e-6, "left")), int(np.searchsorted(ft, e["t"] + 1e-6, "right"))
            xy[lo:hi], rel[lo:hi], ok[lo:hi] = e["kp2"], e["kp3"], True
        out["xy"][side], out["rel"][side], out["ok"][side] = xy, rel, ok
    out["debug"] = {"keys": keys, "times": times, "cand_boxes": raw["cand_boxes"], "cand_scores": raw["cand_scores"],
                    "cand_right": raw["cand_right"], "preds": preds, "left": raw["left"], "right": raw["right"],
                    "posed": {s: sorted(track[i]) for i, s in enumerate("LR")}, "samples": dbg,
                    "kept_t": {s: [e["t"] for e in stab[s]] for s in "LR"}, "raw": raw}
    out["stats"] = dict(raw["stats"], poses_rejected_not_confirmed=int(unconfirmed), poses_rejected_duplicate=int(duplicates),
                        pose_hypotheses_chosen=dict(zip(("box", "refit", "rotated", "mirror_alt"), map(int, hyp))),
                        stabilizer=stats)
    return out


def smooth_runs(t, ok, radius=3):
    """Median then mean filter of (N, 3) translations inside each run of valid frames."""
    out = t.copy()
    idx = np.flatnonzero(ok)
    if not len(idx):
        return out
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        a = t[run]
        for f in (np.median, np.mean):
            a = np.stack([f(a[max(0, i - radius):i + radius + 1], axis=0) for i in range(len(a))])
        out[run] = a
    return out


def place(ref, cam, other=None, cam_o=None, T_o=None):
    """Metric camera-frame joints for the reference view ``ref``; with ``other`` (second view of a
    stereo pair, its camera and its 4x4 transform into the reference frame) depth is triangulated."""
    n = len(ref["frame_t"])
    res, stats = {}, {}
    for side in ("L", "R"):
        ok, rel, xy = ref["ok"][side].copy(), ref["rel"][side].astype(np.float64), ref["xy"][side]
        rays = cam.unproject(xy)
        ok &= np.isfinite(rays).all((1, 2))
        t = np.zeros((n, 3))
        src = np.zeros(n, np.uint8)
        for i in np.flatnonzero(ok):
            t[i], _ = fisheye.solve_translation(rel[i], rays[i])
            src[i] = 1
        z_mono = t[:, 2].copy()
        st = {"frames": int(ok.sum())}
        scale = 1.0
        if other is not None:
            R, c = T_o[:3, :3], T_o[:3, 3]
            m = min(n, len(other["frame_t"]))
            both = np.zeros(n, bool)
            both[:m] = ok[:m] & other["ok"][side][:m]
            rays_o = np.full((n, 21, 3), np.nan)
            rays_o[:m] = cam_o.unproject(other["xy"][side][:m]) @ R.T
            both &= np.isfinite(rays_o).all((1, 2))
            org = np.concatenate([np.zeros((21, 3)), np.tile(c, (21, 1))])

            def fit(i, s):
                return fisheye.solve_translation(np.tile(rel[i] * s, (2, 1)), np.concatenate([rays[i], rays_o[i]]), org)
            # Hand size: per frame, the scale of the model hand that best fits both views; median over the video.
            scales = []
            for i in np.flatnonzero(both):
                ts, rms = fit(i, 1.0)
                if rms < STEREO_RMS_M and DEPTH_RANGE[0] < ts[2] < DEPTH_RANGE[1] and z_mono[i] > 0:
                    scales.append(ts[2] / z_mono[i])
            if len(scales) >= 10:
                scale = float(np.clip(np.median(scales), *SCALE_RANGE))
            rel = rel * scale
            for i in np.flatnonzero(ok):
                t[i], _ = fisheye.solve_translation(rel[i], rays[i])
            z_mono = t[:, 2].copy()
            rms_all, ratio, reproj = [], [], []
            for i in np.flatnonzero(both):
                ts, rms = fit(i, scale)
                if rms < STEREO_RMS_M and DEPTH_RANGE[0] < ts[2] < DEPTH_RANGE[1]:
                    ratio.append(z_mono[i] / ts[2])
                    t[i], src[i] = ts, 2
                    rms_all.append(rms)
                    p = cam_o.project((rel[i] + ts - c) @ R)
                    reproj.append(np.linalg.norm(p - other["xy"][side][i], axis=1).mean())
            st.update(frames_both_views=int(both.sum()), frames_stereo=int((src == 2).sum()), hand_scale=round(scale, 3),
                      scale_samples=len(scales),
                      ray_fit_rms_mm=round(1000 * float(np.median(rms_all)), 1) if rms_all else None,
                      second_view_reproj_px=round(float(np.median(reproj)), 1) if reproj else None,
                      mono_over_stereo_depth_median=round(float(np.median(ratio)), 3) if ratio else None,
                      mono_vs_stereo_depth_abs_err_pct=round(100 * float(np.median(np.abs(np.asarray(ratio) - 1))), 1)
                      if ratio else None)
        ok &= (t[:, 2] > DEPTH_RANGE[0]) & (t[:, 2] < 3.0)
        src[~ok] = 0
        t = smooth_runs(t, ok)
        xyz = np.where(ok[:, None, None], rel + t[:, None], 0).astype(np.float32)
        res[side] = {"kp3d": xyz, "kp3d_wrist_rel_mm": np.where(ok[:, None, None], rel * 1000, 0).astype(np.float32),
                     "kp2d": np.where(ok[:, None, None], xy, 0).astype(np.float32), "valid": ok, "depth_source": src}
        if ok.any():
            st["wrist_depth_m_median"] = round(float(np.median(t[ok, 2])), 3)
        st["coverage"] = round(float(ok.mean()), 4)
        stats[side] = st
    return res, stats


def draw(img, kp, right, label):
    color = (80, 220, 90) if right else (60, 170, 255)
    th = max(2, round(img.shape[1] / 500))
    for a, b in EDGES:
        cv2.line(img, tuple(np.rint(kp[a]).astype(int)), tuple(np.rint(kp[b]).astype(int)), COLORS[(b - 1) // 4], th, cv2.LINE_AA)
    for p in kp:
        cv2.circle(img, tuple(np.rint(p).astype(int)), th + 1, color, -1, cv2.LINE_AA)
    x, y = np.rint(kp[0]).astype(int)
    cv2.putText(img, label, (x + 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, th * .4, color, th, cv2.LINE_AA)


def render_overlay(mp4, res, n, out, width=960):
    with av.open(str(mp4)) as src, av.open(str(out), "w") as dst:
        vs = src.streams.video[0]
        fps = round(float(vs.average_rate or 30))
        stream = None
        for i, f in enumerate(src.decode(vs)):
            if i >= n:
                break
            img = f.to_ndarray(format="bgr24")
            for key, right in (("left", False), ("right", True)):
                h = res[key]
                if h["valid"][i]:
                    tag = f"{'R' if right else 'L'} {h['kp3d'][i, 0, 2]:.2f}m" + ("*" if h["depth_source"][i] == 2 else "")
                    draw(img, h["kp2d"][i], right, tag)
            hh = int(round(img.shape[0] * width / img.shape[1] / 2)) * 2
            img = cv2.resize(img, (width, hh), interpolation=cv2.INTER_AREA)
            if stream is None:
                stream = dst.add_stream("libx264", rate=fps)
                stream.width, stream.height, stream.pix_fmt = width, hh, "yuv420p"
                stream.options = {"crf": "23", "preset": "fast"}
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                dst.mux(pkt)
        if stream is not None:
            for pkt in stream.encode():
                dst.mux(pkt)


def find_samples(root):
    """(name, reference mp4, second mp4 or None, camera.json or None) for every NW_bench sample."""
    out = []
    for group in ("mono", "stereo"):
        for case in sorted((root / group).glob("case*"), key=lambda p: int(p.name[4:])):
            ref, second = case / "videos" / "cam3.mp4", case / "videos" / "cam2.mp4"
            if ref.is_file():
                out.append((f"{group}/{case.name}", ref, second if second.is_file() else None, case / "camera.json"))
    for mp4 in sorted((root / "mono_video_only").glob("*.mp4")):
        out.append((f"mono_video_only/{mp4.stem}", mp4, None, None))
    return out


def run_sample(name, ref_mp4, second_mp4, camera_json, models, args, out_dir):
    started = time.perf_counter()
    if camera_json is not None:
        cams, T, ref_name = fisheye.load_cameras(camera_json)
        cam = cams[ref_name]
        camera = {"model": "double_sphere", "reference_camera": ref_name, "source": str(camera_json)}
    else:
        with av.open(str(ref_mp4)) as c:
            w, h = c.streams.video[0].width, c.streams.video[0].height
        cam = fisheye.Pinhole(FOCAL_FRAC * max(w, h), w / 2, h / 2, w, h)
        camera = {"model": "pinhole_assumed", "focal_px": cam.f,
                  "note": "no calibration: metric depth is proportional to the assumed focal length"}
    ref = finish_view(track_view(ref_mp4, cam, camera_json is not None, models, args), cam)
    other = cam_o = T_o = None
    if second_mp4 is not None and not args.mono_only:
        o_name = next(k for k in cams if k != ref_name)
        cam_o, T_o = cams[o_name], T[o_name]
        other = finish_view(track_view(second_mp4, cam_o, True, models, args, swap=ref["stats"]["left_right_swapped"]), cam_o)
    res, stats = place(ref, cam, other, cam_o, T_o)
    n = len(ref["frame_t"])
    out = {"frame_t": ref["frame_t"].astype(np.float32), "left": res["L"], "right": res["R"],
           "image_wh": np.array([cam.width, cam.height], np.int32), "camera": camera,
           "units": "kp3d: meters in the reference camera frame (+X right, +Y down, +Z forward); kp3d_wrist_rel_mm: "
                    "millimeters relative to the wrist, same axes; kp2d: pixels of the reference video; depth_source: "
                    "0 none, 1 mono (hand-size based), 2 stereo; joints: wrist, then thumb, index, middle, ring, little "
                    "(base to tip)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "handpose.pkl", "wb") as f:
        pickle.dump(out, f)
    flat = {"frame_t": out["frame_t"], "image_wh": out["image_wh"]}
    for side, key in (("L", "left"), ("R", "right")):
        for a, v in res[side].items():
            flat[f"{side}_{a}"] = v
    np.savez_compressed(out_dir / "handpose.npz", **flat)
    if args.debug:
        with open(out_dir / "debug.pkl", "wb") as f:
            pickle.dump({"ref": ref["debug"], "second": other["debug"] if other is not None else None}, f)
    if not args.no_video:
        render_overlay(ref_mp4, out, n, out_dir / "overlay.mp4")
    summary = {"sample": name, "frames": n, "coverage_L": stats["L"]["coverage"], "coverage_R": stats["R"]["coverage"],
               "elapsed_s": round(time.perf_counter() - started, 1), "camera": camera, "placement": stats,
               "reference_view": ref["stats"]}
    if other is not None:
        summary["second_view"] = other["stats"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def configure():
    """Bare-hand settings of the EgoForce modules: no white-glove appearance terms, pose hypotheses scored
    against skin color, spike repair that leaves fast motion alone."""
    pose_select.glove_mask = skin_mask
    stabilize.hampel = isolated_spikes
    labeling.BIRTH = labeling.DEATH = BIRTH_DEATH


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="NW_bench folder (mono/, stereo/, mono_video_only/)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=None, help="Sample names, e.g. stereo/case0 mono_video_only/H02")
    p.add_argument("--shard", default=None, help="i/n: process every n-th sample starting at i (one process per GPU)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mono-only", action="store_true", help="Ignore cam2 of stereo cases")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--debug", action="store_true", help="Also write debug.pkl (candidates, labels, posed samples)")
    p.add_argument("--pose-hz", type=float, default=15.0, help="Detection and pose rate; 0 = every frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--detector", default=None, help="Hand detector weights (default: weights/hand_detector/detector.pt, "
                   "or $HANDPOSE_HAND_DETECTOR); see train_detector.py for weights adapted to a set of videos")
    p.add_argument("--no-fp16", action="store_true")
    a = p.parse_args()

    samples = find_samples(a.root)
    if a.only:
        samples = [s for s in samples if s[0] in set(a.only)]
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        samples = samples[i::n]
    todo = [s for s in samples if a.overwrite or not (a.out / s[0].replace("/", "_") / "summary.json").is_file()]
    print(f"{len(samples)} samples, {len(todo)} to process", flush=True)
    if not todo:
        return

    import torch
    from hand_detector import HandDetector
    from WiLoR.inference import WiLoRPoseExtractor
    cv2.setNumThreads(2)
    torch.set_num_threads(8)
    fp16 = a.device == "cuda" and not a.no_fp16
    configure()
    models = (HandDetector(model=a.detector, device=a.device), WiLoRPoseExtractor(device=a.device, fp16=fp16))
    a.out.mkdir(parents=True, exist_ok=True)
    for k, (name, ref, second, cj) in enumerate(todo, 1):
        try:
            s = run_sample(name, ref, second, cj, models, a, a.out / name.replace("/", "_"))
        except Exception as e:  # keep going on the other samples
            print(f"[{k}/{len(todo)}] {name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        with open(a.out / "summary.jsonl", "a") as f:
            f.write(json.dumps({q: v for q, v in s.items() if q not in ("reference_view", "second_view")}) + "\n")
        extra = ""
        st = s["placement"]
        if "frames_stereo" in st["L"]:
            extra = (f"  stereo frames L {st['L']['frames_stereo']} R {st['R']['frames_stereo']}"
                     f"  reproj px L {st['L']['second_view_reproj_px']} R {st['R']['second_view_reproj_px']}")
        print(f"[{k}/{len(todo)}] {name}: {s['frames']} frames  L {s['coverage_L']:.1%}  R {s['coverage_R']:.1%}"
              f"  {s['elapsed_s']:.0f} s{extra}", flush=True)


if __name__ == "__main__":
    main()
