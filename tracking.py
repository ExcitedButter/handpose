"""Persistent two-hand localization, without pose inference or learned weights.

CSRT follows every source picture. YOLO confirmations are associated to the
existing tracks; missing tracks trigger local recovery even when the detector
returned two other boxes. Visual tracking alone never refreshes detection age.
Hand labels follow image geometry: a new track starts on its own screen side,
and whenever both hands are tracked the one further right is the right hand
(slots are swapped if they disagree). A single hand keeps its track's label.
A slot that has not been confirmed for ``reacquire_s`` seconds is released and
can re-initialize anywhere, instead of waiting for a hand near its last box.
"""
from dataclasses import dataclass
import itertools
import math

import cv2
import numpy as np


def area(box):
    return float(np.prod(np.maximum(0., np.asarray(box)[2:]-np.asarray(box)[:2])))


def intersection(a, b):
    return float(np.prod(np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))))


def iou(a, b):
    inter = intersection(a, b)
    return inter/max(area(a)+area(b)-inter, 1.)


def center(box):
    return (np.asarray(box)[:2]+np.asarray(box)[2:])/2


def clip(box, w, h):
    result = np.asarray(box, np.float32).copy()
    result[[0, 2]] = np.clip(result[[0, 2]], 0, w)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, h)
    return result


@dataclass
class _Track:
    right: int
    track_id: str
    bbox: np.ndarray
    crop: np.ndarray
    last_detection: float
    score: float
    visual: object = None
    visible: bool = True
    source: str = "detector"
    raw_detection: np.ndarray | None = None


class HandTracker:
    def __init__(self, detector, *, conf=.12, detect_every=5, max_gap_s=.6,
                 recovery_conf=.08, tracker_scale=.5, reacquire_s=2.0, order_margin=.02):
        if detect_every < 1 or max_gap_s <= 0 or not 0 < tracker_scale <= 1:
            raise ValueError("Require detect_every >= 1, max_gap_s > 0 and 0 < tracker_scale <= 1")
        self.detector = detector
        self.conf = float(conf)
        self.detect_every = int(detect_every)
        self.max_gap_s = float(max_gap_s)
        self.recovery_conf = float(recovery_conf)
        self.tracker_scale = float(tracker_scale)
        self.reacquire_s = float(reacquire_s)
        self.order_margin = float(order_margin)
        self.now = 0.
        self.tracks = {}
        self.frame_index = 0
        self.previous_time = None
        self.generations = {0: 0, 1: 0}
        self.last_recovery = {0: -float("inf"), 1: -float("inf")}
        self.stats = {k: 0 for k in ("detector_calls", "recovery_calls", "rejected_merged", "rejected_jump", "expired_tracks", "visual_failures", "label_swaps")}

    def _query(self, image, roi, conf, source, diagnostics):
        h, w = image.shape[:2]
        box = np.array([0, 0, w, h]) if roi is None else np.rint(clip(roi, w, h)).astype(int)
        x0, y0, x1, y1 = box.tolist()
        if x1-x0 < 16 or y1-y0 < 16:
            return []
        diagnostics["detector_calls"] += 1
        diagnostics["recovery_calls"] += int(source == "recovery")
        rows = self.detector.predict(image[y0:y1, x0:x1], conf=conf, iou=.5)
        output = []
        for row in rows:
            if len(row) < 5 or not np.isfinite(row[:5]).all():
                continue
            candidate = np.asarray(row[:4], np.float32)+np.array([x0, y0, x0, y0], np.float32)
            candidate = clip(candidate, w, h)
            if area(candidate) < w*h*.001 or row[4] < conf:
                continue
            output.append({"bbox": candidate, "score": float(row[4]), "source": source})
        return output

    def _init_visual(self, track, small):
        factory = getattr(cv2, "TrackerCSRT_create", None)
        if factory is None:
            raise RuntimeError("Persistent tracking requires OpenCV CSRT (opencv-contrib-python)")
        visual = factory()
        box = track.bbox*self.tracker_scale
        x0, y0, x1, y1 = np.rint(box).astype(int)
        visual.init(small, (int(x0), int(y0), max(2, int(x1-x0)), max(2, int(y1-y0))))
        track.visual = visual

    def _visual_box(self, track, small):
        if track.visual is None:
            return None
        ok, (x, y, w, h) = track.visual.update(small)
        if not ok or w <= 1 or h <= 1:
            return None
        return np.array([x, y, x+w, y+h], np.float32)/self.tracker_scale

    def _live(self, side):
        """Track in this slot, or None once it has gone unconfirmed for reacquire_s."""
        track = self.tracks.get(side)
        if track is None or self.now-track.last_detection > self.reacquire_s:
            return None
        return track

    def _merged(self, candidate):
        live = [self._live(k) for k in (0, 1)]
        if any(t is None for t in live):
            return False
        a, b = [t.bbox for t in live]
        covers_both = all(intersection(candidate, box)/max(1., area(box)) > .55 for box in [a, b])
        overlap = intersection(a, b)/max(1., min(area(a), area(b)))
        # Overlap can be genuine during crossing, but it must not disable all
        # merged-box checks after a tracker begins drifting onto the other hand.
        expanded = area(candidate) > 1.55*min(area(a), area(b)) or (
            candidate[2]-candidate[0] > 1.45*min(a[2]-a[0], b[2]-b[0]))
        return covers_both and (overlap < .45 or expanded)

    def _compatible(self, track, candidate):
        a, b = track.bbox, candidate
        ratio = area(b)/max(area(a), 1.)
        shape_ratio = (b[2:]-b[:2])/np.maximum(a[2:]-a[:2], 1.)
        distance = float(np.linalg.norm(center(a)-center(b)))/max(float(np.linalg.norm(a[2:]-a[:2])), 1.)
        return .28 < ratio < 2.5 and np.all(shape_ratio < [1.65, 1.8]) and (iou(a, b) >= .12 or distance < .38)

    def _score(self, side, detection, w):
        box = detection["bbox"]
        track = self._live(side)
        if track is None:
            x = center(box)[0]/w
            other = self._live(1-side)
            if other is None:
                # Without another hand, a new slot initializes on its own side.
                if (side == 0 and x > .57) or (side == 1 and x < .43):
                    return -float("inf")
                return detection["score"] + .25-.5*abs(x-(.30 if side == 0 else .70))
            # With another hand tracked, any separate box may start this slot;
            # the geometric ordering in update() fixes the labels.
            if intersection(box, other.bbox)/max(1., min(area(box), area(other.bbox))) > .5:
                return -float("inf")
            return detection["score"] + .1
        if not self._compatible(track, box):
            return -float("inf")
        dist = float(np.linalg.norm(center(track.bbox)-center(box)))/max(float(np.linalg.norm(track.bbox[2:]-track.bbox[:2])), 1.)
        return detection["score"] + 1.2*iou(track.bbox, box)-.5*dist-.1*abs(math.log(area(box)/max(1., area(track.bbox))))

    def _associate(self, candidates, w, diagnostics, sides=(0, 1)):
        filtered = []
        for detection in candidates:
            if self._merged(detection["bbox"]):
                diagnostics["rejected_merged"] += 1
                continue
            filtered.append(detection)
        best, best_score = {}, 0.
        # At most eight full-view proposals plus tile candidates. Exhaustive
        # assignment is inexpensive for two hands and permits either to be absent.
        choices = [None, *range(len(filtered))]
        for indices in itertools.product(choices, repeat=len(sides)):
            picked = [i for i in indices if i is not None]
            if len(picked) != len(set(picked)):
                continue
            if len(picked) == 2:
                a, b = [filtered[i]["bbox"] for i in picked]
                overlap = intersection(a, b)/max(1., min(area(a), area(b)))
                if overlap > .7:
                    continue
            else:
                overlap = 0.
            total = -.5*overlap
            assignment = {}
            for side, index in zip(sides, indices):
                if index is not None:
                    score = self._score(side, filtered[index], w)
                    if not np.isfinite(score) or score <= 0:
                        total = -float("inf")
                        break
                    total += score
                    assignment[side] = filtered[index]
            if total > best_score:
                best, best_score = assignment, total
        for detection in filtered:
            if self.tracks and not any(self._compatible(t, detection["bbox"]) for t in self.tracks.values()):
                diagnostics["rejected_jump"] += 1
        return best

    def _confirm(self, side, detection, t, small):
        box = detection["bbox"].copy()
        track = self._live(side)
        if track is None or t-track.last_detection > self.max_gap_s:
            self.generations[side] += 1
            name = "right" if side else "left"
            track = _Track(side, f"{name}-{self.generations[side]:04d}", box, box.copy(), t, detection["score"])
            self.tracks[side] = track
        else:
            # A visual track gives a useful small-frame motion estimate; a
            # detector measurement corrects it without fully recropping anew.
            track.bbox = (.65*box+.35*track.bbox).astype(np.float32)
        track.raw_detection = box
        track.last_detection, track.score = t, detection["score"]
        track.visible, track.source = True, detection["source"]
        self._init_visual(track, small)

    def _order_by_geometry(self, w, diagnostics):
        """When both hands are visible, the one further right is the right hand."""
        a, b = self.tracks.get(0), self.tracks.get(1)
        shown = [t is not None and t.visible and self.now-t.last_detection <= self.max_gap_s for t in (a, b)]
        if not all(shown) or center(a.bbox)[0] <= center(b.bbox)[0] + self.order_margin*w:
            return
        a.right, b.right = 1, 0
        self.tracks = {0: b, 1: a}
        self.last_recovery = {0: self.last_recovery[1], 1: self.last_recovery[0]}
        for side, track in self.tracks.items():
            self.generations[side] += 1
            track.track_id = f"{'right' if side else 'left'}-{self.generations[side]:04d}"
        diagnostics["label_swaps"] += 1

    def update(self, image_bgr, timestamp_s):
        t = float(timestamp_s)
        if not np.isfinite(t) or (self.previous_time is not None and t <= self.previous_time):
            raise ValueError("Tracking timestamps must be finite and strictly increasing")
        h, w = image_bgr.shape[:2]
        self.now = t
        small = cv2.resize(image_bgr, (round(w*self.tracker_scale), round(h*self.tracker_scale)), interpolation=cv2.INTER_AREA)
        diagnostics = {k: 0 for k in self.stats}
        elapsed = 1/30 if self.previous_time is None else t-self.previous_time
        # Expiry controls emissions, but retain the last track box for gated
        # reacquisition rather than relabeling a crossing hand by screen side.
        expired_now = []
        for side, track in self.tracks.items():
            track.source = "visual_tracker"
            track.visible = False
            if t-track.last_detection <= self.max_gap_s:
                box = self._visual_box(track, small)
                if box is not None and self._compatible(track, box):
                    track.bbox = clip(box, w, h)
                    track.visible = True
                else:
                    diagnostics["visual_failures"] += 1
            elif self.previous_time-track.last_detection <= self.max_gap_s:
                expired_now.append(side)
                diagnostics["expired_tracks"] += 1
        scheduled = self.frame_index % self.detect_every == 0 or not self.tracks or bool(expired_now)
        matched = {}
        if scheduled:
            detections = self._query(image_bgr, None, self.conf, "detector", diagnostics)
            if not self.tracks:
                # Always bootstrap with both half views, even if the full view
                # returned a merged hand plus a background false positive.
                detections += self._query(image_bgr, [0, 0, w//2, h], self.conf, "detector", diagnostics)
                detections += self._query(image_bgr, [w//2, 0, w, h], self.conf, "detector", diagnostics)
            matched = self._associate(detections, w, diagnostics)
            for side, detection in matched.items():
                self._confirm(side, detection, t, small)
        # Recovery belongs to each track, not to the total detector count.
        for side in (0, 1):
            track = self._live(side)
            needs_recovery = side not in matched and (scheduled or track is None or not track.visible)
            if not needs_recovery or t-self.last_recovery[side] < .10-1e-6:
                continue
            self.last_recovery[side] = t
            if track is None:
                roi = np.array([0 if side == 0 else w*.40, h*.15, w*.60 if side == 0 else w, h])
            else:
                c = center(track.bbox)
                wh = np.maximum(track.bbox[2:]-track.bbox[:2], [w*.08, h*.12])
                roi = np.concatenate([c-.8*wh, c+.8*wh])
                other = self._live(1-side)
                if other is not None:
                    # Divide the recovery context between the predicted hand
                    # centers. Unlike fixed image halves this boundary follows
                    # their motion and preserves identity through crossing.
                    delta = center(other.bbox)-c
                    axis = int(np.argmax(np.abs(delta)/np.maximum(wh, 1.)))
                    if abs(delta[axis]) > .25*min(wh[axis], (other.bbox[2:]-other.bbox[:2])[axis]):
                        boundary = (c[axis]+center(other.bbox)[axis])/2
                        if delta[axis] > 0:
                            roi[axis+2] = min(roi[axis+2], boundary)
                        else:
                            roi[axis] = max(roi[axis], boundary)
            candidates = self._query(image_bgr, roi, self.recovery_conf, "recovery", diagnostics)
            accepted = self._associate(candidates, w, diagnostics, sides=(side,))
            if side in accepted:
                candidate = accepted[side]
                other = self._live(1-side)
                if other is not None and intersection(candidate["bbox"], other.bbox)/max(1., min(area(candidate["bbox"]),area(other.bbox))) > .7:
                    continue
                self._confirm(side, candidate, t, small)
        self._order_by_geometry(w, diagnostics)
        output = []
        for side in (0, 1):
            track = self.tracks.get(side)
            if track is None or not track.visible or t-track.last_detection > self.max_gap_s:
                continue
            # Stabilize scale more strongly than center. These are observed-hand
            # extents; WiLoR's existing 2.2 crop enlargement remains downstream.
            old_center, old_wh = center(track.crop), track.crop[2:]-track.crop[:2]
            new_center, new_wh = center(track.bbox), track.bbox[2:]-track.bbox[:2]
            c = old_center+(1-math.exp(-elapsed/.04))*(new_center-old_center)
            wh = old_wh+(1-math.exp(-elapsed/.20))*(new_wh-old_wh)
            track.crop = clip(np.concatenate([c-wh/2, c+wh/2]), w, h)
            output.append({"bbox": track.crop.copy(), "raw_bbox": track.bbox.copy(),
                "detector_bbox": None if track.raw_detection is None else track.raw_detection.copy(),
                "right": side, "detector_score": track.score, "track_id": track.track_id,
                "track_source": track.source, "seconds_since_detection": max(0., t-track.last_detection),
                "handedness_verified": False})
        for key in self.stats:
            self.stats[key] += diagnostics[key]
        diagnostics["track_ids"] = [track["track_id"] for track in output]
        self.frame_index += 1
        self.previous_time = t
        return output, diagnostics
