#!/usr/bin/env python3
"""Offline, zero-phase stabilization of per-frame hand pose predictions.

Steps, per hand (left/right label from the tracker):
  1. drop detections with detector score below ``MIN_SCORE``;
  2. reject poses whose wrist lies outside the hand's own detection box
     (farther than ``BOX_TOL`` box diagonals from its center); predictions
     without a box fall back to rejecting wrist jumps larger than
     ``MAX_JUMP_PX`` between consecutive samples;
  3. split into runs at holes longer than ``MAX_GAP_S``;
  4. repair isolated spikes with a Hampel filter (local median replacement);
  5. smooth each run with a centered Savitzky-Golay filter (zero phase, no lag);
  6. interpolate onto the video frame times, never across a hole longer than
     ``MAX_FILL_S`` (frames inside longer holes are reported as missing).

Input predictions are ``(t_seconds, row)`` pairs where ``row["hands"]`` holds
dicts with ``right``, ``detector_score``, ``keypoints_2d`` (21x2, pixels) and
``joints_3d`` (21x3, meters, root relative).
"""
import argparse
import gzip
import json

import numpy as np
from scipy.signal import savgol_filter

MIN_SCORE = 0.05
MAX_JUMP_PX = 200.0
BOX_TOL = 0.6
MAX_GAP_S = 1.20
MAX_FILL_S = 0.30
SG_WINDOW_S = 0.7
SG_POLY = 2


def load(path):
    """Read predictions.jsonl or predictions.jsonl.gz written by run.py."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            out.append((float(row.get("offset_s", 0.0)), row))
    out.sort(key=lambda item: item[0])
    return out


def med_step(seq, key):
    if len(seq) < 2:
        return 0.0
    return float(np.median([np.linalg.norm(b[key] - a[key], axis=1).mean() for a, b in zip(seq, seq[1:])]))


def lag_samples(raw, filt, max_shift=4):
    """Integer shift that best aligns the filtered wrist x onto the raw one (0 = no lag)."""
    if len(raw) < 16:
        return 0
    a = np.asarray([r[0] for r in raw], np.float64)
    b = np.asarray([f[0] for f in filt], np.float64)
    a, b = a - a.mean(), b - b.mean()
    best, best_shift = -9e9, 0
    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            x, y = a[-s:], b[:len(b) + s]
        elif s > 0:
            x, y = a[:len(a) - s], b[s:]
        else:
            x, y = a, b
        n = min(len(x), len(y))
        if n < 8:
            continue
        d = np.linalg.norm(x[:n]) * np.linalg.norm(y[:n])
        c = float(np.dot(x[:n], y[:n]) / d) if d > 0 else -9e9
        if c > best:
            best, best_shift = c, s
    return best_shift


def hampel(A, k=5, nsig=3.0):
    """Replace coordinates further than nsig robust sigmas from the local median."""
    N = A.shape[0]
    if N < 2 * k + 1:
        return A.copy(), 0
    out = A.copy()
    fixed = 0
    flat = A.reshape(N, -1)
    of = out.reshape(N, -1)
    for i in range(N):
        lo, hi = max(0, i - k), min(N, i + k + 1)
        win = flat[lo:hi]
        med = np.median(win, axis=0)
        mad = np.median(np.abs(win - med), axis=0) * 1.4826
        bad = np.abs(flat[i] - med) > np.maximum(nsig * mad, 1e-9)
        if bad.any():
            of[i][bad] = med[bad]
            fixed += int(bad.sum())
    return out, fixed


def split_runs(seq, max_gap):
    runs = [[]]
    for d in seq:
        if runs[-1] and d["t"] - runs[-1][-1]["t"] > max_gap:
            runs.append([])
        runs[-1].append(d)
    return [r for r in runs if r]


def smooth_run(run, win_s=SG_WINDOW_S, poly=SG_POLY):
    n = len(run)
    if n < 5:
        return [dict(d) for d in run]
    dt = np.median(np.diff([d["t"] for d in run])) or 0.1
    w = max(5, int(round(win_s / dt)) | 1)
    if w > n:
        w = n if n % 2 else n - 1
    if w < 5 or w <= poly:
        return [dict(d) for d in run]
    K2 = np.stack([d["kp2"] for d in run])
    K3 = np.stack([d["kp3"] for d in run])
    S2 = savgol_filter(K2, w, poly, axis=0, mode="interp")
    S3 = savgol_filter(K3, w, poly, axis=0, mode="interp")
    return [dict(t=run[i]["t"], kp2=S2[i], kp3=S3[i], score=run[i]["score"]) for i in range(n)]


def build(preds, frame_times, min_score=MIN_SCORE, max_jump=MAX_JUMP_PX,
          win_s=SG_WINDOW_S, poly=SG_POLY, max_gap=MAX_GAP_S, max_fill=MAX_FILL_S):
    """Return ({'L': frames, 'R': frames}, stats); each frame is dict(t, kp2, kp3, score)."""
    raw = {"L": [], "R": []}
    dropped = 0
    for t, row in preds:
        for h in row.get("hands", []):
            score = float(h.get("detector_score", 0.0))
            if score < min_score:
                dropped += 1
                continue
            kp2 = np.asarray(h["keypoints_2d"], np.float64)
            kp3 = np.asarray(h["joints_3d"], np.float64)
            if kp2.shape != (21, 2) or kp3.shape != (21, 3):
                continue
            box = h.get("bbox")
            box = np.asarray(box, np.float64) if box is not None else None
            raw["R" if h.get("right") else "L"].append(dict(t=t, kp2=kp2, kp3=kp3, score=score, box=box))
    out, stats = {}, {}
    for side in ("L", "R"):
        kept, ref = [], None
        for d in sorted(raw[side], key=lambda d: d["t"]):
            box = d.pop("box")
            if box is not None and box.shape == (4,):
                # The left/right labeling already enforces motion continuity of
                # the boxes; only reject poses inconsistent with their own box.
                c = (box[:2] + box[2:]) / 2
                diag = float(np.hypot(*(box[2:] - box[:2])))
                if np.linalg.norm(d["kp2"][0] - c) <= BOX_TOL * max(diag, 1.0):
                    kept.append(d)
                continue
            gap = (d["t"] - kept[-1]["t"]) if kept else 9.0
            if ref is None or gap > max_gap or np.linalg.norm(d["kp2"][0] - ref) <= max_jump:
                kept.append(d)
                ref = d["kp2"][0]
        pre2 = med_step(kept, "kp2")
        nfix = 0
        for run in split_runs(kept, max_gap):
            if len(run) >= 11:
                H2, a2 = hampel(np.stack([d["kp2"] for d in run]))
                H3, a3 = hampel(np.stack([d["kp3"] for d in run]))
                nfix += a2 + a3
                for i, d in enumerate(run):
                    d["kp2"], d["kp3"] = H2[i], H3[i]
        sm, lags = [], []
        for run in split_runs(kept, max_gap):
            f = smooth_run(run, win_s, poly)
            if len(run) >= 16:
                lags.append(lag_samples([d["kp2"][0] for d in run], [d["kp2"][0] for d in f]))
            sm.extend(f)
        frames = []
        if len(sm) >= 2:
            ta = np.array([d["t"] for d in sm])
            A2 = np.stack([d["kp2"] for d in sm])
            A3 = np.stack([d["kp3"] for d in sm])
            for ft in frame_times:
                i = int(np.clip(np.searchsorted(ta, ft), 1, len(ta) - 1))
                if not (ta[i - 1] - 1e-6 <= ft <= ta[i] + 1e-6):
                    continue
                if ta[i] - ta[i - 1] > max_fill:
                    continue
                w = float(np.clip((ft - ta[i - 1]) / max(ta[i] - ta[i - 1], 1e-9), 0, 1))
                frames.append(dict(t=float(ft), kp2=A2[i - 1] * (1 - w) + A2[i] * w,
                                   kp3=A3[i - 1] * (1 - w) + A3[i] * w, score=sm[i]["score"]))
        out[side] = frames
        stats[side] = dict(raw=len(raw[side]), kept=len(kept), frames=len(frames), dropped=dropped,
                           runs=len(split_runs(kept, max_gap)), spikes=nfix,
                           sample_step_px_before=pre2, sample_step_px_after=med_step(sm, "kp2"),
                           frame_step_px=med_step(frames, "kp2"),
                           lag_samples=float(np.mean(lags)) if lags else 0.0)
    return out, stats


def to_arrays(stab, frame_times, scale=1.0):
    """Pack stabilized frames into dense per-frame arrays.

    Returns L/R_xy (N,21,2) pixels multiplied by ``scale``, L/R_mm (N,21,3)
    root-relative millimeters, and L/R_ok (N,) presence flags.
    """
    tf = np.asarray(frame_times, np.float64)
    N = len(tf)
    arrs = {}
    for side in ("L", "R"):
        xy = np.zeros((N, 21, 2), np.float16)
        mm = np.zeros((N, 21, 3), np.float16)
        ok = np.zeros(N, bool)
        for e in stab[side]:
            # repeated frame times (duplicated pictures) all receive the value
            lo = int(np.searchsorted(tf, e["t"] - 1e-6, side="left"))
            hi = int(np.searchsorted(tf, e["t"] + 1e-6, side="right"))
            if hi > lo:
                xy[lo:hi] = (e["kp2"] * scale).astype(np.float16)
                mm[lo:hi] = (e["kp3"] * 1000.0).astype(np.float16)
                ok[lo:hi] = True
        arrs[side + "_xy"], arrs[side + "_mm"], arrs[side + "_ok"] = xy, mm, ok
    return arrs


def main():
    ap = argparse.ArgumentParser(description="Stabilize run.py predictions onto video frame times and print statistics.")
    ap.add_argument("--preds", required=True, help="predictions.jsonl[.gz] from run.py")
    ap.add_argument("--timestamps", required=True, help="timestamps.json of the recording")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--max-jump", type=float, default=MAX_JUMP_PX)
    ap.add_argument("--win", type=float, default=SG_WINDOW_S)
    ap.add_argument("--poly", type=int, default=SG_POLY)
    a = ap.parse_args()
    with open(a.timestamps) as f:
        ft = json.load(f)["head"]
    _, st = build(load(a.preds), ft, a.min_score, a.max_jump, a.win, a.poly)
    for side in ("L", "R"):
        d = st[side]
        print(f"{side}: raw {d['raw']} -> kept {d['kept']} ({d['runs']} runs) -> {d['frames']}/{len(ft)} frames, "
              f"spikes repaired {d['spikes']}, lag {d['lag_samples']:+.2f} samples, "
              f"step {d['sample_step_px_before']:.1f} -> {d['sample_step_px_after']:.1f} px")


if __name__ == "__main__":
    main()
