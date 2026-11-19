"""Camera models and per-hand perspective views for wide-angle (fisheye) video.

The pose network expects a perspective crop centered on the hand. On a
fisheye image a hand away from the optical axis is sheared and compressed, so
each hand is first re-rendered through a virtual pinhole camera that looks
straight at it. Predictions made in that view are mapped back through the
real camera model, which also yields viewing rays for metric 3D placement.

Coordinates: +X right, +Y down, +Z forward; pixels (u, v) = (column, row).
"""
import json
from pathlib import Path

import cv2
import numpy as np


class DoubleSphere:
    """Double Sphere camera model (Usenko et al., 2018)."""

    def __init__(self, fx, fy, cx, cy, xi, alpha, width, height):
        self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha = fx, fy, cx, cy, xi, alpha
        self.width, self.height = int(width), int(height)

    def project(self, X):
        """(..., 3) points -> (..., 2) pixels; NaN where the point is outside the model's field of view."""
        X = np.asarray(X, np.float64)
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        d1 = np.sqrt(x * x + y * y + z * z)
        k = self.xi * d1 + z
        d2 = np.sqrt(x * x + y * y + k * k)
        den = self.alpha * d2 + (1 - self.alpha) * k
        a = self.alpha
        w1 = a / (1 - a) if a <= 0.5 else (1 - a) / a
        w2 = (w1 + self.xi) / np.sqrt(2 * w1 * self.xi + self.xi ** 2 + 1)
        ok = (z > -w2 * d1) & (den > 1e-9)
        den = np.where(ok, den, np.nan)
        return np.stack([self.fx * x / den + self.cx, self.fy * y / den + self.cy], -1)

    def unproject(self, uv):
        """(..., 2) pixels -> (..., 3) unit rays (NaN outside the model)."""
        uv = np.asarray(uv, np.float64)
        mx, my = (uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy
        r2 = mx * mx + my * my
        a, xi = self.alpha, self.xi
        s = 1 - (2 * a - 1) * r2
        s = np.where(s >= 0, s, np.nan)
        mz = (1 - a * a * r2) / (a * np.sqrt(s) + 1 - a)
        f = (mz * xi + np.sqrt(mz * mz + (1 - xi * xi) * r2)) / (mz * mz + r2)
        ray = np.stack([f * mx, f * my, f * mz - xi], -1)
        return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


class Pinhole:
    """Distortion-free camera; used for videos without calibration (assumed focal length)."""

    def __init__(self, f, cx, cy, width, height):
        self.f, self.cx, self.cy, self.width, self.height = float(f), float(cx), float(cy), int(width), int(height)

    def project(self, X):
        X = np.asarray(X, np.float64)
        z = np.where(X[..., 2] > 1e-9, X[..., 2], np.nan)
        return np.stack([self.f * X[..., 0] / z + self.cx, self.f * X[..., 1] / z + self.cy], -1)

    def unproject(self, uv):
        uv = np.asarray(uv, np.float64)
        ray = np.stack([(uv[..., 0] - self.cx) / self.f, (uv[..., 1] - self.cy) / self.f, np.ones(uv.shape[:-1])], -1)
        return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


def load_cameras(camera_json):
    """NW_bench ``camera.json`` -> ({name: DoubleSphere}, {name: 4x4 T_reference_camera}, reference name)."""
    with open(Path(camera_json)) as f:
        d = json.load(f)
    cams, T = {}, {}
    for name, c in d["cameras"].items():
        if c.get("model") != "double_sphere":
            raise ValueError(f"{camera_json}: unsupported camera model {c.get('model')!r}")
        i = c["intrinsics"]
        cams[name] = DoubleSphere(i["fx"], i["fy"], i["cx"], i["cy"], i["xi"], i["alpha"], c["width"], c["height"])
        T[name] = np.asarray(c.get("T_reference_camera", np.eye(4)), np.float64).reshape(4, 4)
    return cams, T, d.get("reference_camera", sorted(cams)[-1])


class HandView:
    """Virtual pinhole camera aimed at one hand box of the real image."""

    def __init__(self, cam, box, size=640, fill=0.34):
        """``fill``: share of the view's side taken by the hand box."""
        self.cam, self.size = cam, int(size)
        x0, y0, x1, y1 = [float(v) for v in box]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rays = cam.unproject(np.array([[cx, cy], [x0, cy], [x1, cy], [cx, y0], [cx, y1]]))
        if not np.isfinite(rays).all():
            raise ValueError("hand box outside the camera model")
        d = rays[0]
        aw = np.arccos(np.clip(rays[1] @ rays[2], -1, 1))
        ah = np.arccos(np.clip(rays[3] @ rays[4], -1, 1))
        half = self.size * fill / 2
        self.f = float(half / np.tan(max(aw, ah, 1e-3) / 2))
        # Minimal roll: the view's x axis stays as close as possible to the camera's x axis.
        x = np.array([1., 0., 0.]) - d[0] * d
        x /= np.linalg.norm(x)
        self.R = np.stack([x, np.cross(d, x), d], 1)          # view -> camera
        hw, hh = self.f * np.tan(aw / 2), self.f * np.tan(ah / 2)
        c = self.size / 2
        self.box = np.array([c - hw, c - hh, c + hw, c + hh], np.float32)

    def render(self, image):
        c = (self.size - 1) / 2
        u, v = np.meshgrid(np.arange(self.size, dtype=np.float64), np.arange(self.size, dtype=np.float64))
        rays = np.stack([(u - c) / self.f, (v - c) / self.f, np.ones_like(u)], -1) @ self.R.T
        m = self.cam.project(rays)
        m = np.where(np.isfinite(m), m, -10).astype(np.float32)
        return cv2.remap(image, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def to_image(self, kp2):
        """(..., 2) view pixels -> real image pixels."""
        c = (self.size - 1) / 2
        kp2 = np.asarray(kp2, np.float64)
        rays = np.stack([(kp2[..., 0] - c) / self.f, (kp2[..., 1] - c) / self.f, np.ones(kp2.shape[:-1])], -1)
        return self.cam.project(rays @ self.R.T)

    def to_camera(self, X):
        """(..., 3) vectors in the view frame -> camera frame."""
        return np.asarray(X, np.float64) @ self.R.T


def solve_translation(rel, rays, origins=None, weights=None):
    """Place a rigid point set on viewing rays.

    ``rel`` (n, 3) points relative to their root, ``rays`` (n, 3) unit rays,
    ``origins`` (n, 3) ray origins (default: camera center). Returns the
    translation t minimizing the squared distances of ``rel + t`` to the rays,
    and the RMS distance (same unit as ``rel``).
    """
    rel, rays = np.asarray(rel, np.float64), np.asarray(rays, np.float64)
    o = np.zeros_like(rel) if origins is None else np.asarray(origins, np.float64)
    w = np.ones(len(rel)) if weights is None else np.asarray(weights, np.float64)
    P = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
    P = P * w[:, None, None]
    A = P.sum(0)
    b = np.einsum("nij,nj->i", P, o - rel)
    t = np.linalg.solve(A + 1e-12 * np.eye(3), b)
    r = np.einsum("nij,nj->ni", P, rel + t - o)
    return t, float(np.sqrt((r * r).sum(1).sum() / max(w.sum(), 1e-9)))
