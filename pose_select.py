"""Pose hypotheses per labeled hand and their offline selection over the recording.

The pose network sometimes returns the mirror-image solution of a hand: the
palm side and the back side are swapped and the thumb lands on the wrong side
of the glove. This happens mostly for right hands seen edge-on or as a fist.
For every labeled hand box several pose hypotheses are computed:

  0. the labeled box (as before),
  1. a box refit to the keypoints of hypothesis 0 (tighter, centered crop),
  2. the refit box with the image rotated by ``ROT_DEG``,
  3. the labeled box with the opposite handedness flag (mirrored crop); its
     result is reflected across its own palm plane, which restores the hand's
     chirality and keeps the 2D projection, so it is the mirror-image
     alternative of the network's answer.

Each hypothesis gets two image scores: the share of points along its skeleton
that lie on white glove pixels, and the share of the white-glove area around
the hand that its rendered MANO mesh covers (a collapsed or mirrored pose
covers less of the glove). ``select`` then runs a Viterbi search per hand over
the whole recording: unary cost = poor glove agreement + poor glove coverage +
disagreement with the other hypotheses + palm toward the camera (rare in the
head-camera view; graded by how squarely the palm faces the camera, so an
edge-on hand is not pushed either way); pairwise cost = change of hand shape,
wrist jump, and palm-orientation flips between consecutive pose samples
(counted only when both orientations are clear, not for edge-on hands).
"""
import cv2
import numpy as np

ROT_DEG = 25.0
KP_RESCALE = 2.0
DET_RESCALE = 2.2
# selection weights
A_GLOVE = 4.0
B_CONS = 0.5
ORIENT_PRIOR = 2.0
G_SHAPE = 3.0
G_WRIST = 1.0
D_ORIENT = 2.0
C_COVER = 4.0        # weight of the mesh coverage of the glove
ORIENT_FULL = 0.5    # palm-to-camera cosine at which the orientation prior is fully applied
ORIENT_CLEAR = 0.4   # |cosine| below which a palm orientation counts as unclear (edge-on)
COVER_DS = 4         # raster downscale for the coverage score

EDGES = [(0, 1 + 4 * f) for f in range(5)] + [(1 + 4 * f + j, 2 + 4 * f + j) for f in range(5) for j in range(3)]
N_HYP = 4


def glove_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 70) & (hsv[..., 2] > 140)).astype(np.uint8)
    return cv2.dilate(m, np.ones((9, 9), np.uint8))


def glove_score(mask, kp):
    w = np.linspace(0.2, 0.8, 4)[:, None]
    pts = np.concatenate([kp] + [kp[a] * (1 - w) + kp[b] * w for a, b in EDGES])
    p = np.rint(pts).astype(int)
    h, wd = mask.shape
    inside = (p[:, 0] >= 0) & (p[:, 0] < wd) & (p[:, 1] >= 0) & (p[:, 1] < h)
    v = np.zeros(len(p))
    v[inside] = mask[p[inside, 1], p[inside, 0]]
    return float(v.mean())


def glove_cover(mask_ds, verts2d, faces, box):
    """Share of the white-glove pixels near the hand box that the rendered mesh covers."""
    h, w = mask_ds.shape
    sil = np.zeros_like(mask_ds)
    cv2.fillPoly(sil, list(np.rint(verts2d[faces] / COVER_DS).astype(np.int32)), 1)
    c, s = (box[:2] + box[2:]) / 2, (box[2:] - box[:2]) * 0.65
    x0, y0 = np.clip(((c - s) / COVER_DS).astype(int), 0, [w, h])
    x1, y1 = np.clip(((c + s) / COVER_DS).astype(int), 0, [w, h])
    g = mask_ds[y0:y1, x0:x1]
    return float((sil[y0:y1, x0:x1] & g).sum()) / max(float(g.sum()), 1.0)


def kp_box(kp):
    lo, hi = kp.min(0), kp.max(0)
    c, s = (lo + hi) / 2, (hi - lo).max() * 1.1 / 2
    return np.array([c[0] - s, c[1] - s, c[0] + s, c[1] + s], np.float32)


def palm_reflect(out, img_wh):
    """Reflect a pose across its palm plane (camera frame) and reproject joints and mesh."""
    J = out["joints_3d"].astype(np.float64) + out["cam_t"][None]
    V = out["vertices"].astype(np.float64) + out["cam_t"][None]
    P = J[[0, 5, 9, 13, 17]]
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    f = out["focal_length"]

    def refl(X):
        return X - 2 * ((X - c) @ n)[:, None] * n[None]

    def proj(X):
        return np.stack([f * X[:, 0] / X[:, 2] + img_wh[0] / 2, f * X[:, 1] / X[:, 2] + img_wh[1] / 2], 1).astype(np.float32)
    Jr, Vr = refl(J), refl(V)
    return proj(Jr), (Jr - out["cam_t"][None]).astype(np.float32), proj(Vr)


def hypotheses(extractor, image, boxes, right, batch_size=2):
    """Pose hypotheses for the labeled boxes of one frame.

    Returns (kp2 (n, N_HYP, 21, 2), kp3 (n, N_HYP, 21, 3) wrist-relative meters,
    evidence (n, N_HYP, 2) = glove point share and glove coverage, base results).
    """
    n = len(boxes)
    kp2 = np.zeros((n, N_HYP, 21, 2), np.float32)
    kp3 = np.zeros((n, N_HYP, 21, 3), np.float32)
    glove = np.zeros((n, N_HYP, 2), np.float32)
    if n == 0:
        return kp2, kp3, glove, []
    img_wh = (image.shape[1], image.shape[0])
    mask = glove_mask(image)
    mask_ds = cv2.resize(mask, (img_wh[0] // COVER_DS, img_wh[1] // COVER_DS), interpolation=cv2.INTER_NEAREST)
    faces = np.asarray(extractor.model.mano.faces, np.int64)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    right = np.asarray(right)
    base = extractor.predict_bboxes(image, boxes, right, batch_size=batch_size, rescale_factor=DET_RESCALE)
    kb = np.stack([kp_box(b["keypoints_2d"]) for b in base])
    refit = extractor.predict_bboxes(image, kb, right, batch_size=batch_size, rescale_factor=KP_RESCALE)
    opp = extractor.predict_bboxes(image, boxes, 1 - right, batch_size=batch_size, rescale_factor=DET_RESCALE)
    for i in range(n):
        res = [(o["keypoints_2d"], o["joints_3d"], o["vertices_2d"]) for o in (base[i], refit[i])]
        c = (kb[i, :2] + kb[i, 2:]) / 2
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ROT_DEG, 1.0)
        rot = cv2.warpAffine(image, M, img_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        r = extractor.predict_bboxes(rot, kb[i:i + 1], right[i:i + 1], batch_size=1, rescale_factor=KP_RESCALE)[0]
        Mi = cv2.invertAffineTransform(M)
        k3 = r["joints_3d"].copy()
        k3[:, :2] = k3[:, :2] @ Mi[:, :2].T
        res.append((r["keypoints_2d"] @ Mi[:, :2].T + Mi[:, 2], k3, r["vertices_2d"] @ Mi[:, :2].T + Mi[:, 2]))
        res.append(palm_reflect(opp[i], img_wh))
        for h, (k2, k3, v2) in enumerate(res):
            kp2[i, h] = k2
            kp3[i, h] = k3 - k3[:1]
            glove[i, h] = glove_score(mask, k2), glove_cover(mask_ds, v2, faces, boxes[i])
    return kp2, kp3, glove, base


def orientation(kp3, right):
    """Cosine between the palm normal and the camera axis: +1 back of the hand toward the camera, -1 palm."""
    n = np.cross(kp3[..., 5, :] - kp3[..., 0, :], kp3[..., 17, :] - kp3[..., 0, :])
    return n[..., 2] / (np.linalg.norm(n, axis=-1) + 1e-9) * (1 if right else -1)


def _shape(kp2):
    rel = kp2 - kp2[..., :1, :]
    s = np.linalg.norm(kp2[..., 9, :] - kp2[..., 0, :], axis=-1) + 1e-3
    return rel / s[..., None, None], s


def select(kp2, kp3, evidence, right):
    """Viterbi choice of one hypothesis per sample for one hand track.

    kp2 (T, H, 21, 2), kp3 (T, H, 21, 3), evidence (T, H, 2) for consecutive pose
    samples of one hand (rows may come from separate runs; call per run).
    Returns the chosen index per sample.
    """
    T, H = evidence.shape[:2]
    if T == 0:
        return np.zeros(0, int)
    glove, cover = evidence[..., 0], evidence[..., 1]
    C = orientation(kp3, right)
    O = np.sign(C)
    S, size = _shape(kp2)
    cons = np.linalg.norm(S - np.median(S, axis=1, keepdims=True), axis=-1).mean(-1)
    U = -A_GLOVE * glove - C_COVER * cover + B_CONS * cons + ORIENT_PRIOR * np.clip(-C / ORIENT_FULL, 0, 1)
    cost = U[0].copy()
    back = np.zeros((T, H), int)
    for t in range(1, T):
        dS = np.linalg.norm(S[t - 1][:, None] - S[t][None], axis=-1).mean(-1)
        sz = 0.5 * (size[t - 1][:, None] + size[t][None])
        dW = np.linalg.norm(kp2[t - 1][:, None, 0] - kp2[t][None, :, 0], axis=-1) / sz
        clear = np.minimum(np.abs(C[t - 1])[:, None], np.abs(C[t])[None])
        dO = (O[t - 1][:, None] != O[t][None]) * np.clip(clear / ORIENT_CLEAR, 0, 1)
        tot = cost[:, None] + G_SHAPE * dS + G_WRIST * np.minimum(dW, 3) + D_ORIENT * dO
        back[t] = tot.argmin(0)
        cost = tot.min(0) + U[t]
    choice = np.zeros(T, int)
    h = int(cost.argmin())
    for t in range(T - 1, -1, -1):
        choice[t] = h
        h = back[t, h]
    return choice
