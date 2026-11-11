from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from .config import load_config, recursive_to
from .datasets.vitdet_dataset import ViTDetDataset
from .models.wilor import WiLoRModel


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


from WiLoR.assets import weights_dir

_IGNORED_PREFIXES = ("mano.", "discriminator.")
_DEFAULT_CKPT = weights_dir() / "wilor" / "anyhand_wilor_nomano.ckpt"
_DEFAULT_CFG = weights_dir() / "wilor" / "anyhand_model_config_wilor.yaml"


def _load_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)!r}")
    # Accept Lightning/DataParallel-style prefixes.
    out = {}
    for k, v in sd.items():
        for prefix in ("model.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


@contextmanager
def _skip_parameter_init():
    """Make torch.nn.init functions no-ops while the module tree is built.

    Every learned parameter is overwritten by the checkpoint (and verified
    below), so random initialization is wasted work.
    """
    names = ("trunc_normal_", "normal_", "uniform_", "constant_", "zeros_", "ones_",
             "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_")
    saved = {n: getattr(torch.nn.init, n) for n in names}
    try:
        for n in names:
            setattr(torch.nn.init, n, lambda tensor, *a, **k: tensor)
        yield
    finally:
        for n, fn in saved.items():
            setattr(torch.nn.init, n, fn)


def load_wilor(
    checkpoint_path: str | Path | None = None,
    cfg_path: str | Path | None = None,
    mano_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[WiLoRModel, Any]:
    """Instantiate WiLoR as a plain torch.nn.Module and load a checkpoint."""
    cfg = load_config(cfg_path or _DEFAULT_CFG, mano_dir=mano_dir)
    with _skip_parameter_init():
        model = WiLoRModel(cfg)
    sd = _load_state_dict(checkpoint_path or _DEFAULT_CKPT)
    # The MANO layer is built from weights/mano/MANO_RIGHT.pt (non-persistent
    # buffers) and the adversarial discriminator is training-only, so entries
    # under these prefixes are ignored. Every other key must match exactly.
    sd = {k: v for k, v in sd.items() if not k.startswith(_IGNORED_PREFIXES)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.checkpoint_report = {"missing_keys": [], "unexpected_keys": [], "ignored_prefixes": list(_IGNORED_PREFIXES)}
    if dtype is not None:
        model = model.to(dtype=dtype)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, cfg


def preprocess_bboxes(
    cfg,
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    right: np.ndarray,
    *,
    rescale_factor: float = 2.0,
    device: str | torch.device | None = None,
    fp16: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert an OpenCV BGR image and hand boxes into a WiLoR inference batch."""
    ds = ViTDetDataset(cfg, image_bgr, np.asarray(boxes_xyxy), np.asarray(right), rescale_factor=rescale_factor, fp16=fp16)
    items = [ds[i] for i in range(len(ds))]
    if not items:
        return {}
    batch: dict[str, torch.Tensor] = {}
    for key in items[0].keys():
        vals = [it[key] for it in items]
        if isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        else:
            batch[key] = torch.as_tensor(np.stack(vals))
    if device is not None:
        batch = recursive_to(batch, torch.device(device))
    return batch


def project_full_img(points: np.ndarray, cam_trans: np.ndarray, focal_length: float, img_res: np.ndarray) -> np.ndarray:
    camera_center = np.array([img_res[0] / 2.0, img_res[1] / 2.0], dtype=np.float32)
    points = points + cam_trans[None, :]
    points = points / points[..., -1:]
    out = np.empty((points.shape[0], 2), dtype=np.float32)
    out[:, 0] = focal_length * points[:, 0] + camera_center[0]
    out[:, 1] = focal_length * points[:, 1] + camera_center[1]
    return out


class WiLoRPoseExtractor:
    """Small inference wrapper for hand pose extraction from known boxes.

    Detector integration is intentionally optional: pass boxes from any detector
    as xyxy pixels and `right` handedness labels where 1=right hand, 0=left hand.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        cfg_path: str | Path | None = None,
        mano_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        fp16: bool = False,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16
        dtype = torch.float16 if fp16 else None
        self.model, self.cfg = load_wilor(checkpoint_path, cfg_path, mano_dir, self.device, dtype=dtype)

    @torch.inference_mode()
    def predict_bboxes(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        right: np.ndarray,
        *,
        rescale_factor: float = 2.0,
        batch_size: int = 16,
    ) -> list[dict[str, np.ndarray | float | int]]:
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        if len(boxes_xyxy) == 0:
            return []
        results = []
        for start in range(0, len(boxes_xyxy), batch_size):
            boxes = boxes_xyxy[start:start + batch_size]
            handed = right[start:start + batch_size]
            batch = preprocess_bboxes(self.cfg, image_bgr, boxes, handed, rescale_factor=rescale_factor, device=self.device, fp16=self.fp16)
            out = self.model(batch)

            pred_cam = out["pred_cam"].clone()
            multiplier = (2 * batch["right"] - 1).to(pred_cam.device, pred_cam.dtype)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

            verts = out["pred_vertices"].detach().float().cpu().numpy()
            joints = out["pred_keypoints_3d"].detach().float().cpu().numpy()
            cam_t = pred_cam_t_full.detach().float().cpu().numpy()
            img_size_np = img_size.detach().float().cpu().numpy()
            focal = float(scaled_focal_length.detach().float().cpu().item())
            handed_np = batch["right"].detach().float().cpu().numpy()
            pred_cam_np = pred_cam.detach().float().cpu().numpy()
            mano_params = {k: v.detach().float().cpu().numpy() for k, v in out["pred_mano_params"].items()}

            for i in range(len(boxes)):
                sign = 2 * handed_np[i] - 1
                vi = verts[i].copy(); ji = joints[i].copy()
                vi[:, 0] = sign * vi[:, 0]
                ji[:, 0] = sign * ji[:, 0]
                results.append({
                    "vertices": vi,
                    "joints_3d": ji,
                    "keypoints_2d": project_full_img(ji, cam_t[i], focal, img_size_np[i]),
                    "vertices_2d": project_full_img(vi, cam_t[i], focal, img_size_np[i]),
                    "cam_t": cam_t[i],
                    "pred_cam": pred_cam_np[i],
                    "bbox": boxes[i].copy(),
                    "right": int(handed_np[i]),
                    "focal_length": focal,
                    # Canonical right-hand model parameters, including for the
                    # flipped left crop. Geometry above is already unmirrored.
                    "mano_global_orient": mano_params["global_orient"][i],
                    "mano_hand_pose": mano_params["hand_pose"][i],
                    "mano_betas": mano_params["betas"][i],
                })
        return results

    def predict_image_path(self, image_path: str | Path, boxes_xyxy: np.ndarray, right: np.ndarray, **kwargs):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        return self.predict_bboxes(img, boxes_xyxy, right, **kwargs)
