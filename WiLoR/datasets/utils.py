from __future__ import annotations

import cv2
import numpy as np


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Expand bbox size `[w, h]` to match target aspect `[w, h]`."""
    if target_aspect_ratio is None:
        return input_shape
    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        return np.array([w, max(w * h_t / w_t, h)])
    return np.array([max(h * w_t / h_t, w), h])


def _rotate_2d(pt: np.ndarray, rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs], dtype=np.float32)


def _affine_from_patch(
    c_x: float,
    c_y: float,
    src_width: float,
    src_height: float,
    dst_width: float,
    dst_height: float,
    scale: float,
    rot: float,
) -> np.ndarray:
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180.0
    src_down = _rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_right = _rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_down = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_right = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.stack([src_center, src_center + src_down, src_center + src_right]).astype(np.float32)
    dst = np.stack([dst_center, dst_center + dst_down, dst_center + dst_right]).astype(np.float32)
    return cv2.getAffineTransform(src, dst)


def generate_image_patch_cv2(
    img: np.ndarray,
    c_x: float,
    c_y: float,
    bb_width: float,
    bb_height: float,
    patch_width: float,
    patch_height: float,
    do_flip: bool,
    scale: float,
    rot: float,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop/resize a hand bbox into the ViT input patch."""
    img_height, img_width, img_channels = img.shape
    del img_height, img_channels
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = _affine_from_patch(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)
    img_patch = cv2.warpAffine(
        img,
        trans,
        (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )
    if img.shape[2] == 4 and border_mode != cv2.BORDER_CONSTANT:
        img_patch[:, :, 3] = cv2.warpAffine(
            img[:, :, 3],
            trans,
            (int(patch_width), int(patch_height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    return img_patch, trans


def convert_cvimg_to_tensor(cvimg: np.ndarray) -> np.ndarray:
    """Convert HWC uint8/float image to CHW float32."""
    return np.transpose(cvimg.copy(), (2, 0, 1)).astype(np.float32)
