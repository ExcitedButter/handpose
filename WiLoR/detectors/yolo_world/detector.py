from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from WiLoR.detectors.yolo_world.model import YOLOWorldModel
from WiLoR.detectors.yolo_world.text import YOLOWorldTextEncoder


from WiLoR.assets import weights_dir

DEFAULT_YOLO_WORLD_DETECTOR = (
    weights_dir()
    / "yolo_world"
    / "yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth"
)


@dataclass(frozen=True)
class YOLOWorldCheckpointInfo:
    """Lightweight summary of an official AILab-CVC YOLO-World checkpoint."""

    path: Path
    num_tensors: int
    image_stem_channels: int
    text_channels: int
    neck_out_channels: tuple[int, ...]
    head_levels: int


def _state_dict_from_checkpoint(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected state_dict mapping, got {type(state_dict)!r}")
    return state_dict  # type: ignore[return-value]


def load_yolo_world_checkpoint_state_dict(path: str | Path = DEFAULT_YOLO_WORLD_DETECTOR) -> Mapping[str, torch.Tensor]:
    """Load the official YOLO-World checkpoint state_dict without MMEngine."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    return _state_dict_from_checkpoint(checkpoint)


def inspect_yolo_world_checkpoint(path: str | Path = DEFAULT_YOLO_WORLD_DETECTOR) -> YOLOWorldCheckpointInfo:
    """Inspect an official YOLO-World PyTorch checkpoint without MMEngine."""

    path = Path(path)
    state_dict = load_yolo_world_checkpoint_state_dict(path)

    stem = state_dict["backbone.image_model.stem.conv.weight"]
    image_stem_channels = int(stem.shape[0])

    text_projection = state_dict["backbone.text_model.model.text_projection.weight"]
    text_channels = int(text_projection.shape[0])

    neck_out_channels: list[int] = []
    for key in state_dict.keys():
        if key.startswith("neck.") and key.endswith("main_conv.conv.weight"):
            neck_out_channels.append(int(state_dict[key].shape[0]))

    head_levels = len({
        key.split(".")[3]
        for key in state_dict.keys()
        if key.startswith("bbox_head.head_module.cls_preds.")
    })

    return YOLOWorldCheckpointInfo(
        path=path,
        num_tensors=len(state_dict),
        image_stem_channels=image_stem_channels,
        text_channels=text_channels,
        neck_out_channels=tuple(neck_out_channels),
        head_levels=head_levels,
    )


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int
    width: int
    height: int


def letterbox_image(image_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    height, width = image_bgr.shape[:2]
    scale = min(size / float(width), size / float(height))
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, LetterboxMeta(scale=scale, pad_x=pad_x, pad_y=pad_y, width=width, height=height)


def scale_boxes_to_original(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, meta.width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, meta.height)
    return out


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[idx] + areas[rest] - inter
        order = rest[inter / np.maximum(union, 1e-6) <= iou]
    return keep


class YOLOWorldDetector:
    """YOLO-World detector backend.

    The repo's detector stack is PyTorch-only, so the backend name no longer
    carries a torch suffix.  The detector returns raw box-score rows:
    ``x1, y1, x2, y2, score``.
    """

    def __init__(
        self,
        detector_path: str | Path | None = None,
        *,
        prompts: Sequence[str] = ("white glove", "hand"),
        input_size: int = 640,
        max_detections: int = 8,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        detector_path = Path(detector_path) if detector_path is not None else DEFAULT_YOLO_WORLD_DETECTOR
        if not detector_path.is_file():
            raise FileNotFoundError(f"Missing YOLO-World checkpoint: {detector_path}. See weights/README.md for download instructions.")
        state_dict = load_yolo_world_checkpoint_state_dict(detector_path)
        self.model = YOLOWorldModel(variant="m").eval().to(self.device)
        missing, unexpected = self.model.load_yolo_world_state_dict(state_dict)
        if missing or unexpected:
            raise RuntimeError(f"YOLO-World model checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.text_encoder = YOLOWorldTextEncoder(device=self.device).eval()
        missing, unexpected = self.text_encoder.load_yolo_world_state_dict(state_dict)
        if missing or unexpected:
            raise RuntimeError(f"YOLO-World text checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.prompts = tuple(prompts)
        self.input_size = int(input_size)
        self.max_detections = int(max_detections)
        self.text_features = self.text_encoder.encode(self.prompts).to(self.device)

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray, *, conf: float = 0.2, iou: float = 0.5) -> np.ndarray:
        boxed, meta = letterbox_image(image_bgr, self.input_size)
        image_rgb = boxed[:, :, ::-1].copy()
        image = torch.from_numpy(image_rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        scores_t, boxes_t = self.model(image, self.text_features)
        scores = scores_t[0].detach().cpu().numpy()
        boxes = boxes_t[0].detach().cpu().numpy().astype(np.float32)
        best_scores = scores.max(axis=1).astype(np.float32)
        keep_mask = best_scores >= float(conf)
        if not np.any(keep_mask):
            return np.zeros((0, 5), dtype=np.float32)
        selected_boxes = scale_boxes_to_original(boxes[keep_mask], meta)
        selected_scores = best_scores[keep_mask]
        keep = nms_numpy(selected_boxes, selected_scores, float(iou))[: self.max_detections]
        selected_boxes = selected_boxes[keep]
        selected_scores = selected_scores[keep]
        return np.concatenate([selected_boxes.astype(np.float32), selected_scores[:, None]], axis=1)
