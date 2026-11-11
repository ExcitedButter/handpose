from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from WiLoR.detectors.yolo_world.blocks import ConvBNAct


class BNContrastiveHead(nn.Module):
    """BatchNorm contrastive region-text classifier used by YOLO-World v2."""

    def __init__(self, embed_dims: int = 512) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims, eps=1e-3, momentum=0.03)
        self.bias = nn.Parameter(torch.zeros(()))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones(()))

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        text_features = torch.nn.functional.normalize(text_features, dim=-1, p=2)
        logits = torch.einsum("bchw,bkc->bkhw", x, text_features)
        return logits * self.logit_scale.exp() + self.bias


class YOLOWorldHead(nn.Module):
    """Plain-PyTorch YOLO-World v2-m detection head.

    Forward returns the detector head contract used by the runtime pipeline:
    sigmoid class scores as ``B x anchors x classes`` and decoded xyxy boxes as
    ``B x anchors x 4`` in letterboxed-image pixels.
    """

    def __init__(self, variant: str = "m", reg_max: int = 16, strides: tuple[int, int, int] = (8, 16, 32)) -> None:
        super().__init__()
        if variant != "m":
            raise ValueError(f"Only YOLO-World v2-m head is ported so far, got {variant!r}")
        self.reg_max = int(reg_max)
        self.strides = tuple(int(s) for s in strides)
        in_channels = (192, 384, 576)
        self.cls_preds = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(channels, 192, 3),
                ConvBNAct(192, 192, 3),
                nn.Conv2d(192, 512, 1),
            )
            for channels in in_channels
        ])
        self.reg_preds = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(channels, 64, 3),
                ConvBNAct(64, 64, 3),
                nn.Conv2d(64, 4 * self.reg_max, 1),
            )
            for channels in in_channels
        ])
        self.cls_contrasts = nn.ModuleList(BNContrastiveHead(512) for _ in in_channels)
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float32), persistent=False)

    def forward(self, img_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor], txt_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_scores: list[torch.Tensor] = []
        bbox_preds: list[torch.Tensor] = []
        for img_feat, cls_pred, reg_pred, contrast in zip(img_feats, self.cls_preds, self.reg_preds, self.cls_contrasts):
            cls_embed = cls_pred(img_feat)
            cls_scores.append(contrast(cls_embed, txt_feats))
            bbox_preds.append(self._distribution_to_distance(reg_pred(img_feat)))
        return self.decode_outputs(cls_scores, bbox_preds)

    def _distribution_to_distance(self, bbox_dist: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = bbox_dist.shape
        dist = bbox_dist.reshape(batch, 4, self.reg_max, height * width).permute(0, 3, 1, 2)
        dist = dist.softmax(dim=3).matmul(self.proj.view(-1, 1)).squeeze(-1)
        return dist.transpose(1, 2).reshape(batch, 4, height, width)

    def decode_outputs(self, cls_scores: list[torch.Tensor], bbox_preds: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        flat_scores = []
        flat_boxes = []
        for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, self.strides):
            batch, _, height, width = bbox_pred.shape
            flat_scores.append(cls_score.permute(0, 2, 3, 1).reshape(batch, -1, cls_score.shape[1]).sigmoid())
            distances = bbox_pred.permute(0, 2, 3, 1).reshape(batch, -1, 4)
            priors = self._grid_priors(height, width, stride, bbox_pred.device, bbox_pred.dtype)
            flat_boxes.append(self._distance_to_bbox(priors, distances, stride))
        return torch.cat(flat_scores, dim=1), torch.cat(flat_boxes, dim=1)

    @staticmethod
    def _grid_priors(height: int, width: int, stride: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([(x.reshape(-1) + 0.5) * stride, (y.reshape(-1) + 0.5) * stride], dim=-1)

    @staticmethod
    def _distance_to_bbox(priors: torch.Tensor, distances: torch.Tensor, stride: int) -> torch.Tensor:
        distances = distances * stride
        x1y1 = priors[None, :, :] - distances[..., 0:2]
        x2y2 = priors[None, :, :] + distances[..., 2:4]
        return torch.cat([x1y1, x2y2], dim=-1)

    def load_prefixed_state_dict(self, state_dict: Mapping[str, torch.Tensor], prefix: str) -> tuple[list[str], list[str]]:
        local_state = OrderedDict(
            (key.removeprefix(prefix), value)
            for key, value in state_dict.items()
            if key.startswith(prefix)
        )
        incompatible = self.load_state_dict(local_state, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
