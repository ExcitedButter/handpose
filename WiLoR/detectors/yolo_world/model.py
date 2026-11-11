from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from WiLoR.detectors.yolo_world.backbone import YOLOWorldCSPDarknetBackbone
from WiLoR.detectors.yolo_world.head import YOLOWorldHead
from WiLoR.detectors.yolo_world.neck import YOLOWorldPAFPN


class YOLOWorldModel(nn.Module):
    """Plain-PyTorch YOLO-World v2-m image model with external text features."""

    def __init__(self, variant: str = "m") -> None:
        super().__init__()
        self.backbone = YOLOWorldCSPDarknetBackbone(variant=variant)
        self.neck = YOLOWorldPAFPN(variant=variant)
        self.head = YOLOWorldHead(variant=variant)

    def forward(self, image: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.backbone(image)
        neck_features = self.neck(image_features, text_features)
        return self.head(neck_features, text_features)

    def load_yolo_world_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        unexpected: list[str] = []
        for module_name, module, prefix in (
            ("backbone", self.backbone, "backbone.image_model."),
            ("neck", self.neck, "neck."),
            ("head", self.head, "bbox_head.head_module."),
        ):
            module_missing, module_unexpected = module.load_prefixed_state_dict(state_dict, prefix=prefix)
            missing.extend(f"{module_name}.{key}" for key in module_missing)
            unexpected.extend(f"{module_name}.{key}" for key in module_unexpected)
        return missing, unexpected
