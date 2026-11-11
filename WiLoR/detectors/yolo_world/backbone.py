from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from WiLoR.detectors.yolo_world.blocks import CSPLayerWithTwoConv, ConvBNAct, SPPFBottleneck


class YOLOWorldCSPDarknetBackbone(nn.Module):
    """Plain-PyTorch YOLOv8-CSPDarknet image backbone for YOLO-World v2-m.

    This ports the image backbone only.  It returns the three feature levels
    consumed by the YOLO-World PAFPN neck: strides 8, 16, and 32.
    """

    def __init__(self, variant: str = "m") -> None:
        super().__init__()
        if variant != "m":
            raise ValueError(f"Only YOLO-World v2-m is ported so far, got {variant!r}")
        self.stem = ConvBNAct(3, 48, 3, stride=2)
        self.stage1 = nn.Sequential(
            ConvBNAct(48, 96, 3, stride=2),
            CSPLayerWithTwoConv(96, 96, num_blocks=2, add_identity=True),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(96, 192, 3, stride=2),
            CSPLayerWithTwoConv(192, 192, num_blocks=4, add_identity=True),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(192, 384, 3, stride=2),
            CSPLayerWithTwoConv(384, 384, num_blocks=4, add_identity=True),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(384, 576, 3, stride=2),
            CSPLayerWithTwoConv(576, 576, num_blocks=2, add_identity=True),
            SPPFBottleneck(576, 576),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5

    def load_prefixed_state_dict(self, state_dict: Mapping[str, torch.Tensor], prefix: str) -> tuple[list[str], list[str]]:
        local_state = OrderedDict(
            (key.removeprefix(prefix), value)
            for key, value in state_dict.items()
            if key.startswith(prefix)
        )
        incompatible = self.load_state_dict(local_state, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
