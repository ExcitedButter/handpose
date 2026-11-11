from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from WiLoR.detectors.yolo_world.blocks import ConvBNAct, MaxSigmoidCSPLayerWithTwoConv


class YOLOWorldPAFPN(nn.Module):
    """Plain-PyTorch YOLO-World v2-m PAFPN neck."""

    def __init__(self, variant: str = "m") -> None:
        super().__init__()
        if variant != "m":
            raise ValueError(f"Only YOLO-World v2-m neck is ported so far, got {variant!r}")
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_layers = nn.ModuleList([
            MaxSigmoidCSPLayerWithTwoConv(960, 384, guide_channels=512, embed_channels=192, num_heads=6, num_blocks=2),
            MaxSigmoidCSPLayerWithTwoConv(576, 192, guide_channels=512, embed_channels=96, num_heads=3, num_blocks=2),
        ])
        self.downsample_layers = nn.ModuleList([
            ConvBNAct(192, 192, 3, stride=2),
            ConvBNAct(384, 384, 3, stride=2),
        ])
        self.bottom_up_layers = nn.ModuleList([
            MaxSigmoidCSPLayerWithTwoConv(576, 384, guide_channels=512, embed_channels=192, num_heads=6, num_blocks=2),
            MaxSigmoidCSPLayerWithTwoConv(960, 576, guide_channels=512, embed_channels=288, num_heads=9, num_blocks=2),
        ])

    def forward(self, img_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor], txt_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = img_feats
        p5 = c5
        p4 = self.top_down_layers[0](torch.cat([self.upsample(p5), c4], dim=1), txt_feats)
        p3 = self.top_down_layers[1](torch.cat([self.upsample(p4), c3], dim=1), txt_feats)
        n3 = p3
        n4 = self.bottom_up_layers[0](torch.cat([self.downsample_layers[0](n3), p4], dim=1), txt_feats)
        n5 = self.bottom_up_layers[1](torch.cat([self.downsample_layers[1](n4), p5], dim=1), txt_feats)
        return n3, n4, n5

    def load_prefixed_state_dict(self, state_dict: Mapping[str, torch.Tensor], prefix: str) -> tuple[list[str], list[str]]:
        local_state = OrderedDict(
            (key.removeprefix(prefix), value)
            for key, value in state_dict.items()
            if key.startswith(prefix)
        )
        incompatible = self.load_state_dict(local_state, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
