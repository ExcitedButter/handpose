from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    """MMCV-style ConvModule used by YOLO-World inference blocks.

    The official YOLO-World v2 checkpoints use conv-without-bias, BatchNorm2d
    with eps=1e-3/momentum=0.03, and SiLU activation for these image blocks.
    Attribute names intentionally match checkpoint keys: ``conv`` and ``bn``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None, act: bool = True) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DarknetBottleneck(nn.Module):
    def __init__(self, channels: int, add_identity: bool = True) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = ConvBNAct(channels, channels, 3)
        self.add_identity = bool(add_identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.add_identity else out


class CSPLayerWithTwoConv(nn.Module):
    """YOLOv8 C2f/CSP layer matching MMYOLO ``CSPLayerWithTwoConv`` keys."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, add_identity: bool = True) -> None:
        super().__init__()
        self.mid_channels = out_channels // 2
        self.main_conv = ConvBNAct(in_channels, 2 * self.mid_channels, 1, padding=0)
        self.blocks = nn.ModuleList(DarknetBottleneck(self.mid_channels, add_identity=add_identity) for _ in range(num_blocks))
        self.final_conv = ConvBNAct((2 + num_blocks) * self.mid_channels, out_channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.main_conv(x).split((self.mid_channels, self.mid_channels), dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.final_conv(torch.cat(parts, dim=1))


class SPPFBottleneck(nn.Module):
    """SPPF block used as YOLOv8 backbone stage4.2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, mid_channels, 1, padding=0)
        self.poolings = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = ConvBNAct(mid_channels * 4, out_channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.poolings(x)
        y2 = self.poolings(y1)
        y3 = self.poolings(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class MaxSigmoidAttnBlock(nn.Module):
    """YOLO-World max-sigmoid image/text attention block."""

    def __init__(self, in_channels: int, out_channels: int, guide_channels: int, embed_channels: int, num_heads: int) -> None:
        super().__init__()
        if out_channels % num_heads != 0 or embed_channels % num_heads != 0:
            raise ValueError("out_channels and embed_channels must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_channels = embed_channels // num_heads
        self.embed_conv = ConvBNAct(in_channels, embed_channels, 1, padding=0, act=False) if embed_channels != in_channels else None
        self.guide_fc = nn.Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.scale = 1.0
        self.project_conv = ConvBNAct(in_channels, out_channels, 3, padding=1, act=False)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        guide = self.guide_fc(guide).reshape(batch, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(batch, self.num_heads, self.head_channels, height, width)
        attn_weight = torch.einsum("bmchw,bnmc->bmhwn", embed, guide).max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels ** 0.5)
        attn_weight = (attn_weight + self.bias[None, :, None, None]).sigmoid() * self.scale
        x = self.project_conv(x).reshape(batch, self.num_heads, -1, height, width)
        x = x * attn_weight.unsqueeze(2)
        return x.reshape(batch, -1, height, width)


class MaxSigmoidCSPLayerWithTwoConv(nn.Module):
    """YOLO-World CSP layer with text-guided max-sigmoid attention."""

    def __init__(self, in_channels: int, out_channels: int, guide_channels: int, embed_channels: int, num_heads: int, num_blocks: int) -> None:
        super().__init__()
        self.mid_channels = out_channels // 2
        self.main_conv = ConvBNAct(in_channels, 2 * self.mid_channels, 1, padding=0)
        self.blocks = nn.ModuleList(DarknetBottleneck(self.mid_channels, add_identity=False) for _ in range(num_blocks))
        self.final_conv = ConvBNAct((3 + num_blocks) * self.mid_channels, out_channels, 1, padding=0)
        self.attn_block = MaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        parts = list(self.main_conv(x).split((self.mid_channels, self.mid_channels), dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        parts.append(self.attn_block(parts[-1], guide))
        return self.final_conv(torch.cat(parts, dim=1))
