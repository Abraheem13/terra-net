"""CNN encoder over rasterized tile geometry.

Input raster channels (C=4): building height AGL, terrain elevation,
vegetation mask, water mask; H=W=128 at (tile_size/128) m/pixel.
A ResNet-ish stack with GroupNorm (small-batch friendly) -> embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, cout)
        self.skip = (nn.Conv2d(cin, cout, 1, stride, bias=False)
                     if (cin != cout or stride != 1) else nn.Identity())
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + self.skip(x))


class TileCNN(nn.Module):
    def __init__(self, in_ch: int = 4, width: int = 64, emb_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 5, 2, 2, bias=False),
                                  nn.GroupNorm(8, width), nn.GELU())
        self.stages = nn.Sequential(
            ConvBlock(width, width),
            ConvBlock(width, width * 2, stride=2),
            ConvBlock(width * 2, width * 2),
            ConvBlock(width * 2, width * 4, stride=2),
            ConvBlock(width * 4, width * 4),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(width * 4, emb_dim))

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(raster)))
