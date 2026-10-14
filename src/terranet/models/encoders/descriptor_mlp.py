"""Descriptor MLP encoder: tile descriptor (32 features) -> embedding.

Deliberately small: the input is already engineered features, not raw pixels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DescriptorEncoder(nn.Module):
    def __init__(self, in_dim: int = 32, emb_dim: int = 256, hidden: int = 256,
                 depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, emb_dim))
        self.net = nn.Sequential(*layers)
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
