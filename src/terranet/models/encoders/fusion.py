"""Fusion of hand-crafted descriptor + learned raster embedding (+ optional GNN).

Gated fusion so ablations "descriptor-only" / "raster-only" are the same module
with a branch disabled — apples-to-apples comparisons.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_tile import TileCNN


class FusionEncoder(nn.Module):
    def __init__(self, desc_dim: int = 116, emb_dim: int = 256,
                 use_descriptor: bool = True, use_raster: bool = True):
        super().__init__()
        assert use_descriptor or use_raster
        self.use_descriptor, self.use_raster = use_descriptor, use_raster
        if use_descriptor:
            self.desc_proj = nn.Sequential(nn.LayerNorm(desc_dim),
                                           nn.Linear(desc_dim, emb_dim), nn.GELU())
        if use_raster:
            self.cnn = TileCNN(emb_dim=emb_dim)
        if use_descriptor and use_raster:
            self.gate = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, desc: torch.Tensor, raster: torch.Tensor) -> torch.Tensor:
        if self.use_descriptor and self.use_raster:
            zd, zr = self.desc_proj(desc), self.cnn(raster)
            g = self.gate(torch.cat([zd, zr], dim=-1))
            z = g * zd + (1 - g) * zr
        elif self.use_descriptor:
            z = self.desc_proj(desc)
        else:
            z = self.cnn(raster)
        return self.norm(z)
