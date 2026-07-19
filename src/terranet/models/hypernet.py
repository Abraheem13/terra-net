"""Hypernetwork head: tile embedding -> (gamma, PL0) + heteroscedastic log-variances.

Outputs are physically constrained: gamma in [1.0, 6.0] via scaled sigmoid,
PL0 in [0, 60] dB. Replaces the fixed Gaussian-kernel transfer of the base paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

GAMMA_RANGE = (1.0, 6.0)
PL0_RANGE = (0.0, 60.0)


class ParamHead(nn.Module):
    def __init__(self, emb_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 4),  # mu_gamma, mu_pl0, logvar_gamma, logvar_pl0
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(z)
        g_lo, g_hi = GAMMA_RANGE
        p_lo, p_hi = PL0_RANGE
        gamma = g_lo + (g_hi - g_lo) * torch.sigmoid(raw[:, 0])
        pl0 = p_lo + (p_hi - p_lo) * torch.sigmoid(raw[:, 1])
        logvar = raw[:, 2:4].clamp(-8, 4)
        return {"mean": torch.stack([gamma, pl0], dim=-1), "logvar": logvar}
