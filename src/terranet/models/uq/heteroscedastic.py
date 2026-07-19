"""Gaussian NLL for heteroscedastic (gamma, PL0) prediction."""
from __future__ import annotations

import torch


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                 scale: torch.Tensor | None = None) -> torch.Tensor:
    """mean/logvar/target: (B, 2). Optional per-dim scale to balance gamma vs PL0."""
    if scale is not None:
        mean, target = mean / scale, target / scale
    inv = torch.exp(-logvar)
    return (0.5 * (inv * (target - mean) ** 2 + logvar)).mean()
