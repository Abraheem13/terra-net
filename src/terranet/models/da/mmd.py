"""Multi-kernel MMD^2 between source and target embeddings (RBF mixture)."""
from __future__ import annotations

import torch


def _rbf(a: torch.Tensor, b: torch.Tensor, sigmas=(1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    d2 = torch.cdist(a, b) ** 2
    return sum(torch.exp(-d2 / (2 * s ** 2)) for s in sigmas) / len(sigmas)


def mmd2(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    return _rbf(zs, zs).mean() + _rbf(zt, zt).mean() - 2 * _rbf(zs, zt).mean()
