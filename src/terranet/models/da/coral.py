"""Deep CORAL loss: match second-order statistics of source/target embeddings."""
from __future__ import annotations

import torch


def coral_loss(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    d = zs.size(1)

    def cov(z):
        zc = z - z.mean(0, keepdim=True)
        return (zc.t() @ zc) / max(z.size(0) - 1, 1)

    return ((cov(zs) - cov(zt)) ** 2).sum() / (4 * d * d)
