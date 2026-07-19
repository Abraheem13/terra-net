"""DANN: gradient-reversal domain classifier over tile embeddings.

Aligns source (training cities) and target (deployment city, unlabeled tiles)
embeddings. Lambda is warmed up on the standard 2/(1+exp(-10p))-1 schedule.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class DomainDiscriminator(nn.Module):
    def __init__(self, emb_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor, lam: float) -> torch.Tensor:
        return self.net(grad_reverse(z, lam)).squeeze(-1)


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    import math
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
