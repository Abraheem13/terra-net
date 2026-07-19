"""Plain MLP on the 116-dim descriptor -> (gamma, pl0). Sanity-check ML baseline."""
from __future__ import annotations

import torch
import torch.nn as nn


class DescriptorMLP(nn.Module):
    def __init__(self, in_dim: int = 116, hidden: int = 256, depth: int = 3, out_dim: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
