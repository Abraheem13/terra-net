"""Composite training loss: heteroscedastic NLL + physics residual + DA penalty.

Physics residual: predicted (gamma, PL0) must reproduce raw per-tile measurements
under the log-distance law — anchors learned parameters to physics rather than
only to LS-fitted pseudo-labels.
"""
from __future__ import annotations

import torch

from terranet.models.uq.heteroscedastic import gaussian_nll


def physics_residual(mean: torch.Tensor, d_m: torch.Tensor, pl_db: torch.Tensor,
                     mask: torch.Tensor, d0: float = 1.0) -> torch.Tensor:
    """mean: (B,2) [gamma, pl0]; d_m/pl_db: (B, M) per-tile measurement samples;
    mask: (B, M) valid entries."""
    gamma, pl0 = mean[:, 0:1], mean[:, 1:2]
    pred = pl0 + 10.0 * gamma * torch.log10(torch.clamp(d_m, min=d0) / d0)
    se = ((pred - pl_db) ** 2) * mask
    return se.sum() / torch.clamp(mask.sum(), min=1.0)


def total_loss(out: dict, target: torch.Tensor, *, phys: torch.Tensor | None = None,
               da: torch.Tensor | None = None, target_scale: torch.Tensor,
               w_phys: float = 0.1, w_da: float = 0.1) -> dict[str, torch.Tensor]:
    nll = gaussian_nll(out["mean"], out["logvar"], target, scale=target_scale)
    loss = nll
    logs = {"nll": nll}
    if phys is not None:
        loss = loss + w_phys * phys
        logs["phys"] = phys
    if da is not None:
        loss = loss + w_da * da
        logs["da"] = da
    logs["loss"] = loss
    return logs
