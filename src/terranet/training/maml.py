"""First-order MAML / Reptile-style k-shot adaptation to a target city.

Given k labeled target tiles (k in {0, 5, 10, 50, 100}), fine-tune a copy of the
trained model for a few inner steps; used to produce sample-efficiency curves.
"""
from __future__ import annotations

import copy

import torch


def kshot_adapt(model, support_batch, inner_steps: int = 20, inner_lr: float = 1e-3,
                head_only: bool = True):
    adapted = copy.deepcopy(model)
    params = (adapted.head.parameters() if head_only else adapted.parameters())
    opt = torch.optim.SGD(params, lr=inner_lr)
    desc, ras, tgt, _ = support_batch
    for _ in range(inner_steps):
        opt.zero_grad()
        out = adapted(desc, ras)
        loss = ((out["mean"] - tgt) / adapted.target_scale).pow(2).mean()
        loss.backward()
        opt.step()
    return adapted
