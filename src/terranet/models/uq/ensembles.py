"""Deep-ensemble utilities: aggregate M independently seeded models."""
from __future__ import annotations

import torch


@torch.no_grad()
def ensemble_predict(models: list, desc, raster) -> dict[str, torch.Tensor]:
    means, vars_ = [], []
    for m in models:
        out = m(desc, raster)
        means.append(out["mean"])
        vars_.append(out["logvar"].exp())
    mu = torch.stack(means).mean(0)
    aleatoric = torch.stack(vars_).mean(0)
    epistemic = torch.stack(means).var(0, unbiased=False)
    return {"mean": mu, "var": aleatoric + epistemic,
            "aleatoric": aleatoric, "epistemic": epistemic}
