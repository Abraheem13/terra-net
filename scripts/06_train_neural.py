#!/usr/bin/env python
"""Neural transfer operator: descriptor encoder + (gamma, PL0) head, LOCO.

Same protocol as 04_fit_baselines.py: train-city normalisation, held-out city
for testing, five seeds. The number of epochs is chosen on N_INNER_EXPENSIVE
inner validation cities (fixed draw, seed 0; early stopping with patience 40),
and the final models are trained on all training cities for the median best
epoch. The head outputs physically constrained means and log-variances
trained with the heteroscedastic Gaussian NLL on source-standardised targets.

Outputs
  outputs/tables/neural_metrics.csv
  outputs/predictions/neural.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS
from terranet.evaluation.metrics import full_report
from terranet.experiments.common import (
    N_INNER_EXPENSIVE,
    SEEDS,
    CityData,
    MaxAbs,
    inner_splits,
    load_cfg,
    load_splits,
)
from terranet.models.encoders.descriptor_mlp import DescriptorEncoder
from terranet.models.hypernet import ParamHead
from terranet.models.uq.heteroscedastic import gaussian_nll
from terranet.utils.logging import get_logger
from terranet.utils.seed import seed_everything

log = get_logger("neural")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, BATCH, LR, WD = 300, 40, 512, 3e-4, 1e-2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = DescriptorEncoder(in_dim=len(DESCRIPTOR_COLUMNS))
        self.head = ParamHead(emb_dim=self.enc.emb_dim)

    def forward(self, x):
        return self.head(self.enc(x))


def train(Xtr, Ytr, seed, epochs, Xva=None, Yva=None, patience=PATIENCE):
    """Train for `epochs` epochs; with (Xva, Yva) early stopping on the
    validation city decides the epoch. Returns (net, epochs_run, best_epoch)."""
    seed_everything(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    Xtr_t, Ytr_t = t(Xtr), t(Ytr)
    ym, ys = Ytr_t.mean(0), Ytr_t.std(0)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, state, bad, best_ep = np.inf, None, 0, epochs
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr_t), device=DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            out = net(Xtr_t[b])
            loss = gaussian_nll(out["mean"] - ym, out["logvar"], Ytr_t[b] - ym, scale=ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if Xva is not None:
            net.eval()
            with torch.no_grad():
                v = net(t(Xva))["mean"].cpu().numpy()
            err = float(np.sqrt((((v - Yva) / Ytr.std(0)) ** 2).mean()))
            if err < best - 1e-5:
                best, bad, best_ep = err, 0, ep
                state = {k: x.detach().clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    return net, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    data = CityData(base, cfg, list(cfg.cities))
    rows, preds = [], []
    for sp in load_splits():
        fold, tr_cities = sp["test_cities"][0], sp["train_cities"]
        Xtr, Ytr = data.pool(tr_cities)
        Xte, Yte = data.d[fold]
        norm = MaxAbs().fit(Xtr)
        Xtr, Xte = norm(Xtr), norm(Xte)
        te = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / fold / "tiles.parquet")
        tile_row = np.flatnonzero(np.isfinite(te.gamma.to_numpy()))
        # inner selection of the number of epochs
        best = []
        for tr, va in inner_splits(tr_cities, N_INNER_EXPENSIVE, seed=0):
            Xa, Ya, Xb, Yb = data.fold(tr, va)
            best.append(train(Xa, Ya, 0, EPOCHS, Xb, Yb)[1])
        epochs = max(1, int(np.median(best)))
        for seed in SEEDS:
            net, _ = train(Xtr, Ytr, seed, epochs)
            with torch.no_grad():
                pred = net(torch.tensor(Xte, dtype=torch.float32, device=DEV))["mean"].cpu().numpy()
            r = {"fold": fold, "operator": "encoder", "seed": seed, "epochs": epochs}
            r.update(full_report(pred[:, 0], Yte[:, 0], "gamma_"))
            r.update(full_report(pred[:, 1], Yte[:, 1], "pl0_"))
            rows.append(r)
            preds.append(pd.DataFrame({"fold": fold, "operator": "encoder", "seed": seed,
                                       "tile_row": tile_row,
                                       "gamma_hat": pred[:, 0], "pl0_hat": pred[:, 1]}))
        log.info(f"{fold:10s} epochs={epochs} "
                 f"gamma_rmse={np.mean([r['gamma_rmse'] for r in rows[-5:]]):.3f}")
    out_t = Path(base.paths.outputs) / "tables"
    pd.DataFrame(rows).to_csv(out_t / "neural_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(
        Path(base.paths.outputs) / "predictions" / "neural.parquet")


if __name__ == "__main__":
    main()
