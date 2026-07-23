#!/usr/bin/env python
"""Neural (gamma, PL0) regressor under LOCO — encoder + hypernet, heteroscedastic NLL.

Plain PyTorch (no Lightning: simpler to debug on a small box). Train-only
descriptor normalization and train-only TARGET standardization (so the NLL
weights gamma and PL0 comparably). Early stop on the val city. Writes per-fold
metrics in the SAME schema as 04_fit_baselines.py for direct comparison.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from terranet.models.encoders.descriptor_mlp import DescriptorEncoder
from terranet.models.hypernet import ParamHead
from terranet.evaluation.metrics import full_report
from terranet.utils.logging import get_logger
from terranet.utils.seed import seed_everything

log = get_logger("neural")
DCOLS = [f"descriptor_{i}" for i in range(116)]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_city(proc, ds, city):
    d = pd.read_parquet(Path(proc) / ds / city / "tiles.parquet")
    d = d[np.isfinite(d.gamma)].reset_index(drop=True)
    return d[DCOLS].to_numpy(np.float32), d[["gamma", "pl0"]].to_numpy(np.float32)


def stack(proc, ds, cities):
    X = [load_city(proc, ds, c)[0] for c in cities]
    Y = [load_city(proc, ds, c)[1] for c in cities]
    return np.concatenate(X), np.concatenate(Y)


def hetero_nll(mean, logvar, target):
    inv = torch.exp(-logvar)
    return (0.5 * (inv * (target - mean) ** 2 + logvar)).mean()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = DescriptorEncoder()
        self.head = ParamHead(emb_dim=self.enc.emb_dim)

    def forward(self, x, ymean, ystd):
        out = self.head(self.enc(x))
        # head outputs physical (gamma, PL0); standardize for the NLL
        mean = (out["mean"] - ymean) / ystd
        return mean, out["logvar"], out["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config); base = OmegaConf.load("configs/base.yaml")
    splits = json.loads(Path("data/splits/loco.json").read_text())
    proc, ds = base.paths.processed, cfg.dataset

    rows = []
    for sp in splits:
        seed_everything(args.seed)
        fold = sp["test_cities"][0]
        Xtr, Ytr = stack(proc, ds, sp["train_cities"])
        Xva, Yva = stack(proc, ds, sp["val_cities"])
        Xte, Yte = load_city(proc, ds, fold)

        mx = np.abs(Xtr).max(0); mx[mx == 0] = 1.0
        Xtr, Xva, Xte = Xtr / mx, Xva / mx, Xte / mx
        ymean, ystd = Ytr.mean(0), Ytr.std(0)

        t = lambda a: torch.tensor(a, device=DEV)
        Xtr_t, Ytr_t = t(Xtr), t(Ytr)
        Xva_t = t(Xva); Xte_t = t(Xte)
        ym, ys = t(ymean), t(ystd)
        Ytr_std = (Ytr_t - ym) / ys

        net = Net().to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

        best_val, best_state, patience, bad = 1e9, None, 40, 0
        bs = 512
        n = len(Xtr_t)
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                mean, logvar, _ = net(Xtr_t[idx], ym, ys)
                loss = hetero_nll(mean, logvar, Ytr_std[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                _, _, phys = net(Xva_t, ym, ys)
                vloss = float(((phys.cpu().numpy() - Yva) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, best_state, bad = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            _, _, phys = net(Xte_t, ym, ys)
            pred = phys.cpu().numpy()
        r = {"model": "terranet_encoder", "fold": fold}
        r.update(full_report(pred[:, 0], Yte[:, 0], "gamma_"))
        r.update(full_report(pred[:, 1], Yte[:, 1], "pl0_"))
        rows.append(r)
        log.info(f"{fold:11s} gamma_rmse={r['gamma_rmse']:.3f} pl0_rmse={r['pl0_rmse']:.2f} "
                 f"(stopped ep {ep})")

    df = pd.DataFrame(rows)
    out = Path(base.paths.outputs) / "tables" / "neural_loco.csv"
    df.to_csv(out, index=False)
    print("\n=== neural encoder LOCO ===")
    print(df.groupby("model")[["gamma_rmse", "pl0_rmse"]].agg(["mean", "std"]).round(3))
    print(f"\nvs ridge baseline gamma_rmse=0.615\nwrote {out}")


if __name__ == "__main__":
    main()
