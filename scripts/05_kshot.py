#!/usr/bin/env python
"""Few-shot cross-city adaptation curve.

For each LOCO fold, train on source cities, then reveal k labeled target-city
tiles and adapt. k=0 is zero-shot. Reports gamma/PL0 RMSE vs k, averaged over
several random draws of the k anchors (few-shot variance matters).

Adaptation = ridge fine-tune: fit a small correction on the k target tiles on
top of the source-trained ridge prediction (residual ridge). Simple, strong,
and honest about how little target data is needed.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge

from terranet.evaluation.metrics import rmse
from terranet.utils.logging import get_logger
from terranet.utils.seed import seed_everything

log = get_logger("kshot")
DCOLS = [f"descriptor_{i}" for i in range(116)]
KS = [0, 3, 5, 10, 25, 50, 100]
N_DRAWS = 10


def load_city(proc, ds, city):
    d = pd.read_parquet(Path(proc) / ds / city / "tiles.parquet")
    d = d[np.isfinite(d.gamma)].reset_index(drop=True)
    return d[DCOLS].to_numpy(np.float32), d[["gamma", "pl0"]].to_numpy(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config); base = OmegaConf.load("configs/base.yaml")
    splits = json.loads(Path("data/splits/loco.json").read_text())
    proc, ds = base.paths.processed, cfg.dataset

    rows = []
    for sp in splits:
        seed_everything(args.seed)
        fold = sp["test_cities"][0]
        Xs = np.concatenate([load_city(proc, ds, c)[0] for c in sp["train_cities"]])
        Ys = np.concatenate([load_city(proc, ds, c)[1] for c in sp["train_cities"]])
        mx = np.abs(Xs).max(0); mx[mx == 0] = 1.0
        Xs = Xs / mx
        Xt_raw, Yt = load_city(proc, ds, fold)
        Xt = Xt_raw / mx

        # source model
        src = Ridge(alpha=1.0).fit(Xs, Ys)
        base_pred = src.predict(Xt)

        rng = np.random.default_rng(args.seed)
        for k in KS:
            if k == 0:
                g = rmse(base_pred[:, 0], Yt[:, 0]); p = rmse(base_pred[:, 1], Yt[:, 1])
                rows.append(dict(fold=fold, k=0, draw=0, gamma_rmse=g, pl0_rmse=p))
                continue
            for draw in range(N_DRAWS):
                idx = rng.choice(len(Xt), size=min(k, len(Xt)), replace=False)
                mask = np.ones(len(Xt), bool); mask[idx] = False   # eval on unseen tiles
                # residual ridge: correct source prediction using k target anchors
                resid = Yt[idx] - src.predict(Xt[idx])
                corr = Ridge(alpha=10.0).fit(Xt[idx], resid)
                pred = src.predict(Xt[mask]) + corr.predict(Xt[mask])
                rows.append(dict(fold=fold, k=k, draw=draw,
                                 gamma_rmse=rmse(pred[:, 0], Yt[mask, 0]),
                                 pl0_rmse=rmse(pred[:, 1], Yt[mask, 1])))
        log.info(f"{fold} done")

    df = pd.DataFrame(rows)
    out = Path(base.paths.outputs) / "tables" / "kshot_loco.csv"
    df.to_csv(out, index=False)

    print("\n=== gamma_rmse vs k (mean over folds x draws) ===")
    piv = df.groupby("k").gamma_rmse.agg(["mean", "std"]).round(3)
    print(piv)
    print("\n=== per-city gamma_rmse: k=0 vs k=10 vs k=50 ===")
    sub = df[df.k.isin([0, 10, 50])].groupby(["fold", "k"]).gamma_rmse.mean().unstack().round(3)
    print(sub)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
