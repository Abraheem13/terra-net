#!/usr/bin/env python
"""Few-shot cross-city adaptation curve.

Source model trained on source cities, then k labeled target tiles are revealed
and used to correct it. Evaluation is always on the UNSEEN target tiles.

  --source ridge|gbdt        zero-shot model
  correction modes compared:
    bias     : single scalar offset per output (most robust at tiny k)
    residual : ridge on the k anchors, predicting source-model residuals
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


def fit_source(kind, X, Y, seed):
    if kind == "ridge":
        m = Ridge(alpha=1.0).fit(X, Y)
        return lambda Z: m.predict(Z)
    import lightgbm as lgb
    ms = [lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                            min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, verbose=-1).fit(X, Y[:, j]) for j in (0, 1)]
    return lambda Z: np.column_stack([m.predict(Z) for m in ms])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--source", choices=["ridge", "gbdt"], default="gbdt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config); base = OmegaConf.load("configs/base.yaml")
    splits = json.loads(Path("data/splits/loco.json").read_text())
    proc, ds = base.paths.processed, cfg.dataset

    rows = []
    for sp in splits:
        seed_everything(args.seed)
        fold = sp["test_cities"][0]
        src_cities = sp["train_cities"] + sp["val_cities"]
        Xs = np.concatenate([load_city(proc, ds, c)[0] for c in src_cities])
        Ys = np.concatenate([load_city(proc, ds, c)[1] for c in src_cities])
        mx = np.abs(Xs).max(0); mx[mx == 0] = 1.0
        Xs = Xs / mx
        Xt_raw, Yt = load_city(proc, ds, fold)
        Xt = Xt_raw / mx

        predict_src = fit_source(args.source, Xs, Ys, args.seed)
        base_pred = predict_src(Xt)

        for mode in ("bias", "residual"):
            rows.append(dict(fold=fold, mode=mode, k=0, draw=0,
                             gamma_rmse=rmse(base_pred[:, 0], Yt[:, 0]),
                             pl0_rmse=rmse(base_pred[:, 1], Yt[:, 1])))

        rng = np.random.default_rng(args.seed)
        for k in KS[1:]:
            for draw in range(N_DRAWS):
                idx = rng.choice(len(Xt), size=min(k, len(Xt)), replace=False)
                mask = np.ones(len(Xt), bool); mask[idx] = False
                resid = Yt[idx] - base_pred[idx]
                for mode in ("bias", "residual"):
                    if mode == "bias":
                        pred = base_pred[mask] + resid.mean(0)
                    else:
                        corr = Ridge(alpha=10.0).fit(Xt[idx], resid)
                        pred = base_pred[mask] + corr.predict(Xt[mask])
                    rows.append(dict(fold=fold, mode=mode, k=k, draw=draw,
                                     gamma_rmse=rmse(pred[:, 0], Yt[mask, 0]),
                                     pl0_rmse=rmse(pred[:, 1], Yt[mask, 1])))
        log.info(f"{fold} done")

    df = pd.DataFrame(rows)
    df["source"] = args.source
    out = Path(base.paths.outputs) / "tables" / f"kshot_{args.source}.csv"
    df.to_csv(out, index=False)

    print(f"\n=== gamma_rmse vs k  (source={args.source}) ===")
    print(df.pivot_table(index="k", columns="mode", values="gamma_rmse").round(3))
    print("\n=== per-city, residual mode: k=0 / 10 / 50 ===")
    sub = df[(df["mode"] == "residual") & df.k.isin([0, 10, 50])]
    print(sub.groupby(["fold", "k"]).gamma_rmse.mean().unstack().round(3))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
