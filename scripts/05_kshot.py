#!/usr/bin/env python
"""Few-shot correction: reveal k labelled target tiles, correct the zero-shot model.

Zero-shot predictions are read from outputs/predictions/loco.parquet (written by
04_fit_baselines.py), so k = 0 reproduces the zero-shot table exactly.

  operators : kernel, ridge, gbdt (every seed), mlp (every seed)
  modes     : bias      - add the plain mean anchor residual
              eb        - precision-weighted mean residual, shrunk by empirical
                          Bayes: anchor i has weight 1 / (se_i^2 + s^2), with
                          se_i the cluster-robust standard error of its label
                          and s^2 the within-city error variance of the
                          operator on the inner validation cities; the weighted
                          mean c (sampling variance V) is shrunk to
                          tau^2 / (tau^2 + V) c with tau^2 the between-city
                          variance of the operator's zero-shot bias on the
                          inner cities (outputs/tables/inner_bias.csv)
              residual  - precision-weighted ridge (alpha = 10) on normalised
                          descriptors -> residual
  sampling  : random    - k tiles uniformly at random
              patch     - the k tiles nearest a random seed tile (a contiguous
                          survey area, e.g. one drive-test route)
              design    - greedy facility location over the target descriptors
                          (deterministic, chosen before any measurement)
Every (fold, k, draw, sampling) uses the same anchors for all operators, so
comparisons are paired. Evaluation is always on the unrevealed tiles.

Output: outputs/tables/kshot.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from terranet.experiments.common import (
    MaxAbs,
    facility_location,
    load_cfg,
    load_splits,
    load_tiles,
    stack,
    xy,
)
from terranet.experiments.fewshot import eb_correction, eb_prior, label_se
from terranet.experiments.parallel import map_folds, n_jobs
from terranet.utils.logging import get_logger

log = get_logger("kshot")
KS = [0, 3, 5, 10, 25, 50, 100]
N_DRAWS = 10
OPERATORS = ["kernel", "ridge", "mlp", "gbdt"]
RESIDUAL_ALPHA = 10.0


def anchors(rng, k, sampling, xy_m, X=None):
    n = len(xy_m)
    if sampling == "design":
        return facility_location(X, k)
    if sampling == "random":
        return rng.choice(n, size=k, replace=False)
    c = xy_m[rng.integers(n)]
    return np.argsort(((xy_m - c) ** 2).sum(1), kind="stable")[:k]


CTX = {}            # predictions and configuration, shared with forked workers


def run_fold(item):
    cfg, base, P, ib = CTX["cfg"], CTX["base"], CTX["P"], CTX["ib"]
    fi, sp = item
    rows = []
    fold = sp["test_cities"][0]
    te = load_tiles(base, cfg, fold)
    Xte, Yte = xy(te)
    Xte = MaxAbs().fit(xy(stack(base, cfg, sp["train_cities"]))[0])(Xte)
    se = label_se(te)
    lat0 = np.radians(te.lat_c.mean())
    xy_m = np.c_[te.lon_c * 111_320 * np.cos(lat0), te.lat_c * 111_320]
    pf = P[P.fold == fold]
    base_preds = {(op, s): g.set_index("tile_row").loc[te.tile_row, ["gamma_hat", "pl0_hat"]]
                  .to_numpy() for (op, s), g in pf.groupby(["operator", "seed"])
                  if op in OPERATORS}
    # empirical-Bayes constants of every operator from the inner cities
    prior = {op: eb_prior(ib, fold, op) for op in OPERATORS}

    for sampling in ("random", "patch", "design"):
        for k in KS:
            if k > len(te) // 2:          # keep most tiles for evaluation
                continue
            n_draws = N_DRAWS if k and sampling != "design" else 1
            for draw in range(n_draws):
                rng = np.random.default_rng([fi, k, draw, sampling == "patch"])
                idx = anchors(rng, k, sampling, xy_m, Xte) if k else np.array([], int)
                mask = np.ones(len(te), bool)
                mask[idx] = False
                for (op, seed), bp in base_preds.items():
                    resid = Yte[idx] - bp[idx]
                    s2, tau2 = prior[op]
                    for mode in ("bias", "eb", "residual"):
                        if k == 0:
                            pred = bp[mask]
                        elif mode == "bias":
                            pred = bp[mask] + resid.mean(0)
                        elif mode == "eb":
                            c, _ = eb_correction(resid, se[idx], s2, tau2)
                            pred = bp[mask] + c
                        else:
                            _, w = eb_correction(resid, se[idx], s2, tau2)
                            pred = bp[mask].copy()
                            for j in range(2):
                                sw = w[:, j] / w[:, j].mean()
                                corr = Ridge(alpha=RESIDUAL_ALPHA).fit(
                                    Xte[idx], resid[:, j], sample_weight=sw)
                                pred[:, j] += corr.predict(Xte[mask])
                        e = pred - Yte[mask]
                        rows.append(dict(
                            fold=fold, operator=op, seed=seed, sampling=sampling,
                            mode=mode, k=k, draw=draw,
                            gamma_rmse=float(np.sqrt((e[:, 0] ** 2).mean())),
                            pl0_rmse=float(np.sqrt((e[:, 1] ** 2).mean())),
                            gamma_mse=float((e[:, 0] ** 2).mean())))
    log.info(f"{fold} done")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--jobs", type=int, default=n_jobs(), help="folds in parallel")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    CTX.update(cfg=cfg, base=base,
               P=pd.read_parquet(Path(base.paths.outputs) / "predictions" / "loco.parquet"),
               ib=pd.read_csv(Path(base.paths.outputs) / "tables" / "inner_bias.csv"))
    rows = []
    for r in map_folds(run_fold, enumerate(load_splits()), args.jobs):
        rows += r

    df = pd.DataFrame(rows)
    out = Path(base.paths.outputs) / "tables" / "kshot.csv"
    df.to_csv(out, index=False)
    s = df[(df["mode"] == "eb") & (df.sampling == "random")]
    print(s.groupby(["operator", "k"]).gamma_rmse.mean().unstack().round(3))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
