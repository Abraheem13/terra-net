#!/usr/bin/env python
"""Few-shot correction: reveal k labelled target tiles, correct the zero-shot model.

Zero-shot predictions are read from outputs/predictions/loco.parquet (written by
04_fit_baselines.py), so k = 0 reproduces the zero-shot table exactly.

  operators : kernel, ridge, gbdt (every seed), mlp (every seed)
  modes     : bias      - add the mean anchor residual
              residual  - ridge (alpha = 10) on normalised descriptors -> residual
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    P = pd.read_parquet(Path(base.paths.outputs) / "predictions" / "loco.parquet")

    rows = []
    for fi, sp in enumerate(load_splits()):
        fold = sp["test_cities"][0]
        te = load_tiles(base, cfg, fold)
        Xte, Yte = xy(te)
        Xte = MaxAbs().fit(xy(stack(base, cfg, sp["train_cities"]))[0])(Xte)
        lat0 = np.radians(te.lat_c.mean())
        xy_m = np.c_[te.lon_c * 111_320 * np.cos(lat0), te.lat_c * 111_320]
        pf = P[P.fold == fold]
        base_preds = {(op, s): g.set_index("tile_row").loc[te.tile_row, ["gamma_hat", "pl0_hat"]]
                      .to_numpy() for (op, s), g in pf.groupby(["operator", "seed"])
                      if op in OPERATORS}

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
                        for mode in ("bias", "residual"):
                            if k == 0:
                                pred = bp[mask]
                            elif mode == "bias":
                                pred = bp[mask] + resid.mean(0)
                            else:
                                corr = Ridge(alpha=RESIDUAL_ALPHA).fit(Xte[idx], resid)
                                pred = bp[mask] + corr.predict(Xte[mask])
                            e = pred - Yte[mask]
                            rows.append(dict(
                                fold=fold, operator=op, seed=seed, sampling=sampling,
                                mode=mode, k=k, draw=draw,
                                gamma_rmse=float(np.sqrt((e[:, 0] ** 2).mean())),
                                pl0_rmse=float(np.sqrt((e[:, 1] ** 2).mean())),
                                gamma_mse=float((e[:, 0] ** 2).mean())))
        log.info(f"{fold} done")

    df = pd.DataFrame(rows)
    out = Path(base.paths.outputs) / "tables" / "kshot.csv"
    df.to_csv(out, index=False)
    s = df[(df["mode"] == "residual") & (df.sampling == "random")]
    print(s.groupby(["operator", "k"]).gamma_rmse.mean().unstack().round(3))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
