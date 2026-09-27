#!/usr/bin/env python
"""Network-level consequences of transfer error on the held-out city.

Each transfer operator yields tile parameters (gamma_hat, PL0_hat); with the
known base-station sites these define a predicted path-loss map for every
(pixel, base station) pair:

    PL_hat(p, b) = PL0_hat[tile(p)] + 10 gamma_hat[tile(p)] log10(d_pb / d0).

The ray-traced map is the ground truth. Pairs absent from the ray-traced data
exceeded the coupling-loss cut-off (150 dB, i.e. received power 10 dB below
the noise floor) and are treated as zero received power. Link budget
(downlink, full buffer, all sites active):

    P_tx = 46 dBm, bandwidth 20 MHz, noise figure 7 dB -> N = -94.0 dBm.

Per pixel we compute the serving site (minimum path loss), SINR and the
3GPP TR 36.942 attenuated-Shannon spectral efficiency
(0.6 log2(1+SINR), SINR >= -10 dB, capped at 4.4 b/s/Hz).

Metrics
  pl_rmse_db       path-loss RMSE over all ray-traced links
  assoc_acc        fraction of pixels with the correct serving site. A tile
                   model applies one (gamma, PL0) to every site seen from a
                   pixel, so its serving site is always the nearest site: this
                   metric is the same for every operator and measures the
                   tile model itself.
  cov_acc_<L>      coverage agreement, covered <=> serving path loss <= L dB
  cov_err_<L>      predicted minus true covered fraction
  sinr_mae_db      mean |SINR_hat - SINR| per pixel
  sinr_ks          Kolmogorov-Smirnov distance between the SINR CDFs
  se_err           predicted minus true mean spectral efficiency (b/s/Hz)
Planning: K sites (K in PLAN_K) chosen greedily from the 20 to maximise
coverage at 140 dB on the PREDICTED map, then scored on the TRUE map, against
the same greedy procedure run on the true map (regret, in covered fraction).

Operators: oracle (the tile labels themselves: error of the log-distance
tile model alone), median, uma_nlos, kernel, knn, ridge, mlp, gbdt, encoder,
and few-shot corrected ridge/gbdt at k = 10 (residual mode; "+k10": random
anchors, the same 10 draws as 05_kshot.py; "+k10d": the facility-location
design of 05_kshot.py).

Every zero-shot operator is also scored on the 10-site network produced by the
same siting rule (the first 10 sites chosen; column `sites`).
Cell edges: `edge_disp_m` is the mean distance between true and predicted
serving-cell boundaries, `handover_err` the relative error of the boundary
length. Sites needed for 90 % coverage are in planning.csv (K = -1).

Outputs: outputs/tables/network_metrics.csv, outputs/tables/planning.csv,
         outputs/predictions/serving_tile.parquet (per-pixel serving-link
         residuals of the zero-shot operators, for 13_certified_coverage.py)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from terranet.evaluation.network import load_city
from terranet.experiments.common import (
    SITES_SMALL,
    MaxAbs,
    facility_location,
    load_cfg,
    load_splits,
    load_tiles,
    stack,
    xy,
)
from terranet.utils.logging import get_logger

log = get_logger("network")
K_FEWSHOT, N_DRAWS, RESIDUAL_ALPHA = 10, 10, 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    out_dir = Path(base.paths.outputs)
    pred = [out_dir / "predictions" / f for f in ("loco.parquet", "neural.parquet")]
    P = pd.concat([pd.read_parquet(f) for f in pred if f.exists()])

    rows, plans, serving = [], [], []
    for fi, sp in enumerate(load_splits()):
        fold = sp["test_cities"][0]
        tiles = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / fold / "tiles.parquet")
        city = load_city(cfg, base, fold)
        log.info(f"{fold}: {city.true_pl.shape[0]:,} pixels x {city.true_pl.shape[1]} sites, "
                 f"max distance mismatch {city.dist_rel_err:.2e}")

        # the reduced network exists only when the city has more sites than it
        small = city.subset(np.arange(SITES_SMALL)) if city.true_pl.shape[1] > SITES_SMALL else None

        def run(op, seed, draw, g_t, p_t, zero_shot=False):
            PLh = city.predicted_pl(g_t, p_t)
            nets = [(city, PLh)] + ([(small, PLh[:, :SITES_SMALL])]
                                    if zero_shot and small is not None else [])
            for c, M in nets:
                key = {"fold": fold, "operator": op, "seed": seed, "draw": draw,
                       "sites": M.shape[1]}
                rows.append({**key, **c.evaluate(M)})
                for q in c.plan(M):
                    plans.append({**key, **q})
            if zero_shot:
                ps, res, ts = city.serving_residual(PLh)
                serving.append(pd.DataFrame({"fold": fold, "operator": op, "seed": seed,
                                             "pred_serv_pl": ps.astype(np.float32),
                                             "resid": res.astype(np.float32),
                                         "true_serv_pl": ts.astype(np.float32)}))

        n = len(tiles)
        run("oracle", 0, 0, tiles.gamma.to_numpy(), tiles.pl0.to_numpy(), zero_shot=True)

        te = load_tiles(base, cfg, fold)
        Xte, Yte = xy(te)
        Xte = MaxAbs().fit(xy(stack(base, cfg, sp["train_cities"]))[0])(Xte)
        for (op, seed), g in P[P.fold == fold].groupby(["operator", "seed"]):
            g = g.set_index("tile_row").loc[te.tile_row]
            gt, pt = np.full(n, np.nan), np.full(n, np.nan)
            gt[te.tile_row], pt[te.tile_row] = g.gamma_hat, g.pl0_hat
            run(op, seed, 0, gt.copy(), pt.copy(), zero_shot=True)
            if op in ("ridge", "gbdt"):
                bp = g[["gamma_hat", "pl0_hat"]].to_numpy()
                designs = [(f"{op}+k10", d, np.random.default_rng([fi, K_FEWSHOT, d, False])
                            .choice(len(te), size=K_FEWSHOT, replace=False))
                           for d in range(N_DRAWS)]
                designs.append((f"{op}+k10d", 0, facility_location(Xte, K_FEWSHOT)))
                for name, draw, idx in designs:
                    corr = Ridge(alpha=RESIDUAL_ALPHA).fit(Xte[idx], Yte[idx] - bp[idx])
                    fp = bp + corr.predict(Xte)
                    fp[idx] = Yte[idx]                        # surveyed tiles are known
                    gt[te.tile_row], pt[te.tile_row] = fp[:, 0], fp[:, 1]
                    run(name, seed, draw, gt.copy(), pt.copy())
        sub = pd.DataFrame([r for r in rows if r["fold"] == fold
                            and r["sites"] == city.true_pl.shape[1]])
        log.info("  " + "  ".join(f"{o}={v:.2f}" for o, v in
                                  sub.groupby("operator").pl_rmse_db.mean().items()))

    pd.DataFrame(rows).to_csv(out_dir / "tables" / "network_metrics.csv", index=False)
    pd.DataFrame(plans).to_csv(out_dir / "tables" / "planning.csv", index=False)
    pd.concat(serving, ignore_index=True).to_parquet(
        out_dir / "predictions" / "serving_tile.parquet")
    log.info("wrote network_metrics.csv and planning.csv")


if __name__ == "__main__":
    main()
