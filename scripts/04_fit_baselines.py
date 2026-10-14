#!/usr/bin/env python
"""Zero-shot LOCO benchmark of transfer operators (nested selection).

Operators (see terranet.experiments.common for the exact protocol):
  median        source-median constant (the no-information null)
  uma_los/nlos  3GPP TR 38.901 UMa single-slope constants
  cost231       COST-231 Hata constants (outside its validity range; site
                height = median of the source sites)
  kernel        Gaussian Nadaraya-Watson (the operator of the base paper)
  gp_ard        Gaussian process with one length scale per descriptor
  knn           k-nearest-neighbour average
  ridge         linear ridge
  coral         ridge on CORAL-aligned descriptors (unlabelled target covariance)
  mlp           two-layer MLP, epochs and penalty from the inner folds (5 seeds)
  gbdt          LightGBM, rounds from the inner folds (5 seeds)
  gbdt_iw       LightGBM with importance weights from a domain classifier
                (unlabelled target descriptors; the rounds of gbdt; 5 seeds)

Every hyper-parameter is chosen by inner leave-one-training-city-out
selection (seed 0) and the operator is refitted on all training cities.

Outputs
  outputs/tables/loco_metrics.csv      one row per (fold, operator, seed)
  outputs/tables/loco_hparams.csv      selected hyper-parameters per fold
  outputs/tables/inner_bias.csv        zero-shot bias and RMSE of every operator
                                       on every inner validation city (used by
                                       the empirical-Bayes few-shot correction)
  outputs/predictions/loco.parquet     per-tile predictions (for 09_network_eval)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.evaluation.metrics import full_report
from terranet.experiments.common import (
    GBDT_PARAMS,
    KNN_GRID,
    RIDGE_GRID,
    SEEDS,
    SIGMA_GRID,
    CityData,
    MaxAbs,
    coral,
    cost231_hata,
    fit_gbdt,
    fit_gp_ard,
    fit_mlp,
    importance_weights,
    inner_select,
    inner_splits,
    knn_predict,
    load_cfg,
    load_splits,
    nw_predict,
    ridge_predict,
    select_gbdt_trees,
    select_mlp,
    tr38901_uma,
)
from terranet.experiments.parallel import map_folds, n_jobs
from terranet.utils.logging import get_logger
from terranet.utils.seed import seed_everything

log = get_logger("loco")
CTX = {}            # corpus and configuration, shared with forked workers


def run_fold(sp):
    """All operators on one held-out city. Returns (rows, hp, preds, inner)."""
    cfg, base, data = CTX["cfg"], CTX["base"], CTX["data"]
    rows, hp, preds, inner = [], [], [], []
    fold, train = sp["test_cities"][0], sp["train_cities"]
    Xtr, Ytr = data.pool(train)
    Xte, Yte = data.d[fold]
    norm = MaxAbs().fit(Xtr)
    Xtr, Xte = norm(Xtr), norm(Xte)
    te = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / fold / "tiles.parquet")
    tile_row = np.flatnonzero(np.isfinite(te.gamma.to_numpy()))

    def record(op, pred, seed=0):
        r = {"fold": fold, "operator": op, "seed": seed}
        r.update(full_report(pred[:, 0], Yte[:, 0], "gamma_"))
        r.update(full_report(pred[:, 1], Yte[:, 1], "pl0_"))
        rows.append(r)
        preds.append(pd.DataFrame({"fold": fold, "operator": op, "seed": seed,
                                   "tile_row": tile_row,
                                   "gamma_hat": pred[:, 0], "pl0_hat": pred[:, 1]}))

    def inner_record(op, fit_predict):
        """Zero-shot error of `op` on every inner validation city."""
        for tr, va in inner_splits(train):
            Xa, Ya, Xb, Yb = data.fold(tr, va)
            e = fit_predict(Xa, Ya, Xb) - Yb
            inner.append(dict(fold=fold, operator=op, inner_city=va,
                              bias_gamma=float(e[:, 0].mean()), bias_pl0=float(e[:, 1].mean()),
                              rmse_gamma=float(np.sqrt((e[:, 0] ** 2).mean())),
                              rmse_pl0=float(np.sqrt((e[:, 1] ** 2).mean()))))

    record("median", np.tile(np.median(Ytr, 0), (len(Yte), 1)))
    inner_record("median", lambda Xa, Ya, Xb: np.tile(np.median(Ya, 0), (len(Xb), 1)))
    for los in (True, False):
        record("uma_los" if los else "uma_nlos",
               np.tile(tr38901_uma(cfg.freq_ghz, cfg.d0_m, los), (len(Yte), 1)))
    h_bs = float(np.median(np.concatenate([
        pd.read_parquet(Path(base.paths.raw) / cfg.dataset / c / "measurements.parquet",
                        columns=["bs_id", "tx_h"]).groupby("bs_id").tx_h.first()
        .to_numpy() for c in train])))
    record("cost231", np.tile(cost231_hata(cfg.freq_ghz, cfg.d0_m, h_bs), (len(Yte), 1)))

    sigma, _ = inner_select(data, train, SIGMA_GRID,
                            lambda s, Xa, Ya, Xb: nw_predict(Xb, Xa, Ya, s))
    record("kernel", nw_predict(Xte, Xtr, Ytr, sigma))
    inner_record("kernel", lambda Xa, Ya, Xb: nw_predict(Xb, Xa, Ya, sigma))

    k_best, _ = inner_select(data, train, KNN_GRID,
                             lambda k, Xa, Ya, Xb: knn_predict(Xb, Xa, Ya, k))
    record("knn", knn_predict(Xte, Xtr, Ytr, k_best))

    alpha, _ = inner_select(data, train, RIDGE_GRID,
                            lambda a, Xa, Ya, Xb: ridge_predict(Xb, Xa, Ya, a))
    record("ridge", ridge_predict(Xte, Xtr, Ytr, alpha))
    inner_record("ridge", lambda Xa, Ya, Xb: ridge_predict(Xb, Xa, Ya, alpha))

    # CORAL: the source is aligned to the (unlabelled) descriptors of the
    # city being predicted, inner validation city or target alike
    alpha_c, _ = inner_select(data, train, RIDGE_GRID,
                              lambda a, Xa, Ya, Xb: ridge_predict(Xb, coral(Xa, Xb), Ya, a))
    record("coral", ridge_predict(Xte, coral(Xtr, Xte), Ytr, alpha_c))

    gp = fit_gp_ard(Xtr, Ytr, seed=0)
    record("gp_ard", gp.predict(Xte))
    w_iw = importance_weights(Xtr, Xte)

    trees = select_gbdt_trees(data, train, seed=0)
    mlp_alpha, mlp_epochs, _ = select_mlp(data, train, seed=0)
    for seed in SEEDS:
        seed_everything(seed)
        record("mlp", fit_mlp(Xtr, Ytr, mlp_alpha, mlp_epochs, seed).predict(Xte), seed)
        record("gbdt", fit_gbdt(Xtr, Ytr, trees, seed).predict(Xte), seed)
        record("gbdt_iw", fit_gbdt(Xtr, Ytr, trees, seed, sample_weight=w_iw).predict(Xte),
               seed)
    # inner zero-shot errors at the selected hyper-parameters (seed 0)
    inner_record("gbdt", lambda Xa, Ya, Xb: fit_gbdt(Xa, Ya, trees, 0).predict(Xb))
    inner_record("mlp", lambda Xa, Ya, Xb: fit_mlp(Xa, Ya, mlp_alpha, mlp_epochs, 0)
                 .predict(Xb))

    hp.append(dict(fold=fold, n_train=len(Xtr), sigma=sigma, knn_k=k_best,
                   ridge_alpha=alpha, coral_alpha=alpha_c, h_bs_m=h_bs,
                   iw_ess=float(w_iw.sum() ** 2 / (w_iw ** 2).sum()),
                   gp_ls_gamma_min=float(np.min(gp.length_scales[0])),
                   mlp_alpha=float(mlp_alpha), mlp_epochs=int(mlp_epochs),
                   gbdt_trees_gamma=int(trees[0]), gbdt_trees_pl0=int(trees[1])))
    sub = pd.DataFrame(rows)
    log.info(f"{fold:10s} " + "  ".join(
        f"{o}={v:.3f}" for o, v in sub.groupby("operator").gamma_rmse.mean().items()))
    return rows, hp, preds, inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--jobs", type=int, default=n_jobs(), help="folds in parallel")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    out_t = Path(base.paths.outputs) / "tables"
    out_p = Path(base.paths.outputs) / "predictions"
    out_t.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)
    if args.jobs > 1:
        GBDT_PARAMS["num_threads"] = 1
    CTX.update(cfg=cfg, base=base, data=CityData(base, cfg, list(cfg.cities)))

    rows, hp, preds, inner = [], [], [], []
    for r, h, p, i in map_folds(run_fold, load_splits(), args.jobs):
        rows += r
        hp += h
        preds += p
        inner += i
    pd.DataFrame(rows).to_csv(out_t / "loco_metrics.csv", index=False)
    pd.DataFrame(hp).to_csv(out_t / "loco_hparams.csv", index=False)
    pd.DataFrame(inner).to_csv(out_t / "inner_bias.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(out_p / "loco.parquet")
    log.info(f"wrote {out_t/'loco_metrics.csv'}")


if __name__ == "__main__":
    main()
