#!/usr/bin/env python
"""Zero-shot LOCO benchmark of transfer operators.

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
  mlp           two-layer MLP (5 seeds)
  gbdt          LightGBM (5 seeds)
  gbdt_iw       LightGBM with importance weights from a domain classifier
                (unlabelled target descriptors; 5 seeds)

Outputs
  outputs/tables/loco_metrics.csv      one row per (fold, operator, seed)
  outputs/tables/loco_hparams.csv      selected hyper-parameters per fold
  outputs/predictions/loco.parquet     per-tile predictions (for 09_network_eval)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from terranet.evaluation.metrics import full_report
from terranet.experiments.common import (
    KNN_GRID,
    SEEDS,
    MaxAbs,
    coral,
    cost231_hata,
    fit_gbdt,
    fit_gp_ard,
    fit_mlp,
    fit_ridge,
    importance_weights,
    knn_predict,
    load_cfg,
    load_splits,
    load_tiles,
    nw_predict,
    select_sigma,
    selection_loss,
    stack,
    tr38901_uma,
    xy,
)
from terranet.utils.logging import get_logger
from terranet.utils.seed import seed_everything

log = get_logger("loco")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    out_t = Path(base.paths.outputs) / "tables"
    out_p = Path(base.paths.outputs) / "predictions"
    out_t.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)

    rows, hp, preds = [], [], []
    for sp in load_splits():
        fold = sp["test_cities"][0]
        tr, va = stack(base, cfg, sp["train_cities"]), stack(base, cfg, sp["val_cities"])
        te = load_tiles(base, cfg, fold)
        Xtr, Ytr = xy(tr)
        Xva, Yva = xy(va)
        Xte, Yte = xy(te)
        norm = MaxAbs().fit(Xtr)
        Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)

        def record(op, pred, seed=0):
            r = {"fold": fold, "operator": op, "seed": seed}
            r.update(full_report(pred[:, 0], Yte[:, 0], "gamma_"))
            r.update(full_report(pred[:, 1], Yte[:, 1], "pl0_"))
            rows.append(r)
            preds.append(pd.DataFrame({"fold": fold, "operator": op, "seed": seed,
                                       "tile_row": te.tile_row.to_numpy(),
                                       "gamma_hat": pred[:, 0], "pl0_hat": pred[:, 1]}))

        record("median", np.tile(np.median(Ytr, 0), (len(Yte), 1)))
        for los in (True, False):
            record("uma_los" if los else "uma_nlos",
                   np.tile(tr38901_uma(cfg.freq_ghz, cfg.d0_m, los), (len(Yte), 1)))
        h_bs = float(np.median(np.concatenate([
            pd.read_parquet(Path(base.paths.raw) / cfg.dataset / c / "measurements.parquet",
                            columns=["bs_id", "tx_h"]).groupby("bs_id").tx_h.first()
            .to_numpy() for c in sp["train_cities"]])))
        record("cost231", np.tile(cost231_hata(cfg.freq_ghz, cfg.d0_m, h_bs), (len(Yte), 1)))

        sigma, _ = select_sigma(Xtr, Ytr, Xva, Yva)
        record("kernel", nw_predict(Xte, Xtr, Ytr, sigma))

        kerr = {k: selection_loss(knn_predict(Xva, Xtr, Ytr, k), Yva) for k in KNN_GRID}
        k_best = min(kerr, key=kerr.get)
        record("knn", knn_predict(Xte, Xtr, Ytr, k_best))

        rg, alpha = fit_ridge(Xtr, Ytr, Xva, Yva)
        record("ridge", rg.predict(Xte))

        # CORAL: alpha selected with the source aligned to the validation city,
        # final model with the source aligned to the (unlabelled) target city
        _, alpha_c = fit_ridge(coral(Xtr, Xva), Ytr, Xva, Yva)
        record("coral", Ridge(alpha=alpha_c).fit(coral(Xtr, Xte), Ytr).predict(Xte))

        gp = fit_gp_ard(Xtr, Ytr, seed=0)
        record("gp_ard", gp.predict(Xte))
        w_iw = importance_weights(Xtr, Xte)

        mlp_alpha, trees = [], []
        for seed in SEEDS:
            seed_everything(seed)
            m = fit_mlp(Xtr, Ytr, Xva, Yva, seed)
            record("mlp", m.predict(Xte), seed)
            mlp_alpha.append(m.alpha)
            g = fit_gbdt(Xtr, Ytr, Xva, Yva, seed)
            record("gbdt", g.predict(Xte), seed)
            trees.append(g.n_trees)
            record("gbdt_iw", fit_gbdt(Xtr, Ytr, Xva, Yva, seed, sample_weight=w_iw)
                   .predict(Xte), seed)

        hp.append(dict(fold=fold, val_city=sp["val_cities"][0], n_train=len(Xtr),
                       sigma=sigma, knn_k=k_best, ridge_alpha=alpha, coral_alpha=alpha_c,
                       h_bs_m=h_bs, iw_ess=float(w_iw.sum() ** 2 / (w_iw ** 2).sum()),
                       gp_ls_gamma_min=float(np.min(gp.length_scales[0])),
                       mlp_alpha=float(np.median(mlp_alpha)),
                       gbdt_trees_gamma=int(np.median([t[0] for t in trees])),
                       gbdt_trees_pl0=int(np.median([t[1] for t in trees]))))
        sub = pd.DataFrame([r for r in rows if r["fold"] == fold])
        log.info(f"{fold:10s} " + "  ".join(
            f"{o}={v:.3f}" for o, v in sub.groupby("operator").gamma_rmse.mean().items()))

    pd.DataFrame(rows).to_csv(out_t / "loco_metrics.csv", index=False)
    pd.DataFrame(hp).to_csv(out_t / "loco_hparams.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(out_p / "loco.parquet")
    log.info(f"wrote {out_t/'loco_metrics.csv'}")


if __name__ == "__main__":
    main()
