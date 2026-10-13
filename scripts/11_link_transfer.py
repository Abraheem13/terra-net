#!/usr/bin/env python
"""Site-aware link-level transfer, evaluated at the network level (LOCO).

Instead of one (gamma, PL0) pair per tile, a gradient-boosted hurdle model
predicts every (pixel, site) link from features available before any
measurement (the link geometry and obstruction features of
terranet.data.link_features and the 32 descriptors of the receiver's tile):
  classifier  P(link below the 150 dB cut-off), trained on all pairs
  regressor   path loss of the links below the cut-off
A link is predicted beyond the cut-off (zero received power) when the
classifier gives less than one half; otherwise its path loss is the
regression. "link_reg" is the regressor alone (every link below the cut-off),
the ablation of the hurdle. The path-loss error is always that of the
regressor over the ray-traced links; `pl_rmse_cens_db` scores every pair with
prediction and truth clipped at the cut-off.

Protocol: the number of boosting rounds of both models is the median best
iteration over N_INNER_EXPENSIVE inner validation cities (fixed draw, seed 0,
early stopping with patience 50); the final models are trained on all
training cities with that number of rounds, five seeds. 120,000 ray-traced
links (regressor) and 120,000 pairs (classifier) are sampled per city.

Few-shot (k surveyed tiles): the observed links in the surveyed tiles carry
their serving site, as minimisation-of-drive-tests (MDT) reports do, and give
a per-site offset o_s = (n_s rbar_s + kappa rbar) / (n_s + kappa), with kappa
estimated by empirical Bayes from the surveyed links ("link+k"), or fixed at
50 links ("link+k{k}f", ablation). Surveys: "random" (k tiles uniformly at
random; for k = 10 the same draws as 09_network_eval.py), "d" (greedy
variance-reduction design of Proposition 5) and "s" (stratified design: one
tile per descriptor stratum). The surveyed pixels keep their model prediction
and every few-shot metric is computed on the unsurveyed pixels only.

Every zero-shot model is also scored on the 10-site network of the same siting
rule (column `sites`), its per-pixel serving-link residuals are written for
13_certified_coverage.py, and its maps are saved for 19_tile_variants.py.

Outputs
  outputs/tables/network_link.csv    same columns as network_metrics.csv
  outputs/tables/planning_link.csv   same columns as planning.csv
  outputs/tables/link_importance.csv mean gain share of each feature
  outputs/tables/link_hparams.csv    rounds selected per fold
Features are cached in outputs/cache/ (regenerated when absent or stale);
models are saved to outputs/models/ for 15_siting.py, 16_band.py and
17_external_scenes.py; maps to outputs/predictions/hurdle/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.evaluation.network import unsurveyed_mask
from terranet.experiments.calibration import calibrate, design_survey, design_survey_stratified
from terranet.experiments.common import (
    LINK_KAPPA,
    N_INNER_EXPENSIVE,
    SEEDS,
    SITES_SMALL,
    MaxAbs,
    inner_splits,
    lgb_eval_kw,
    load_cfg,
    load_splits,
    load_tiles,
    stack,
    xy,
)
from terranet.experiments.linkdata import FEATURES, city_data, save_hurdle
from terranet.utils.logging import get_logger

log = get_logger("link")
N_PER_CITY = 120_000
KS, N_DRAWS = (10, 25, 50), 10
LINK_GBDT = dict(learning_rate=0.1, num_leaves=127, min_child_samples=200, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, verbose=-1)
MAX_TREES, PATIENCE = 1500, 50


def sample_pairs(city, X, n, rng):
    """Random (pixel, site) pairs with the label 'below the cut-off'."""
    y = np.isfinite(city.true_pl.reshape(-1))
    idx = rng.choice(y.size, size=min(n, y.size), replace=False)
    return X[idx], y[idx].astype(int)


def sample_links(city, X, n, rng):
    y = city.true_pl.reshape(-1)
    idx = np.flatnonzero(np.isfinite(y))
    idx = rng.choice(idx, size=min(n, idx.size), replace=False)
    return X[idx], y[idx]


def training_set(data, cities, sampler, rng):
    parts = [sampler(*data[c], N_PER_CITY, rng) for c in cities]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    import lightgbm as lgb
    cfg, base = load_cfg(args.config)
    out = Path(base.paths.outputs)
    data = {c: city_data(cfg, base, c) for c in cfg.cities}
    log.info("features ready: " + ", ".join(f"{c} {X.shape[0]:,}" for c, (_, X) in data.items()))

    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    rows, plans, imp, serving, hps = [], [], [], [], []
    for fi, sp in enumerate(load_splits()):
        fold, train = sp["test_cities"][0], sp["train_cities"]
        city, Xte = data[fold]
        small = city.subset(np.arange(SITES_SMALL)) if city.true_pl.shape[1] > SITES_SMALL else None
        P, S = city.true_pl.shape
        te = load_tiles(base, cfg, fold)
        tile_ids = te.tile_row.to_numpy()
        Xdesc = MaxAbs().fit(xy(stack(base, cfg, train))[0])(xy(te)[0])

        # inner selection of the number of rounds (seed 0)
        best_r, best_c = [], []
        for tr, va in inner_splits(train, N_INNER_EXPENSIVE, seed=0):
            rng = np.random.default_rng([fi, 0])
            Xa, ya = training_set(data, tr, sample_links, rng)
            Xv, yv = sample_links(*data[va], N_PER_CITY, rng)
            m = lgb.LGBMRegressor(n_estimators=MAX_TREES, random_state=0, **LINK_GBDT)
            m.fit(Xa, ya, callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
                  **lgb_eval_kw(Xv, yv))
            best_r.append(int(m.best_iteration_))
            Xa, ya = training_set(data, tr, sample_pairs, rng)
            Xv, yv = sample_pairs(*data[va], N_PER_CITY, rng)
            c = lgb.LGBMClassifier(n_estimators=MAX_TREES, random_state=0, **LINK_GBDT)
            c.fit(Xa, ya, callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
                  **lgb_eval_kw(Xv, yv))
            best_c.append(int(c.best_iteration_))
        n_reg, n_clf = max(1, int(np.median(best_r))), max(1, int(np.median(best_c)))
        hps.append(dict(fold=fold, trees_reg=n_reg, trees_clf=n_clf,
                        inner_reg=str(best_r), inner_clf=str(best_c)))

        for seed in SEEDS:
            rng = np.random.default_rng([fi, seed])
            Xtr, ytr = training_set(data, train, sample_links, rng)
            m = lgb.LGBMRegressor(n_estimators=n_reg, random_state=seed, **LINK_GBDT).fit(Xtr, ytr)
            m.booster_.save_model(str(models / f"link_{fold}_{seed}.txt"))
            Xc, yc = training_set(data, train, sample_pairs, rng)
            clf = lgb.LGBMClassifier(n_estimators=n_clf, random_state=seed, **LINK_GBDT).fit(Xc, yc)
            clf.booster_.save_model(str(models / f"linkclf_{fold}_{seed}.txt"))
            PLr = m.predict(Xte).reshape(P, S)
            keep = clf.predict_proba(Xte)[:, 1].reshape(P, S) >= 0.5
            PLh = np.where(keep, PLr, np.inf)
            save_hurdle(base, fold, seed, PLr, keep)
            gain = m.booster_.feature_importance("gain")
            imp.append(pd.Series(gain / gain.sum(), index=FEATURES).rename(f"{fold}_{seed}"))

            def run(op, draw, M, R, zero_shot=False, mask=None, kappa=np.nan):
                nets = [(city, M, R)] + ([(small, M[:, :SITES_SMALL], R[:, :SITES_SMALL])]
                                         if zero_shot and small is not None else [])
                for c, MM, RR in nets:
                    key = {"fold": fold, "operator": op, "seed": seed, "draw": draw,
                           "sites": MM.shape[1]}
                    rows.append({**key, "trees": n_reg, "trees_clf": n_clf, "kappa": kappa,
                                 **c.evaluate(MM, pl_for_error=RR, mask=mask)})
                    for q in c.plan(MM, mask=mask):
                        plans.append({**key, **q})
            run("link", 0, PLh, PLr, zero_shot=True)
            run("link_reg", 0, PLr, PLr, zero_shot=True)
            for op, M in (("link", PLh), ("link_reg", PLr)):
                ps, res, ts = city.serving_residual(M)
                serving.append(pd.DataFrame({"fold": fold, "operator": op, "seed": seed,
                                             "pred_serv_pl": ps.astype(np.float32),
                                             "resid": res.astype(np.float32),
                                             "true_serv_pl": ts.astype(np.float32)}))

            # few-shot site calibration from k surveyed tiles (MDT reports)
            for k in KS:
                for draw in range(N_DRAWS):
                    r2 = np.random.default_rng([fi, k, draw, False])
                    pick = tile_ids[r2.choice(len(te), size=k, replace=False)]
                    mask = unsurveyed_mask(city, pick)
                    M, R, kap = calibrate(city, PLr, keep, pick)
                    run(f"link+k{k}", draw, M, R, mask=mask, kappa=kap)
                    M, R, kap = calibrate(city, PLr, keep, pick, kappa=LINK_KAPPA)
                    run(f"link+k{k}f", draw, M, R, mask=mask, kappa=kap)
                for suffix, pick in (("d", tile_ids[design_survey(city.tile, tile_ids, keep, k)]),
                                     ("s", tile_ids[design_survey_stratified(
                                         city.tile, tile_ids, keep, k, Xdesc)])):
                    mask = unsurveyed_mask(city, pick)
                    M, R, kap = calibrate(city, PLr, keep, pick)
                    run(f"link+k{k}{suffix}", 0, M, R, mask=mask, kappa=kap)
            sub = [r for r in rows if r["fold"] == fold and r["seed"] == seed
                   and r["sites"] == S]
            log.info(f"{fold} seed {seed}: trees={n_reg}/{n_clf} " + " ".join(
                f"{o}={np.mean([r['pl_rmse_db'] for r in sub if r['operator'] == o]):.2f}"
                for o in ["link"] + [f"link+k{k}{d}" for k in KS for d in ("", "d", "s")]))

    T = out / "tables"
    pd.DataFrame(rows).to_csv(T / "network_link.csv", index=False)
    pd.DataFrame(plans).to_csv(T / "planning_link.csv", index=False)
    pd.DataFrame(hps).to_csv(T / "link_hparams.csv", index=False)
    pd.concat(imp, axis=1).mean(1).sort_values(ascending=False).rename("gain_share") \
        .to_csv(T / "link_importance.csv", index_label="feature")
    pd.concat(serving, ignore_index=True).to_parquet(
        out / "predictions" / "serving_link.parquet")
    log.info("wrote network_link.csv, planning_link.csv, link_importance.csv")


if __name__ == "__main__":
    main()
