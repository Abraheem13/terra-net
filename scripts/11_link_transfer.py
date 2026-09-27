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
regressor over the ray-traced links.

Protocol (as for the tile operators): train on the training cities, early
stopping on the validation city, test on the held-out city, five seeds.
120,000 ray-traced links are sampled per training/validation city.

Few-shot (k surveyed tiles): the observed links in the surveyed tiles carry
their serving site, as minimisation-of-drive-tests (MDT) reports do, and give
a per-site offset
  o_s = (n_s rbar_s + kappa rbar) / (n_s + kappa),   kappa = 50 links,
i.e. the mean residual of site s shrunk towards the overall mean residual.
Surveys: "random" (k tiles uniformly at random; for k = 10 the same draws as
09_network_eval.py) and "design": the k tiles chosen greedily to maximise
  F(T) = sum_s nhat_s(T) / (nhat_s(T) + kappa),
the total variance reduction of the site offsets under the same shrinkage
model, where nhat_s(T) counts the links of site s in T that the zero-shot
hurdle classifier predicts below the cut-off (no measurement is used). F is monotone
submodular, so greedy attains at least (1 - 1/e) of the optimum.

Every zero-shot model is also scored on the 10-site network of the same siting
rule (column `sites`), and its per-pixel serving-link residuals are written for
13_certified_coverage.py.

Outputs
  outputs/tables/network_link.csv    same columns as network_metrics.csv
  outputs/tables/planning_link.csv   same columns as planning.csv
  outputs/tables/link_importance.csv mean gain share of each feature
Features are cached in outputs/cache/ (regenerated when absent or stale);
models are saved to outputs/models/ for 15_siting.py and 16_band.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS
from terranet.data.link_features import LINK_FEATURES, city_link_matrix
from terranet.evaluation.network import load_city
from terranet.experiments.common import (
    _LGB_EVAL_XY,
    LINK_KAPPA,
    SEEDS,
    SITES_SMALL,
    load_cfg,
    load_splits,
    load_tiles,
)
from terranet.utils.logging import get_logger

log = get_logger("link")
FEATURES = LINK_FEATURES + DESCRIPTOR_COLUMNS
N_PER_CITY = 120_000
KS, N_DRAWS, KAPPA = (10, 25, 50), 10, LINK_KAPPA
LINK_GBDT = dict(learning_rate=0.1, num_leaves=127, min_child_samples=200, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, verbose=-1)
MAX_TREES, PATIENCE = 1500, 50


def city_data(cfg, base, name, cache_dir):
    """City object plus the (P*S, F) feature matrix, cached."""
    city = load_city(cfg, base, name)
    f = cache_dir / f"link_{name}.npy"
    X = np.load(f) if f.exists() else None
    if X is None or X.shape[0] != city.true_pl.size:
        tiles = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / name / "tiles.parquet")
        X = city_link_matrix(city, Path(base.paths.raw) / cfg.scenes_dir / name,
                             tiles[DESCRIPTOR_COLUMNS].to_numpy(np.float32),
                             float(cfg.freq_ghz) * 1e9)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(f, X)
    return city, X


def design_survey(tile_of_pixel, tile_ids, pred_obs, k):
    """Greedy maximisation of F(T) = sum_s n_s(T) / (n_s(T) + KAPPA)."""
    S = pred_obs.shape[1]
    counts = np.zeros((len(tile_ids), S))
    pos = {t: i for i, t in enumerate(tile_ids)}
    rows = np.array([pos.get(t, -1) for t in tile_of_pixel])
    ok = rows >= 0
    np.add.at(counts, rows[ok], pred_obs[ok].astype(float))
    n = np.zeros(S)
    chosen = []
    for _ in range(k):
        f0 = (n / (n + KAPPA)).sum()
        gain = ((n + counts) / (n + counts + KAPPA)).sum(1) - f0
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        n += counts[j]
    return np.array(chosen)


def calibrate(city, PLr, keep, surveyed_tiles):
    """Per-site offsets from the links observed in the surveyed tiles.

    PLr: regression map; keep: hurdle decision (predicted below the cut-off).
    Returns (network map, calibrated regression map). In the surveyed tiles the
    links are known, including which sites are not received at all."""
    P, S = PLr.shape
    surveyed = np.isin(city.tile, surveyed_tiles)
    obs = np.isfinite(city.true_pl) & surveyed[:, None]
    res = (city.true_pl - PLr)[obs]
    site = np.broadcast_to(np.arange(S), (P, S))[obs]
    rbar = res.mean() if res.size else 0.0
    n_s = np.bincount(site, minlength=S)
    sum_s = np.bincount(site, weights=res, minlength=S)
    off = (sum_s + KAPPA * rbar) / (n_s + KAPPA)
    R = PLr + off[None, :]
    M = np.where(keep, R, np.inf)
    M[surveyed] = city.true_pl[surveyed]            # surveyed links are known
    return M, R


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    import lightgbm as lgb
    cfg, base = load_cfg(args.config)
    out = Path(base.paths.outputs)
    cache = out / "cache"
    data = {c: city_data(cfg, base, c, cache) for c in cfg.cities}
    log.info("features ready: " + ", ".join(f"{c} {X.shape[0]:,}" for c, (_, X) in data.items()))

    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    rows, plans, imp, serving = [], [], [], []
    for fi, sp in enumerate(load_splits()):
        fold = sp["test_cities"][0]
        city, Xte = data[fold]
        # the reduced network exists only when the city has more sites than it
        small = city.subset(np.arange(SITES_SMALL)) if city.true_pl.shape[1] > SITES_SMALL else None
        P, S = city.true_pl.shape
        te = load_tiles(base, cfg, fold)
        tile_ids = te.tile_row.to_numpy()
        for seed in SEEDS:
            rng = np.random.default_rng([fi, seed])
            parts = [sample_links(*data[c], N_PER_CITY, rng) for c in sp["train_cities"]]
            Xtr = np.vstack([p[0] for p in parts])
            ytr = np.concatenate([p[1] for p in parts])
            Xva, yva = sample_links(*data[sp["val_cities"][0]], N_PER_CITY, rng)
            m = lgb.LGBMRegressor(n_estimators=MAX_TREES, random_state=seed, **LINK_GBDT)
            kw = ({"eval_X": (Xva,), "eval_y": (yva,)} if _LGB_EVAL_XY
                  else {"eval_set": [(Xva, yva)]})
            m.fit(Xtr, ytr, callbacks=[lgb.early_stopping(PATIENCE, verbose=False)], **kw)
            m.booster_.save_model(str(models / f"link_{fold}_{seed}.txt"),
                                  num_iteration=m.best_iteration_)
            # hurdle: which links are below the cut-off (all pairs, censored included)
            parts = [sample_pairs(*data[c], N_PER_CITY, rng) for c in sp["train_cities"]]
            Xc = np.vstack([p[0] for p in parts])
            yc = np.concatenate([p[1] for p in parts])
            Xcv, ycv = sample_pairs(*data[sp["val_cities"][0]], N_PER_CITY, rng)
            clf = lgb.LGBMClassifier(n_estimators=MAX_TREES, random_state=seed, **LINK_GBDT)
            kw = ({"eval_X": (Xcv,), "eval_y": (ycv,)} if _LGB_EVAL_XY
                  else {"eval_set": [(Xcv, ycv)]})
            clf.fit(Xc, yc, callbacks=[lgb.early_stopping(PATIENCE, verbose=False)], **kw)
            clf.booster_.save_model(str(models / f"linkclf_{fold}_{seed}.txt"),
                                    num_iteration=clf.best_iteration_)
            PLr = m.predict(Xte, num_iteration=m.best_iteration_).reshape(P, S)
            keep = clf.predict_proba(Xte, num_iteration=clf.best_iteration_)[:, 1] \
                .reshape(P, S) >= 0.5
            PLh = np.where(keep, PLr, np.inf)
            gain = m.booster_.feature_importance("gain")
            imp.append(pd.Series(gain / gain.sum(), index=FEATURES).rename(f"{fold}_{seed}"))

            def run(op, draw, M, R, zero_shot=False):
                nets = [(city, M, R)] + ([(small, M[:, :SITES_SMALL], R[:, :SITES_SMALL])]
                                         if zero_shot and small is not None else [])
                for c, MM, RR in nets:
                    key = {"fold": fold, "operator": op, "seed": seed, "draw": draw,
                           "sites": MM.shape[1]}
                    rows.append({**key, "trees": int(m.best_iteration_),
                                 "trees_clf": int(clf.best_iteration_),
                                 "clf_acc": float((keep == city.obs).mean()),
                                 **c.evaluate(MM, pl_for_error=RR)})
                    for q in c.plan(MM):
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
                    pick = r2.choice(len(te), size=k, replace=False)
                    run(f"link+k{k}", draw, *calibrate(city, PLr, keep, tile_ids[pick]))
                pick = design_survey(city.tile, tile_ids, keep, k)
                run(f"link+k{k}d", 0, *calibrate(city, PLr, keep, tile_ids[pick]))
            sub = [r for r in rows if r["fold"] == fold and r["seed"] == seed
                   and r["sites"] == S]
            log.info(f"{fold} seed {seed}: trees={m.best_iteration_} " + " ".join(
                f"{o}={np.mean([r['pl_rmse_db'] for r in sub if r['operator'] == o]):.2f}"
                for o in ["link"] + [f"link+k{k}{d}" for k in KS for d in ("", "d")]))

    T = out / "tables"
    pd.DataFrame(rows).to_csv(T / "network_link.csv", index=False)
    pd.DataFrame(plans).to_csv(T / "planning_link.csv", index=False)
    pd.concat(imp, axis=1).mean(1).sort_values(ascending=False).rename("gain_share") \
        .to_csv(T / "link_importance.csv", index_label="feature")
    pd.concat(serving, ignore_index=True).to_parquet(
        out / "predictions" / "serving_link.parquet")
    log.info("wrote network_link.csv, planning_link.csv, link_importance.csv")


if __name__ == "__main__":
    main()
