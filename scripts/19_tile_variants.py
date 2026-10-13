#!/usr/bin/env python
"""Decomposition of the gap between the tile model and link transfer, and
non-learned per-link baselines (LOCO, network level).

Receiver-tile variants of the tile model, with exact labels (oracle), the
source median and GBDT transfer:
  <op>        the receiver-tile model of Proposition 1 (as in 09_network_eval)
  <op>+h      the same map with the hurdle of the link model of 11_link_transfer
              (a link is removed where the classifier predicts it beyond the
              cut-off; seed-matched)
  <op>+los    a dual-slope tile model: one (gamma, PL0) pair per tile for the
              line-of-sight links and one for the obstructed links, the LoS
              state of each (pixel, site) pair decided from the geometry
              (no building above the line of sight); GBDT transfers the four
              labels (inner-selected rounds, five seeds)
  <op>+los+h  both
The three variants show how much of the gap is the cut-off, how much the
line-of-sight state and how much the remaining geometry of the link.

Non-learned per-link baselines, each with the deterministic hurdle
"predicted path loss <= cut-off":
  fspl_ke     free space plus the single knife-edge loss of ITU-R P.526
  deygout     free space plus the Deygout three-edge loss of ITU-R P.526
  uma_geo     3GPP TR 38.901 UMa with the LoS state of each link from the
              geometry (PL_NLOS = max(PL_LOS, PL'_NLOS))
and the link model of 11_link_transfer for reference (same maps).

Outputs: outputs/tables/network_variants.csv, outputs/tables/planning_variants.csv,
         outputs/tables/los_labels.csv (per-city dual-slope label summary)
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS, scene_building_parts
from terranet.data.labels import fit_tiles
from terranet.data.link_features import (
    HeightMap,
    deygout_loss_db,
    free_space_db,
    uma_tr38901_db,
)
from terranet.data.sionna_gen.geo import LocalFrame
from terranet.experiments.common import (
    SEEDS,
    CityData,
    MaxAbs,
    fit_gbdt,
    load_cfg,
    load_splits,
    select_gbdt_trees,
)
from terranet.experiments.linkdata import KE_COL, LOS_COL, city_data, load_hurdle
from terranet.utils.logging import get_logger

log = get_logger("variants")
CUTOFF = 150.0
TARGETS4 = ["gamma_los", "pl0_los", "gamma_nlos", "pl0_nlos"]


def los_labels(cfg, city, X, n_tiles, n_min):
    """Dual-slope labels of every tile from the (pixel, site) matrix: a pair
    fitted to the LoS links and one to the obstructed links, each falling
    back to the all-link pair where fewer than n_min links exist."""
    obs = city.obs
    los = (X[:, LOS_COL].reshape(obs.shape) == 0)
    p, s = np.nonzero(obs)
    t, x, y = city.tile[p], city.x[p, s], city.true_pl[p, s]
    f_all = fit_tiles(t, x, y, s, n_tiles, min_count=n_min)
    out, n_los = {}, {}
    for name, m in (("los", los[p, s]), ("nlos", ~los[p, s])):
        f = fit_tiles(np.where(m, t, -1), x, y, s, n_tiles, min_count=n_min,
                      mu=f_all.mu, tau2=f_all.tau2)
        ok = np.isfinite(f.gamma)
        out[f"gamma_{name}"] = np.where(ok, f.gamma, f_all.gamma)
        out[f"pl0_{name}"] = np.where(ok, f.pl0, f_all.pl0)
        n_los[name] = ok
    return f_all, out, los, n_los


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    raw, proc = Path(base.paths.raw), Path(base.paths.processed)
    n_min, d0, f_hz = int(cfg.min_measurements_per_tile), float(cfg.d0_m), float(cfg.freq_ghz) * 1e9
    splits = load_splits()

    # 1. per city: City, features, LoS state, dual-slope labels
    cities, los_state, labels4, lab_rows = {}, {}, {}, []
    data4 = CityData(base, cfg, [])          # descriptors -> four labels
    for c in cfg.cities:
        city, X = city_data(cfg, base, c)
        tiles = pd.read_parquet(proc / cfg.dataset / c / "tiles.parquet")
        fitted = np.flatnonzero(np.isfinite(tiles.gamma.to_numpy()))
        f_all, lab, los, n_los = los_labels(cfg, city, X, len(tiles), n_min)
        d_all = float(np.nanmax(np.abs(f_all.gamma[fitted] - tiles.gamma.to_numpy()[fitted])))
        if d_all > 1e-6:
            log.warning(f"{c}: all-link labels differ from tiles.parquet by up to {d_all:.2e}")
        cities[c], los_state[c], labels4[c] = (city, X), los, lab
        Y4 = np.column_stack([lab[k][fitted] for k in TARGETS4])
        data4.d[c] = (tiles[DESCRIPTOR_COLUMNS].to_numpy(float)[fitted], Y4)
        share_los = float(los[city.obs].mean())
        lab_rows.append(dict(city=c, tiles=len(fitted), los_links_share=share_los,
                             tiles_los=int(n_los["los"][fitted].sum()),
                             tiles_nlos=int(n_los["nlos"][fitted].sum()),
                             gamma_los_median=float(np.median(lab["gamma_los"][fitted])),
                             gamma_nlos_median=float(np.median(lab["gamma_nlos"][fitted])),
                             pl0_los_median=float(np.median(lab["pl0_los"][fitted])),
                             pl0_nlos_median=float(np.median(lab["pl0_nlos"][fitted]))))
        log.info(f"{c}: LoS share {share_los:.2f}, gamma LoS/NLoS "
                 f"{lab_rows[-1]['gamma_los_median']:.2f}/{lab_rows[-1]['gamma_nlos_median']:.2f}")
    pd.DataFrame(lab_rows).to_csv(Path(base.paths.outputs) / "tables" / "los_labels.csv",
                                  index=False)

    # 2. leave-one-city-out
    rows, plans = [], []
    for sp in splits:
        fold, train = sp["test_cities"][0], sp["train_cities"]
        city, X = cities[fold]
        los = los_state[fold]
        tiles = pd.read_parquet(proc / cfg.dataset / fold / "tiles.parquet")
        n = len(tiles)
        P, S = city.true_pl.shape
        hurdles = {s: load_hurdle(base, fold, s) for s in SEEDS}

        def tile_map(g, p):
            return city.predicted_pl(g, p)

        def dual_map(lab):
            t = city.tile[:, None]
            g = np.where(los, lab["gamma_los"][t], lab["gamma_nlos"][t])
            a = np.where(los, lab["pl0_los"][t], lab["pl0_nlos"][t])
            return a + g * city.x

        def score(op, seed, M, R=None):
            key = {"fold": fold, "operator": op, "seed": seed, "sites": S}
            rows.append({**key, **city.evaluate(M, pl_for_error=R)})
            plans.extend({**key, **q} for q in city.plan(M))

        def with_hurdle(op, M, seed=None):
            for s in (SEEDS if seed is None else [seed]):
                _, keep = hurdles[s]
                score(f"{op}+h", s, np.where(keep, M, np.inf), M)

        # exact labels
        M = tile_map(tiles.gamma.to_numpy(), tiles.pl0.to_numpy())
        score("oracle", 0, M)
        with_hurdle("oracle", M)
        M = dual_map(labels4[fold])
        score("oracle+los", 0, M)
        with_hurdle("oracle+los", M)
        # source median of the four labels and of the two
        Ytr4 = data4.pool(train)[1]
        med = np.median(Ytr4, 0)
        lab_med = {k: np.full(n, med[i]) for i, k in enumerate(TARGETS4)}
        M = dual_map(lab_med)
        score("median+los", 0, M)
        with_hurdle("median+los", M)
        tr = pd.concat([pd.read_parquet(proc / cfg.dataset / c / "tiles.parquet") for c in train])
        tr = tr[np.isfinite(tr.gamma)]
        g_med, p_med = float(tr.gamma.median()), float(tr.pl0.median())
        M = tile_map(np.full(n, g_med), np.full(n, p_med))
        score("median", 0, M)
        with_hurdle("median", M)
        # GBDT transfer of the two labels (predictions of 04) and of the four
        loco = pd.read_parquet(Path(base.paths.outputs) / "predictions" / "loco.parquet")
        loco = loco[(loco.fold == fold) & (loco.operator == "gbdt")]
        for seed, g in loco.groupby("seed"):
            gt, pt = np.full(n, g_med), np.full(n, p_med)
            gt[g.tile_row], pt[g.tile_row] = g.gamma_hat, g.pl0_hat
            M = tile_map(gt, pt)
            score("gbdt", seed, M)
            with_hurdle("gbdt", M, seed)
        trees = select_gbdt_trees(data4, train, seed=0)
        Xtr, _ = data4.pool(train)
        norm = MaxAbs().fit(Xtr)
        Xte = norm(tiles[DESCRIPTOR_COLUMNS].to_numpy(float))
        fitted = np.flatnonzero(np.isfinite(tiles.gamma.to_numpy()))
        for seed in SEEDS:
            pred = fit_gbdt(norm(Xtr), Ytr4, trees, seed).predict(Xte[fitted])
            lab = {k: np.full(n, med[i]) for i, k in enumerate(TARGETS4)}
            for i, k in enumerate(TARGETS4):
                lab[k][fitted] = pred[:, i]
            M = dual_map(lab)
            score("gbdt+los", seed, M)
            with_hurdle("gbdt+los", M, seed)

        # non-learned per-link baselines
        d3 = d0 * 10 ** (city.x / 10)
        d2 = np.hypot(city.px[:, None] - city.sx[None, :], city.py[:, None] - city.sy[None, :])
        fs = free_space_db(d3, f_hz)
        ke = X[:, KE_COL].reshape(P, S)
        meta = json.loads((raw / cfg.scenes_dir / fold / "scene_meta.json").read_text())
        frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
        bld = gpd.read_file(raw / cfg.scenes_dir / fold / "building.geojson")
        parts, h = scene_building_parts(bld, frame, float(meta["size_m"]))
        hm = HeightMap(parts, h, float(meta["size_m"]))
        dey = deygout_loss_db(hm, city.px, city.py, city.rx_h, city.sx, city.sy, city.sz,
                              f_hz).reshape(P, S)
        uma = uma_tr38901_db(d3, d2, city.sz[None, :], f_hz, los)
        for op, R in (("fspl_ke", fs + ke), ("deygout", fs + dey), ("uma_geo", uma)):
            score(op, 0, np.where(R <= CUTOFF, R, np.inf), R)
        for s, (PLr, keep) in hurdles.items():
            score("link", s, np.where(keep, PLr, np.inf), PLr)
        sub = pd.DataFrame([r for r in rows if r["fold"] == fold])
        log.info(f"{fold}: " + " ".join(f"{o}={v:.2f}" for o, v in
                                        sub.groupby("operator").pl_rmse_db.mean().items()))

    T = Path(base.paths.outputs) / "tables"
    pd.DataFrame(rows).to_csv(T / "network_variants.csv", index=False)
    pd.DataFrame(plans).to_csv(T / "planning_variants.csv", index=False)
    log.info("wrote network_variants.csv, planning_variants.csv, los_labels.csv")


if __name__ == "__main__":
    main()
