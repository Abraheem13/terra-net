#!/usr/bin/env python
"""Robustness to the siting policy: models trained on the main corpus (sites
on the tallest rooftops) are applied, unchanged, to four cities re-traced with
sites on randomly chosen rooftops (configs/data/sionna_cities_randsite.yaml).

Operators scored on the random-site networks:
  oracle_rand   tile labels fitted on the random-site links (the tile model
                with perfect labels for this network)
  oracle_main   tile labels of the main corpus (tallest-rooftop sites): the
                exact labels of the same tiles under a different siting
  median, gbdt  the LOCO tile predictions of 04_fit_baselines.py
  link          the LOCO hurdle link models of 11_link_transfer.py (all seeds)
Only tiles labelled in both corpora are scored.

Output: outputs/tables/siting.csv
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS
from terranet.data.labels import fit_tiles, link_distance_m, load_links, log_distance
from terranet.data.link_features import city_link_matrix
from terranet.data.sionna_gen.geo import LocalFrame
from terranet.data.tiling import grid_frame, point_tile_index, region_from_points
from terranet.evaluation.network import City
from terranet.experiments.common import SEEDS, load_cfg
from terranet.utils.logging import get_logger

log = get_logger("siting")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--rand", default="configs/data/sionna_cities_randsite.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    rcfg = OmegaConf.load(args.rand)
    raw, out = Path(base.paths.raw), Path(base.paths.outputs)
    P = pd.read_parquet(out / "predictions" / "loco.parquet")
    rows = []
    for name in rcfg.rt_cities:
        main_meas, _ = load_links(raw / cfg.dataset / name / "measurements.parquet", cfg)
        meas, n_qc = load_links(raw / rcfg.dataset / name / "measurements.parquet", cfg)
        # the tile grid of the main corpus, so tiles are identical
        grid = grid_frame(region_from_points(main_meas.rx_lon.to_numpy(),
                                             main_meas.rx_lat.to_numpy()),
                          float(cfg.tile_size_m))
        tiles_main = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / name
                                     / "tiles.parquet")
        assert len(tiles_main) == len(grid), "grid mismatch with tiles.parquet"
        idx = point_tile_index(grid, meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy())
        x = log_distance(link_distance_m(meas), float(cfg.d0_m))
        f = fit_tiles(idx, x, meas.pathloss_db.to_numpy(float), meas.bs_id.to_numpy(),
                      len(grid), min_count=int(cfg.min_measurements_per_tile))
        tiles = tiles_main.copy()
        both = np.isfinite(f.gamma) & np.isfinite(tiles_main.gamma.to_numpy())
        tiles["gamma"] = np.where(both, f.gamma, np.nan)
        tiles["pl0"] = np.where(both, f.pl0, np.nan)
        meta = json.loads((raw / cfg.scenes_dir / name / "scene_meta.json").read_text())
        city = City(meas, grid, tiles, float(cfg.d0_m),
                    LocalFrame.from_center(meta["lat0"], meta["lon0"]))

        def score(op, seed, M, R=None):
            r = {"city": name, "operator": op, "seed": seed, "qc_removed": n_qc,
                 "tiles": int(both.sum()), "pixels": M.shape[0],
                 "site_h_median": float(np.median(city.sz))}
            r.update(city.evaluate(M, pl_for_error=R))
            r["regret5"] = next(q["regret"] for q in city.plan(M) if q["K"] == 5)
            rows.append(r)

        score("oracle_rand", 0, city.predicted_pl(tiles.gamma.to_numpy(), tiles.pl0.to_numpy()))
        score("oracle_main", 0, city.predicted_pl(tiles_main.gamma.to_numpy(),
                                                  tiles_main.pl0.to_numpy()))
        n = len(grid)
        for op in ("median", "gbdt"):
            for seed, g in P[(P.fold == name) & (P.operator == op)].groupby("seed"):
                gt, pt = np.full(n, np.nan), np.full(n, np.nan)
                gt[g.tile_row], pt[g.tile_row] = g.gamma_hat, g.pl0_hat
                score(op, seed, city.predicted_pl(gt, pt))
        X = city_link_matrix(city, raw / cfg.scenes_dir / name,
                             tiles_main[DESCRIPTOR_COLUMNS].to_numpy(np.float32),
                             float(cfg.freq_ghz) * 1e9)
        for seed in SEEDS:
            bst = lgb.Booster(model_file=str(out / "models" / f"link_{name}_{seed}.txt"))
            clf = lgb.Booster(model_file=str(out / "models" / f"linkclf_{name}_{seed}.txt"))
            R = bst.predict(X).reshape(city.true_pl.shape)
            keep = clf.predict(X).reshape(city.true_pl.shape) >= 0.5
            score("link", seed, np.where(keep, R, np.inf), R)
        sub = pd.DataFrame([r for r in rows if r["city"] == name])
        log.info(f"{name}: " + " ".join(f"{o}={v:.2f}" for o, v in
                                         sub.groupby("operator").pl_rmse_db.mean().items()))
    o = out / "tables" / "siting.csv"
    pd.DataFrame(rows).to_csv(o, index=False)
    log.info(f"wrote {o}")


if __name__ == "__main__":
    main()
