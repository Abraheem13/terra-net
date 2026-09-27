#!/usr/bin/env python
"""Sensitivity to the tile size G (50, 100, 200 m).

For every G the whole tile pipeline is rerun: labels (empirical-Bayes, CR2),
descriptors, zero-shot LOCO transfer (source median, ridge, GBDT with five
seeds; the protocol of 04_fit_baselines.py) and the network evaluation of the
oracle labels and the transferred parameters (09_network_eval.py). G = 100 m
reproduces the main pipeline.

Output: outputs/tables/tile_size.csv  one row per (G, fold, operator, seed)
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import (
    DESCRIPTOR_COLUMNS,
    scene_building_parts,
    tile_descriptors,
)
from terranet.data.labels import fit_tiles, link_distance_m, load_links, log_distance
from terranet.data.sionna_gen.geo import LocalFrame
from terranet.data.tiling import grid_frame, point_tile_index, region_from_points
from terranet.evaluation.network import City
from terranet.experiments.common import (
    SEEDS,
    TARGETS,
    MaxAbs,
    fit_gbdt,
    fit_ridge,
    load_cfg,
    load_splits,
)
from terranet.utils.logging import get_logger

log = get_logger("tile_size")
SIZES = (50.0, 100.0, 200.0)


def city_tiles(cfg, base, city, G):
    raw = Path(base.paths.raw)
    meas, _ = load_links(raw / cfg.dataset / city / "measurements.parquet", cfg)
    scene = raw / cfg.scenes_dir / city
    meta = json.loads((scene / "scene_meta.json").read_text())
    frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
    grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()), G)
    idx = point_tile_index(grid, meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy())
    x = log_distance(link_distance_m(meas), float(cfg.d0_m))
    f = fit_tiles(idx, x, meas.pathloss_db.to_numpy(float), meas.bs_id.to_numpy(), len(grid),
                  min_count=int(cfg.min_measurements_per_tile))
    bld = gpd.read_file(scene / "building.geojson")
    bld = bld[bld.geometry.notna()].reset_index(drop=True)
    parts, h = scene_building_parts(bld, frame, float(meta["size_m"]))
    tiles = pd.concat([grid.reset_index(drop=True), f.frame(),
                       tile_descriptors(grid, parts, h, frame)], axis=1)
    return tiles, City(meas, grid, tiles, float(cfg.d0_m), frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    rows = []
    for G in SIZES:
        data = {c: city_tiles(cfg, base, c, G) for c in cfg.cities}
        fitted = {c: t[np.isfinite(t.gamma)].reset_index() for c, (t, _) in data.items()}
        for sp in load_splits():
            fold = sp["test_cities"][0]
            tr = pd.concat([fitted[c] for c in sp["train_cities"]], ignore_index=True)
            va = pd.concat([fitted[c] for c in sp["val_cities"]], ignore_index=True)
            te = fitted[fold]
            tiles, city = data[fold]
            norm = MaxAbs().fit(tr[DESCRIPTOR_COLUMNS].to_numpy(float))
            X = {k: norm(d[DESCRIPTOR_COLUMNS].to_numpy(float)) for k, d in
                 (("tr", tr), ("va", va), ("te", te))}
            Y = {k: d[TARGETS].to_numpy(float) for k, d in (("tr", tr), ("va", va), ("te", te))}
            labels = dict(n_tiles=len(te),
                          shrink_median=float(np.median(te.shrink)),
                          se_iid=float(np.sqrt(np.nanmean(te.gamma_se ** 2))),
                          se_cl=float(np.sqrt(np.nanmean(te.gamma_se_cl ** 2))),
                          fit_rmse_db=float(np.sqrt(np.average(te.fit_rmse_db ** 2, weights=te.n))))

            def score(op, seed, pred):
                gt = np.full(len(tiles), np.nan)
                pt = np.full(len(tiles), np.nan)
                gt[te["index"]], pt[te["index"]] = pred[:, 0], pred[:, 1]
                e = pred - Y["te"]
                r = dict(G=G, fold=fold, operator=op, seed=seed,
                         gamma_rmse=float(np.sqrt((e[:, 0] ** 2).mean())),
                         pl0_rmse=float(np.sqrt((e[:, 1] ** 2).mean())), **labels)
                ev = city.evaluate(city.predicted_pl(gt, pt))
                r.update({k: ev[k] for k in ("pl_rmse_db", "assoc_acc", "sinr_mae_db",
                                             "cov_acc_130", "edge_disp_m")})
                rows.append(r)

            score("oracle", 0, Y["te"])
            score("median", 0, np.tile(np.median(Y["tr"], 0), (len(te), 1)))
            rg, _ = fit_ridge(X["tr"], Y["tr"], X["va"], Y["va"])
            score("ridge", 0, rg.predict(X["te"]))
            for seed in SEEDS:
                score("gbdt", seed, fit_gbdt(X["tr"], Y["tr"], X["va"], Y["va"], seed)
                      .predict(X["te"]))
            log.info(f"G={G:.0f} {fold}: " + " ".join(
                f"{o}={np.mean([r['pl_rmse_db'] for r in rows if r['G'] == G and r['fold'] == fold and r['operator'] == o]):.2f}"  # noqa: E501
                for o in ("oracle", "median", "ridge", "gbdt")))
    out = Path(base.paths.outputs) / "tables" / "tile_size.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby(["G", "operator"])[["gamma_rmse", "pl_rmse_db", "assoc_acc"]].mean().round(3))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
