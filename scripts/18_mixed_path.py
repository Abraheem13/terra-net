#!/usr/bin/env python
"""Mixed-path tile models: the composition of the published framework.

A receiver-tile model gives every link seen from a pixel the parameters of
the pixel's tile (Proposition 1). The framework of Ozyurt (2026), following
MAPLE, instead composes the loss of a link from every tile its straight path
crosses. This script evaluates that composition on the corpus, in the two
variants of `terranet.evaluation.mixed_path` (sum, as published; average,
path-length weighted), under the protocol of the paper:

  1. every city: tile parameters (A_t, gamma_t) fitted jointly to all of its
     ray-traced links (the mixed-path labels; "oracle" when the city is the
     target);
  2. every leave-one-city-out split: the labels of the tiles crossed by at
     least 30 links of the training cities are transferred to every tile of
     the held-out city with the source median, Gaussian-kernel transfer
     (bandwidth chosen on the validation city) and GBDT (early stopping on the
     validation city), and the predicted map is scored with the network
     metrics and site planning of `City`.

With --tile-size G the tiles of the composition (and their descriptors) have
side G while the pixels scored are those of the main 100 m corpus, so that
every tile size is evaluated on the same pixels; the framework was evaluated
with tiles of 200 m to 1000 m.

Outputs: outputs/tables/mixed_path.csv (network metrics), mixed_path_plan.csv
(planning), mixed_path_labels.csv (per-city fit); with a tile size other than
the corpus one, the same files with the suffix _G<size>.
"""
import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import (
    DESCRIPTOR_COLUMNS,
    scene_building_parts,
    tile_descriptors,
)
from terranet.data.labels import load_links
from terranet.data.sionna_gen.geo import LocalFrame
from terranet.data.tiling import grid_frame, region_from_points
from terranet.evaluation import mixed_path as mp
from terranet.evaluation.network import City
from terranet.experiments.common import (
    MaxAbs,
    fit_gbdt,
    load_cfg,
    load_splits,
    nw_predict,
    select_sigma,
)
from terranet.utils.logging import get_logger

log = get_logger("mixed")
MIN_LINKS = 30


def load(cfg, base, name, G):
    """The city (pixels of the main corpus), the segment lengths of every
    (pixel, site) path over tiles of side G, and the descriptors of those
    tiles (column tile_row = grid row)."""
    raw, proc = Path(base.paths.raw), Path(base.paths.processed)
    meas, _ = load_links(raw / cfg.dataset / name / "measurements.parquet", cfg)
    tiles = pd.read_parquet(proc / cfg.dataset / name / "tiles.parquet")
    region = region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy())
    grid0 = grid_frame(region, float(cfg.tile_size_m))
    scene = raw / cfg.scenes_dir / name
    meta = json.loads((scene / "scene_meta.json").read_text())
    frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
    city = City(meas, grid0, tiles, float(cfg.d0_m), frame)
    if G == float(cfg.tile_size_m):
        grid, desc = grid0, tiles[DESCRIPTOR_COLUMNS].reset_index(drop=True)
    else:
        grid = grid_frame(region, G)
        bld = gpd.read_file(scene / "building.geojson")
        bld = bld[bld.geometry.notna()].reset_index(drop=True)
        parts, h = scene_building_parts(bld, frame, float(meta["size_m"]))
        desc = tile_descriptors(grid, parts, h, frame)[DESCRIPTOR_COLUMNS].reset_index(drop=True)
    desc.insert(0, "tile_row", np.arange(len(desc)))
    return city, mp.segment_lengths(city, grid, frame, d0=float(cfg.d0_m)), desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--tile-size", type=float, default=None)
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    gs = float(args.tile_size or cfg.tile_size_m)
    tag = "" if gs == float(cfg.tile_size_m) else f"_G{gs:.0f}"
    out = Path(base.paths.outputs) / "tables"
    splits = load_splits()
    cities = [sp["test_cities"][0] for sp in splits]

    # intermediate results are cached so that an interrupted run resumes
    cache = Path(base.paths.outputs) / "predictions" / f"mixed_path{tag}"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. mixed-path labels of every city (fitted on all of its links)
    def cached(path, model):
        """Per-model cache file, or the rows of `model` in an older combined one."""
        f = cache / path.format(m=model)
        if f.exists():
            return pd.read_csv(f)
        old = cache / path.replace("_{m}", "")
        if old.exists():
            d = pd.read_csv(old)
            d = d[d.model == model]
            return d if len(d) else None
        return None

    labels, fits, descs = {}, [], {}
    for c in cities:
        t0 = time.time()
        fd = cache / f"desc_{c}.parquet"
        todo = []
        for m in mp.MODELS:
            fl, ff = cache / f"labels_{c}_{m}.parquet", cached(f"fit_{c}_{{m}}.csv", m)
            if fl.exists() and ff is not None:
                labels[c, m] = pd.read_parquet(fl)
                fits.extend(ff.to_dict("records"))
            else:
                todo.append(m)
        if not todo and fd.exists():
            descs[c] = pd.read_parquet(fd)
            continue
        city, paths, descs[c] = load(cfg, base, c, gs)
        descs[c].to_parquet(fd)
        x, obs = city.x.ravel(), city.obs.ravel()
        y = city.true_pl.ravel()[obs]
        po = paths.rows(obs)
        for model in todo:
            A, G, n = mp.fit(model, po, x[obs], y)
            labels[c, model] = pd.DataFrame({"tile_row": np.arange(len(A)), "A": A,
                                             "gamma": G, "n_links": n})
            e = mp.predict(model, po, x[obs], A, G) - y
            row = {"city": c, "model": model, "fit_rmse_db": float(np.sqrt((e ** 2).mean())),
                   "tiles": int((n >= MIN_LINKS).sum())}
            fits.append(row)
            labels[c, model].to_parquet(cache / f"labels_{c}_{model}.parquet")
            pd.DataFrame([row]).to_csv(cache / f"fit_{c}_{model}.csv", index=False)
            log.info(f"{c} {model}: fit {row['fit_rmse_db']:.2f} dB "
                     f"({paths.L.nnz:,} segments, {time.time() - t0:.0f} s)")
    pd.DataFrame(fits).sort_values(["city", "model"]).to_csv(
        out / f"mixed_path_labels{tag}.csv", index=False)

    def table(c, model, all_tiles=False):
        d = descs[c].merge(labels[c, model], on="tile_row")
        if not all_tiles:
            d = d[d.n_links >= MIN_LINKS]
        return d

    # 2. transfer under leave-one-city-out
    rows, plans = [], []
    for sp in splits:
        fold = sp["test_cities"][0]
        todo = []
        for m in mp.MODELS:
            er, ep = cached(f"eval_{fold}_{{m}}.csv", m), cached(f"plan_{fold}_{{m}}.csv", m)
            if er is not None and ep is not None:
                rows.extend(er.to_dict("records"))
                plans.extend(ep.to_dict("records"))
            else:
                todo.append(m)
        if not todo:
            continue
        city, paths, _ = load(cfg, base, fold, gs)
        x = city.x.ravel()
        P, S = city.true_pl.shape
        T = paths.shape[1]
        for model in todo:
            n_rows, n_plans = len(rows), len(plans)
            tr = pd.concat([table(c, model) for c in sp["train_cities"]])
            va = pd.concat([table(c, model) for c in sp["val_cities"]])
            te = table(fold, model, all_tiles=True)
            Xtr, Ytr = tr[DESCRIPTOR_COLUMNS].to_numpy(float), tr[["A", "gamma"]].to_numpy()
            Xva, Yva = va[DESCRIPTOR_COLUMNS].to_numpy(float), va[["A", "gamma"]].to_numpy()
            Xte = te[DESCRIPTOR_COLUMNS].to_numpy(float)
            norm = MaxAbs().fit(Xtr)
            Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)
            ok = np.isfinite(Xte).all(1)
            med = np.median(Ytr, 0)
            sigma, _ = select_sigma(Xtr, Ytr, Xva, Yva)
            gb = fit_gbdt(Xtr, Ytr, Xva, Yva, seed=0)
            preds = {"oracle": te[["A", "gamma"]].to_numpy(),
                     "median": np.tile(med, (len(te), 1))}
            for op, f in (("kernel", lambda X: nw_predict(X, Xtr, Ytr, sigma)),
                          ("gbdt", gb.predict)):
                p = np.tile(med, (len(te), 1))
                p[ok] = f(Xte[ok])
                preds[op] = p
            for op, p in preds.items():
                A, G = np.full(T, med[0]), np.full(T, med[1])
                A[te.tile_row], G[te.tile_row] = p[:, 0], p[:, 1]
                PLh = mp.predict(model, paths, x, A, G).reshape(P, S)
                key = {"fold": fold, "model": model, "operator": op}
                rows.append({**key, **city.evaluate(PLh)})
                plans.extend({**key, **q} for q in city.plan(PLh))
            log.info(f"{fold} {model}: " + "  ".join(
                f"{r['operator']}={r['pl_rmse_db']:.2f}/{r['assoc_acc']:.3f}"
                for r in rows[n_rows:]))
            pd.DataFrame(rows[n_rows:]).to_csv(cache / f"eval_{fold}_{model}.csv", index=False)
            pd.DataFrame(plans[n_plans:]).to_csv(cache / f"plan_{fold}_{model}.csv",
                                                 index=False)
    pd.DataFrame(rows).to_csv(out / f"mixed_path{tag}.csv", index=False)
    pd.DataFrame(plans).to_csv(out / f"mixed_path_plan{tag}.csv", index=False)
    log.info(f"wrote mixed_path{tag}.csv")


if __name__ == "__main__":
    main()
