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

Outputs: outputs/tables/mixed_path.csv (network metrics), mixed_path_plan.csv
(planning), mixed_path_labels.csv (per-city fit).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS
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
    load_tiles,
    nw_predict,
    select_sigma,
)
from terranet.utils.logging import get_logger

log = get_logger("mixed")
MIN_LINKS = 30


def load(cfg, base, name):
    raw, proc = Path(base.paths.raw), Path(base.paths.processed)
    meas, _ = load_links(raw / cfg.dataset / name / "measurements.parquet", cfg)
    tiles = pd.read_parquet(proc / cfg.dataset / name / "tiles.parquet")
    grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()),
                      float(cfg.tile_size_m))
    meta = json.loads((raw / cfg.scenes_dir / name / "scene_meta.json").read_text())
    frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
    city = City(meas, grid, tiles, float(cfg.d0_m), frame)
    return city, mp.segment_lengths(city, grid, frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    out = Path(base.paths.outputs) / "tables"
    splits = load_splits()
    cities = [sp["test_cities"][0] for sp in splits]

    # intermediate results are cached so that an interrupted run resumes
    cache = Path(base.paths.outputs) / "predictions" / "mixed_path"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. mixed-path labels of every city (fitted on all of its links)
    labels, fits = {}, []
    for c in cities:
        t0 = time.time()
        done = [cache / f"labels_{c}_{m}.parquet" for m in mp.MODELS]
        if all(f.exists() for f in done) and (cache / f"fit_{c}.csv").exists():
            for m, f in zip(mp.MODELS, done, strict=True):
                labels[c, m] = pd.read_parquet(f)
            fits.extend(pd.read_csv(cache / f"fit_{c}.csv").to_dict("records"))
            continue
        city, L = load(cfg, base, c)
        x, obs = city.x.ravel(), city.obs.ravel()
        y = city.true_pl.ravel()[obs]
        for model in mp.MODELS:
            A, G, n = mp.fit(model, L[obs], x[obs], y)
            labels[c, model] = pd.DataFrame({"tile_row": np.arange(len(A)), "A": A,
                                             "gamma": G, "n_links": n})
            e = mp.predict(model, L[obs], x[obs], A, G) - y
            fits.append({"city": c, "model": model, "fit_rmse_db": float(np.sqrt((e ** 2).mean())),
                         "tiles": int((n >= MIN_LINKS).sum())})
        for m, f in zip(mp.MODELS, done, strict=True):
            labels[c, m].to_parquet(f)
        pd.DataFrame(fits[-2:]).to_csv(cache / f"fit_{c}.csv", index=False)
        log.info(f"{c}: {L.nnz:,} segments, fit " + ", ".join(
            f"{f['model']} {f['fit_rmse_db']:.2f} dB" for f in fits[-2:])
            + f" ({time.time() - t0:.0f} s)")
    pd.DataFrame(fits).to_csv(out / "mixed_path_labels.csv", index=False)

    def table(c, model, all_tiles=False):
        d = load_tiles(base, cfg, c, fitted_only=False)[["tile_row", *DESCRIPTOR_COLUMNS]]
        d = d.merge(labels[c, model], on="tile_row")
        if not all_tiles:
            d = d[d.n_links >= MIN_LINKS]
        return d

    # 2. transfer under leave-one-city-out
    rows, plans = [], []
    for sp in splits:
        fold = sp["test_cities"][0]
        fr, fp = cache / f"eval_{fold}.csv", cache / f"plan_{fold}.csv"
        if fr.exists() and fp.exists():
            rows.extend(pd.read_csv(fr).to_dict("records"))
            plans.extend(pd.read_csv(fp).to_dict("records"))
            continue
        n_rows, n_plans = len(rows), len(plans)
        city, L = load(cfg, base, fold)
        x = city.x.ravel()
        P, S = city.true_pl.shape
        for model in mp.MODELS:
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
                A = np.full(L.shape[1], med[0])
                G = np.full(L.shape[1], med[1])
                A[te.tile_row], G[te.tile_row] = p[:, 0], p[:, 1]
                PLh = mp.predict(model, L, x, A, G).reshape(P, S)
                key = {"fold": fold, "model": model, "operator": op}
                rows.append({**key, **city.evaluate(PLh)})
                plans.extend({**key, **q} for q in city.plan(PLh))
            log.info(f"{fold} {model}: " + "  ".join(
                f"{r['operator']}={r['pl_rmse_db']:.2f}/{r['assoc_acc']:.3f}"
                for r in rows[-len(preds):]))
        pd.DataFrame(rows[n_rows:]).to_csv(fr, index=False)
        pd.DataFrame(plans[n_plans:]).to_csv(fp, index=False)
    pd.DataFrame(rows).to_csv(out / "mixed_path.csv", index=False)
    pd.DataFrame(plans).to_csv(out / "mixed_path_plan.csv", index=False)
    log.info("wrote mixed_path.csv")


if __name__ == "__main__":
    main()
