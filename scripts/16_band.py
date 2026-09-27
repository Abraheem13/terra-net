#!/usr/bin/env python
"""Second band (7.5 GHz): tile labels across bands and cross-band transfer.

Labels: for the tiles labelled in both bands, the median exponent per band,
the correlation of the per-tile exponents and the median intercept change.

Cross-band transfer: the LOCO hurdle link models trained at 3.5 GHz (never
having seen the held-out city in either band) predict the 7.5 GHz links, with
the free-space frequency offset 20 log10(7.5 / 3.5) dB added to the path loss;
a link is kept when the classifier predicts it below the cut-off and its
scaled path loss is at most the cut-off:
  link_x_phys  features computed at 7.5 GHz (the knife-edge parameter nu
               depends on the wavelength; every other feature is geometric)
  link_x_fs    features computed at 3.5 GHz (frequency offset only)
Both are scored with the network evaluation of 09/11 on the 7.5 GHz truth and
compared with the models trained at 7.5 GHz (outputs/sionna_7p5/tables).

Output: outputs/tables/band_labels.csv, outputs/tables/band_transfer.csv
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS
from terranet.data.link_features import city_link_matrix
from terranet.evaluation.network import load_city
from terranet.experiments.common import SEEDS, load_cfg
from terranet.utils.logging import get_logger

log = get_logger("band")
CUTOFF_DB = 150.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--band2", default="configs/data/sionna_7p5.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    cfg2, base2 = load_cfg(args.band2)
    f1, f2 = float(cfg.freq_ghz), float(cfg2.freq_ghz)
    offset = 20 * np.log10(f2 / f1)
    proc, raw = Path(base.paths.processed), Path(base.paths.raw)
    models = Path(base.paths.outputs) / "models"
    lab, rows = [], []
    for name in cfg.cities:
        t1 = pd.read_parquet(proc / cfg.dataset / name / "tiles.parquet")
        t2 = pd.read_parquet(proc / cfg2.dataset / name / "tiles.parquet")
        both = np.isfinite(t1.gamma.to_numpy()) & np.isfinite(t2.gamma.to_numpy())
        g1, g2 = t1.gamma.to_numpy()[both], t2.gamma.to_numpy()[both]
        lab.append(dict(city=name, tiles=int(both.sum()),
                        gamma_median_1=float(np.median(g1)), gamma_median_2=float(np.median(g2)),
                        gamma_corr=float(np.corrcoef(g1, g2)[0, 1]),
                        pl0_shift_median=float(np.median(t2.pl0.to_numpy()[both]
                                                         - t1.pl0.to_numpy()[both]))))
        city = load_city(cfg2, base2, name)
        D = t2[DESCRIPTOR_COLUMNS].to_numpy(np.float32)
        cache = Path(base2.paths.outputs) / "cache" / f"link_{name}.npy"
        X2 = np.load(cache) if cache.exists() else None
        if X2 is None or X2.shape[0] != city.true_pl.size:
            X2 = city_link_matrix(city, raw / cfg.scenes_dir / name, D, f2 * 1e9)
        X1 = city_link_matrix(city, raw / cfg.scenes_dir / name, D, f1 * 1e9)
        for seed in SEEDS:
            bst = lgb.Booster(model_file=str(models / f"link_{name}_{seed}.txt"))
            clf = lgb.Booster(model_file=str(models / f"linkclf_{name}_{seed}.txt"))
            for op, X in (("link_x_phys", X2), ("link_x_fs", X1)):
                R = bst.predict(X).reshape(city.true_pl.shape) + offset
                keep = (clf.predict(X).reshape(city.true_pl.shape) >= 0.5) & (R <= CUTOFF_DB)
                M = np.where(keep, R, np.inf)
                r = {"fold": name, "operator": op, "seed": seed}
                r.update(city.evaluate(M, pl_for_error=R))
                r["regret5"] = next(q["regret"] for q in city.plan(M) if q["K"] == 5)
                rows.append(r)
        sub = pd.DataFrame([r for r in rows if r["fold"] == name])
        log.info(f"{name}: gamma {lab[-1]['gamma_median_1']:.2f} -> "
                 f"{lab[-1]['gamma_median_2']:.2f} (r={lab[-1]['gamma_corr']:.2f}); " + " ".join(
                     f"{o}={v:.2f}" for o, v in sub.groupby("operator").pl_rmse_db.mean().items()))
    T = Path(base.paths.outputs) / "tables"
    pd.DataFrame(lab).to_csv(T / "band_labels.csv", index=False)
    pd.DataFrame(rows).to_csv(T / "band_transfer.csv", index=False)
    log.info("wrote band_labels.csv and band_transfer.csv")


if __name__ == "__main__":
    main()
