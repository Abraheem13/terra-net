#!/usr/bin/env python
"""Empirical check of the kernel-collapse analysis (Theorem 1) on every fold.

For the bandwidth sigma selected on the validation city (loco_hparams.csv) and
every target tile u:
  D_u      spread of squared descriptor distances to the source tiles
  eps_u    D_u / (2 sigma^2)   (Theorem 1 requires eps_u < 1)
  n_eff/N  effective neighbourhood fraction
  w_max    largest single normalised weight (-> 1 in the nearest-neighbour limit)
  dev_u    |gamma_hat_u - mean source gamma|
  bound_u  D_u sigma_gamma / (4 sigma^2)
and, per fold, the ratio of the spread of kernel predictions to the spread of
the true target labels (0 for a constant predictor).

Output: outputs/tables/degeneracy.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.experiments.common import MaxAbs, load_cfg, load_splits, load_tiles, stack, xy
from terranet.utils.logging import get_logger

log = get_logger("degeneracy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    hp = pd.read_csv(Path(base.paths.outputs) / "tables" / "loco_hparams.csv").set_index("fold")
    rows = []
    for sp in load_splits():
        fold = sp["test_cities"][0]
        Xtr, Ytr = xy(stack(base, cfg, sp["train_cities"]))
        Xte, Yte = xy(load_tiles(base, cfg, fold))
        norm = MaxAbs().fit(Xtr)
        Xtr, Xte = norm(Xtr), norm(Xte)
        sigma = float(hp.loc[fold, "sigma"])
        g = Ytr[:, 0]
        g_bar, s_g = g.mean(), g.std()
        sq = (Xtr ** 2).sum(1)
        D, neff, wmax, dev, pred = [], [], [], [], []
        for s in range(0, len(Xte), args.batch):
            q = Xte[s:s + args.batch]
            d2 = np.maximum(sq[None] + (q ** 2).sum(1)[:, None] - 2 * q @ Xtr.T, 0.0)
            dmin = d2.min(1, keepdims=True)
            w = np.exp(-(d2 - dmin) / (2 * sigma ** 2))
            w /= w.sum(1, keepdims=True)
            p = w @ g
            D.append(d2.max(1) - dmin[:, 0])
            neff.append(1.0 / (w ** 2).sum(1))
            wmax.append(w.max(1))
            dev.append(np.abs(p - g_bar))
            pred.append(p)
        D, neff, wmax, dev, pred = map(np.concatenate, (D, neff, wmax, dev, pred))
        eps = D / (2 * sigma ** 2)
        bound = D * s_g / (4 * sigma ** 2)
        inr = eps < 1
        rows.append(dict(
            fold=fold, sigma=sigma, N=len(Xtr), eps_median=float(np.median(eps)),
            frac_eps_lt1=float(inr.mean()), neff_frac=float(np.median(neff) / len(Xtr)),
            wmax_median=float(np.median(wmax)), dev_mean=float(dev.mean()),
            bound_holds_in_regime=(float((dev[inr] <= bound[inr] + 1e-12).mean())
                                   if inr.any() else np.nan),
            pred_spread_ratio=float(pred.std() / Yte[:, 0].std())))
        r = rows[-1]
        log.info(f"{fold:10s} sigma={sigma:<5g} eps_med={r['eps_median']:.3g} "
                 f"neff/N={r['neff_frac']:.3g} wmax={r['wmax_median']:.3f} "
                 f"spread={r['pred_spread_ratio']:.2f}")
    out = Path(base.paths.outputs) / "tables" / "degeneracy.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
