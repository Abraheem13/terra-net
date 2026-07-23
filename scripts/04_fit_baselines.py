#!/usr/bin/env python
"""LOCO baseline ladder for cross-city (gamma, PL0) transfer.

Models:
  empirical_median   - predict train-set median (gamma, PL0); the no-info floor
  gaussian_transfer  - base paper: similarity-weighted average over training tiles
  descriptor_mlp     - MLP on 116-dim descriptors
  descriptor_ridge   - linear ridge (fast, strong linear baseline)

Correctness: descriptor normalization is REFIT on training cities only and
applied to the held-out city (no test-statistic leakage). Raw (unnormalized)
descriptors are read from disk; each fold normalizes from scratch.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from terranet.models.baselines.descriptor_transfer import GaussianDescriptorTransfer
from terranet.evaluation.metrics import full_report
from terranet.utils.logging import get_logger
from terranet.utils.seed import seed_everything

log = get_logger("baselines")
DCOLS = [f"descriptor_{i}" for i in range(116)]


def load_city(processed, dataset, city):
    d = pd.read_parquet(Path(processed) / dataset / city / "tiles.parquet")
    d = d[np.isfinite(d.gamma)].reset_index(drop=True)
    return d[DCOLS].to_numpy(np.float32), d[["gamma", "pl0"]].to_numpy(np.float32)


def refit_norm(X):
    """Column max over TRAIN only; returns normalizer fn."""
    mx = np.abs(X).max(0); mx[mx == 0] = 1.0
    return lambda Z: Z / mx


def stack(processed, dataset, cities):
    Xs, Ys = [], []
    for c in cities:
        X, Y = load_city(processed, dataset, c); Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load("configs/base.yaml")
    splits = json.loads(Path("data/splits/loco.json").read_text())
    proc, ds = base.paths.processed, cfg.dataset

    rows = []
    for sp in splits:
        seed_everything(args.seed)
        Xtr_raw, Ytr = stack(proc, ds, sp["train_cities"])
        Xva_raw, Yva = stack(proc, ds, sp["val_cities"])
        Xte_raw, Yte = load_city(proc, ds, sp["test_cities"][0])

        norm = refit_norm(Xtr_raw)                    # TRAIN-only normalization
        Xtr, Xva, Xte = norm(Xtr_raw), norm(Xva_raw), norm(Xte_raw)
        fold = sp["test_cities"][0]

        def record(model, pred):
            r = {"model": model, "fold": fold}
            r.update(full_report(pred[:, 0], Yte[:, 0], "gamma_"))
            r.update(full_report(pred[:, 1], Yte[:, 1], "pl0_"))
            rows.append(r)
            log.info(f"{fold:11s} {model:20s} "
                     f"gamma_rmse={r['gamma_rmse']:.3f} pl0_rmse={r['pl0_rmse']:.2f}")

        # 1. no-info floor
        record("empirical_median",
                np.tile(np.median(Ytr, 0), (len(Yte), 1)))

        # 2. base-paper Gaussian descriptor transfer (sigma tuned on val)
        gdt = GaussianDescriptorTransfer()
        gdt.select_sigma(Xtr, Ytr, Xva, Yva)
        record("gaussian_transfer", gdt.predict(Xte))

        # 3. ridge
        rg = Ridge(alpha=1.0).fit(Xtr, Ytr)
        record("descriptor_ridge", rg.predict(Xte))

        # 4. MLP
        mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=400,
                           early_stopping=True, random_state=args.seed).fit(Xtr, Ytr)
        record("descriptor_mlp", mlp.predict(Xte))

    out = Path(base.paths.outputs) / "tables" / "baselines_loco.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    log.info("\n=== LOCO mean gamma_rmse by model ===")
    summary = df.groupby("model")[["gamma_rmse", "pl0_rmse"]].agg(["mean", "std"])
    print(summary.round(3))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
