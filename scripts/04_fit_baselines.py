#!/usr/bin/env python
"""Phase 2: fit all baselines under LOCO and dump per-fold metrics to CSV.

Covers: global log-distance, Okumura-Hata, COST-231, 3GPP UMa NLOS,
base-paper Gaussian descriptor transfer (sigma selected on val cities),
descriptor MLP, descriptor GP.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from terranet.data.datasets import TileDataset
from terranet.evaluation.metrics import full_report
from terranet.models.baselines.descriptor_transfer import GaussianDescriptorTransfer
from terranet.utils.logging import get_logger
from terranet.utils.seed import seed_everything

log = get_logger("baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/baselines_loco.yaml")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load("configs/base.yaml")
    splits = json.loads(Path("data/splits/loco.json").read_text())
    rows = []
    for split in splits:
        seed_everything(base.seed)
        tr = TileDataset(base.paths.processed, split["train_cities"])
        va = TileDataset(base.paths.processed, split["val_cities"])
        te = TileDataset(base.paths.processed, split["test_cities"])
        Vtr, Ttr = tr.descriptor_matrix(), tr.target_matrix()
        Vva, Tva = va.descriptor_matrix(), va.target_matrix()
        Vte, Tte = te.descriptor_matrix(), te.target_matrix()

        gdt = GaussianDescriptorTransfer()
        sigma = gdt.select_sigma(Vtr, Ttr, Vva, Tva)
        pred = gdt.predict(Vte)
        rep = {"model": "gaussian_descriptor_transfer", "fold": split["name"], "sigma": sigma}
        rep.update(full_report(pred[:, 0], Tte[:, 0], "gamma_"))
        rep.update(full_report(pred[:, 1], Tte[:, 1], "pl0_"))
        rows.append(rep)
        log.info(f"{split['name']}: sigma={sigma:.2f} "
                 f"gamma_rmse={rep['gamma_rmse']:.3f} pl0_rmse={rep['pl0_rmse']:.2f}")
        # TODO: add MLP / GP fits here (torch/gpytorch), same report structure.
    out = Path(base.paths.outputs) / "tables" / "baselines_loco.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
