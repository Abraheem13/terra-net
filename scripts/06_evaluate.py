#!/usr/bin/env python
"""Phase 3/4: evaluate a run dir on its held-out city: metrics + conformal UQ + k-shot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from terranet.data.datasets import TileDataset
from terranet.evaluation.calibration import gaussian_interval_coverage, regression_ece
from terranet.evaluation.metrics import full_report
from terranet.models.uq.conformal import SplitConformal
from terranet.training.trainer import TerraNetModule


@torch.no_grad()
def collect(model, loader, device):
    means, sig, tgt = [], [], []
    for desc, ras, t, _ in loader:
        out = model(desc.to(device), ras.to(device))
        means.append(out["mean"].cpu()); sig.append(out["logvar"].exp().sqrt().cpu()); tgt.append(t)
    return (torch.cat(means).numpy(), torch.cat(sig).numpy(), torch.cat(tgt).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = OmegaConf.load(run / "config.yaml")
    fold = run.name.split("_s")[0].replace(f"{cfg.name}_", "")
    splits = {s["name"]: s for s in json.loads(Path("data/splits/loco.json").read_text())}
    split = splits[fold]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TerraNetModule(cfg).to(device).eval()
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))

    va = DataLoader(TileDataset(cfg.paths.processed, split["val_cities"]), batch_size=256)
    te = DataLoader(TileDataset(cfg.paths.processed, split["test_cities"]), batch_size=256)
    mu_v, sd_v, t_v = collect(model, va, device)
    mu_t, sd_t, t_t = collect(model, te, device)

    report = {}
    for j, name in enumerate(["gamma", "pl0"]):
        report.update(full_report(mu_t[:, j], t_t[:, j], f"{name}_"))
        cp = SplitConformal(alpha=cfg.uq.conformal_alpha).calibrate(
            mu_v[:, j], t_v[:, j], sd_v[:, j])
        lo, hi = cp.interval(mu_t[:, j], sd_t[:, j])
        report[f"{name}_cp_coverage"] = SplitConformal.empirical_coverage(lo, hi, t_t[:, j])
        report[f"{name}_cp_width"] = float((hi - lo).mean())
        report[f"{name}_ece"] = regression_ece(mu_t[:, j], sd_t[:, j], t_t[:, j])
        report[f"{name}_gauss_cov"] = gaussian_interval_coverage(
            mu_t[:, j], sd_t[:, j], t_t[:, j])
    (run / "eval.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if isinstance(v, float)}, indent=2))


if __name__ == "__main__":
    main()
