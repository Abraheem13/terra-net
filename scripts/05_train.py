#!/usr/bin/env python
"""Phase 3: train TERRA-Net under LOCO (one run per fold x seed x ablation)."""
import argparse
import json
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from terranet.data.datasets import TileDataset
from terranet.training.trainer import TerraNetModule
from terranet.utils.io import snapshot_run
from terranet.utils.seed import seed_everything


class DAPairLoader:
    """Yields {'source': batch, 'target': batch} pairs for DA training."""

    def __init__(self, src: DataLoader, tgt: DataLoader):
        self.src, self.tgt = src, tgt

    def __iter__(self):
        tgt_it = iter(self.tgt)
        for sb in self.src:
            try:
                tb = next(tgt_it)
            except StopIteration:
                tgt_it = iter(self.tgt)
                tb = next(tgt_it)
            yield {"source": sb, "target": tb}

    def __len__(self):
        return len(self.src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/terranet_loco.yaml")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load("configs/base.yaml"),
                          OmegaConf.load("configs/model/terranet.yaml"),
                          OmegaConf.load(args.config))
    splits = json.loads(Path("data/splits/loco.json").read_text())
    for split in splits:
        if args.fold and split["name"] != args.fold:
            continue
        seed_everything(args.seed)
        run_dir = Path(cfg.paths.outputs) / "checkpoints" / f"{cfg.name}_{split['name']}_s{args.seed}"
        snapshot_run(cfg, run_dir)
        tr = TileDataset(cfg.paths.processed, split["train_cities"])
        va = TileDataset(cfg.paths.processed, split["val_cities"])
        te = TileDataset(cfg.paths.processed, split["test_cities"])  # unlabeled for DA
        bs = cfg.optim.batch_size
        src = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)
        tgt = DataLoader(te, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
        val = DataLoader(va, batch_size=bs, num_workers=4)
        model = TerraNetModule(cfg)
        trainer = L.Trainer(
            max_epochs=cfg.optim.max_epochs, precision=cfg.optim.precision,
            accelerator="auto", devices="auto", default_root_dir=str(run_dir),
            log_every_n_steps=20, gradient_clip_val=1.0,
        )
        trainer.fit(model, DAPairLoader(src, tgt), val)
        torch.save(model.state_dict(), run_dir / "model.pt")


if __name__ == "__main__":
    main()
