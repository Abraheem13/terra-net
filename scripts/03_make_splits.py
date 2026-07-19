#!/usr/bin/env python
"""Phase 1: write LOCO splits to data/splits/loco.json."""
import argparse

from omegaconf import OmegaConf

from terranet.data.splits import loco_splits, save_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="loco")
    ap.add_argument("--config", default="configs/data/deepmimo.yaml")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    assert args.protocol == "loco"
    splits = loco_splits(list(cfg.cities), n_val=1, seed=0)
    save_splits(splits, "data/splits/loco.json")
    print(f"wrote {len(splits)} LOCO splits")


if __name__ == "__main__":
    main()
