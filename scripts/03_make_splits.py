#!/usr/bin/env python
"""Leave-one-city-out splits: each city is the target once; one of the
remaining cities (drawn with a fixed seed) is the validation city used only for
hyper-parameter selection; the rest are training (source) cities.

Outputs data/splits/loco.json and outputs/tables/splits.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.experiments.common import load_cfg


def loco_splits(cities: list[str], seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for test in cities:
        rest = [c for c in cities if c != test]
        val = str(rng.choice(rest))
        out.append({"name": f"loco_{test}", "test_cities": [test], "val_cities": [val],
                    "train_cities": [c for c in rest if c != val]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    splits = loco_splits(list(cfg.cities), args.seed)
    p = Path(base.paths.splits) / "loco.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(splits, indent=2))
    t = Path(base.paths.outputs) / "tables"
    t.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"test": s["test_cities"][0], "val": s["val_cities"][0],
                   "n_train_cities": len(s["train_cities"])} for s in splits]
                 ).to_csv(t / "splits.csv", index=False)
    print(f"wrote {len(splits)} LOCO splits -> {p}")


if __name__ == "__main__":
    main()
