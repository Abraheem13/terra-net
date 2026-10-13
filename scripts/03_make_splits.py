#!/usr/bin/env python
"""Leave-one-city-out splits: each city is the target once and every other
city is a training (source) city. Hyper-parameters are chosen inside the
training cities by an inner leave-one-training-city-out loop
(terranet.experiments.common.inner_splits), so no city is reserved for
validation and the procedure is symmetric in the source cities.

Outputs data/splits/loco.json and outputs/tables/splits.csv.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from terranet.experiments.common import load_cfg


def loco_splits(cities: list[str]) -> list[dict]:
    return [{"name": f"loco_{test}", "test_cities": [test],
             "train_cities": [c for c in cities if c != test]} for test in cities]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    cfg, base = load_cfg(args.config)
    splits = loco_splits(list(cfg.cities))
    p = Path(base.paths.splits) / "loco.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(splits, indent=2))
    t = Path(base.paths.outputs) / "tables"
    t.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"test": s["test_cities"][0], "n_train_cities": len(s["train_cities"])}
                  for s in splits]).to_csv(t / "splits.csv", index=False)
    print(f"wrote {len(splits)} LOCO splits -> {p}")


if __name__ == "__main__":
    main()
