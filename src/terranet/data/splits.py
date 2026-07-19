"""Split protocols.

- LOCO: leave-one-city-out (primary protocol).
- Geographic within-city blocks for val (anti spatial-leakage; random splits over
  spatially autocorrelated measurements inflate performance).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Split:
    name: str
    train_cities: list[str]
    val_cities: list[str]
    test_cities: list[str]

    def to_json(self) -> dict:
        return self.__dict__


def loco_splits(cities: list[str], n_val: int = 1, seed: int = 0) -> list[Split]:
    """One split per held-out test city; `n_val` of the remaining cities as val."""
    rng = np.random.default_rng(seed)
    splits = []
    for test_city in cities:
        rest = [c for c in cities if c != test_city]
        val = list(rng.choice(rest, size=n_val, replace=False))
        train = [c for c in rest if c not in val]
        splits.append(Split(f"loco_{test_city}", train, val, [test_city]))
    return splits


def geographic_block_split(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon",
    n_blocks: int = 5, val_block: int = 0, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Checkerboard block split within a city: measurements are grouped into
    n_blocks x n_blocks spatial blocks; whole blocks go to val, never points."""
    qlat = pd.qcut(df[lat_col], n_blocks, labels=False, duplicates="drop")
    qlon = pd.qcut(df[lon_col], n_blocks, labels=False, duplicates="drop")
    block = (qlat * n_blocks + qlon).astype(int)
    rng = np.random.default_rng(seed)
    val_blocks = rng.choice(np.unique(block), size=max(1, len(np.unique(block)) // 5), replace=False)
    mask = block.isin(val_blocks)
    return df[~mask].copy(), df[mask].copy()


def save_splits(splits: list[Split], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([s.to_json() for s in splits], indent=2))
