"""PyTorch datasets over processed tiles.

Processed layout (written by scripts/02_extract_descriptors.py):
  data/processed/<dataset>/<city>/tiles.parquet      one row per tile:
      m, z, descriptor_0..115, gamma, pl0, n_measurements, raster_path
  data/processed/<dataset>/<city>/rasters.zarr       (n_tiles, C, H, W) tile rasters
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

DESCRIPTOR_DIM = 116
RASTER_CHANNELS = 4  # building height AGL, terrain elevation, vegetation mask, water mask


class TileDataset(Dataset):
    """Yields (descriptor[116], raster[C,H,W], target[gamma, pl0], city_idx)."""

    def __init__(self, processed_root: str | Path, cities: list[str],
                 dataset: str = "deepmimo", min_measurements: int = 10):
        self.rows: list[tuple[Path, int, int]] = []  # (city_dir, row_idx, city_idx)
        self.frames: dict[Path, pd.DataFrame] = {}
        self.rasters: dict[Path, zarr.Array] = {}
        self.city_names = cities
        root = Path(processed_root) / dataset
        for ci, city in enumerate(cities):
            cdir = root / city
            df = pd.read_parquet(cdir / "tiles.parquet")
            df = df[df["n_measurements"] >= min_measurements].reset_index(drop=True)
            self.frames[cdir] = df
            self.rasters[cdir] = zarr.open(str(cdir / "rasters.zarr"), mode="r")
            self.rows += [(cdir, i, ci) for i in range(len(df))]

        self._desc_cols = [f"descriptor_{i}" for i in range(DESCRIPTOR_DIM)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        cdir, i, ci = self.rows[idx]
        row = self.frames[cdir].iloc[i]
        desc = torch.tensor(row[self._desc_cols].to_numpy(dtype=np.float32))
        raster = torch.tensor(self.rasters[cdir][int(row["raster_idx"])], dtype=torch.float32)
        target = torch.tensor([row["gamma"], row["pl0"]], dtype=torch.float32)
        return desc, raster, target, ci

    def descriptor_matrix(self) -> np.ndarray:
        return np.concatenate(
            [df[self._desc_cols].to_numpy(np.float32) for df in self.frames.values()], axis=0
        )

    def target_matrix(self) -> np.ndarray:
        return np.concatenate(
            [df[["gamma", "pl0"]].to_numpy(np.float32) for df in self.frames.values()], axis=0
        )
