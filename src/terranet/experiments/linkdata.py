"""Link-level data shared by the network-level scripts: the City object with
its (pixel, site) feature matrix (cached), and the hurdle link model's
predicted maps saved per fold and seed so that later scripts (receiver-tile
variants, external tests) reuse exactly the same hurdle decisions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS
from terranet.data.link_features import LINK_FEATURES, city_link_matrix
from terranet.evaluation.network import load_city

FEATURES = LINK_FEATURES + DESCRIPTOR_COLUMNS
LOS_COL = LINK_FEATURES.index("blocked_frac")      # 0 -> line of sight
KE_COL = LINK_FEATURES.index("ke_loss_db")


def city_data(cfg, base, name, cache_dir=None):
    """City object plus the (P*S, F) feature matrix, cached in outputs/cache."""
    cache_dir = Path(base.paths.outputs) / "cache" if cache_dir is None else Path(cache_dir)
    city = load_city(cfg, base, name)
    f = cache_dir / f"link_{name}.npy"
    X = np.load(f) if f.exists() else None
    if X is None or X.shape[0] != city.true_pl.size:
        tiles = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / name / "tiles.parquet")
        X = city_link_matrix(city, Path(base.paths.raw) / cfg.scenes_dir / name,
                             tiles[DESCRIPTOR_COLUMNS].to_numpy(np.float32),
                             float(cfg.freq_ghz) * 1e9)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(f, X)
    return city, X


def hurdle_path(base, fold, seed) -> Path:
    return Path(base.paths.outputs) / "predictions" / "hurdle" / f"{fold}_{seed}.npz"


def save_hurdle(base, fold, seed, PLr, keep):
    p = hurdle_path(base, fold, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, plr=PLr.astype(np.float32), keep=keep.astype(bool))


def load_hurdle(base, fold, seed):
    """(regression map, hurdle decision) of the link model of (fold, seed)."""
    z = np.load(hurdle_path(base, fold, seed))
    return z["plr"].astype(float), z["keep"]
