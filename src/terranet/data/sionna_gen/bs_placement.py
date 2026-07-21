"""Rooftop base-station placement.

Realistic macro-cell siting: BSs sit on tall buildings, spread apart by a
minimum separation so coverage areas differ. Deterministic given a seed, so
scene generation is reproducible.
"""
from __future__ import annotations

import numpy as np
import trimesh


def rooftop_sites(mesh_path, n_bs: int, size_m: float, mast_agl: float,
                  min_sep_m: float = 250.0, seed: int = 0) -> np.ndarray:
    """Return (n, 3) BS positions in local metres, z = rooftop + mast."""
    mesh = trimesh.load(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    rng = np.random.default_rng(seed)

    # candidate rooftops: local maxima of vertex height on a coarse grid
    cell = 25.0
    gx = np.floor((verts[:, 0] + size_m / 2) / cell).astype(int)
    gy = np.floor((verts[:, 1] + size_m / 2) / cell).astype(int)
    key = gx * 100000 + gy
    order = np.lexsort((-verts[:, 2], key))
    _, first = np.unique(key[order], return_index=True)
    cand = verts[order[first]]
    cand = cand[cand[:, 2] > 6.0]
    if len(cand) == 0:
        raise RuntimeError("no rooftop candidates above 6 m")

    cand = cand[np.argsort(-cand[:, 2])]             # tallest first
    chosen: list[np.ndarray] = []
    for p in cand:
        if len(chosen) >= n_bs:
            break
        if all(np.hypot(p[0] - c[0], p[1] - c[1]) >= min_sep_m for c in chosen):
            chosen.append(p)
    # relax separation if the scene is too dense/small to fit n_bs
    if len(chosen) < n_bs:
        rest = [p for p in cand if not any(np.allclose(p, c) for c in chosen)]
        rng.shuffle(rest)
        chosen += rest[: n_bs - len(chosen)]

    out = np.asarray(chosen[:n_bs], float)
    out[:, 2] += mast_agl
    return out
