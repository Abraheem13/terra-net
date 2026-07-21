import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from terranet.data.sionna_gen.bs_placement import rooftop_sites  # noqa: E402


def _fake_city(tmp_path, n=40, size=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    meshes = []
    for _ in range(n):
        h = float(rng.uniform(8, 60))
        b = trimesh.creation.box(extents=[30, 30, h])
        b.apply_translation([rng.uniform(-size/2, size/2),
                             rng.uniform(-size/2, size/2), h / 2])
        meshes.append(b)
    p = tmp_path / "buildings.ply"
    trimesh.util.concatenate(meshes).export(p)
    return p


def test_returns_requested_count_and_height(tmp_path):
    p = _fake_city(tmp_path)
    bs = rooftop_sites(p, n_bs=8, size_m=1000.0, mast_agl=6.0, min_sep_m=100.0)
    assert bs.shape == (8, 3)
    assert (bs[:, 2] > 6.0).all()          # rooftop + mast


def test_separation_respected_when_feasible(tmp_path):
    p = _fake_city(tmp_path, n=60, size=2000.0)
    bs = rooftop_sites(p, n_bs=5, size_m=2000.0, mast_agl=6.0, min_sep_m=300.0)
    d = np.hypot(bs[:, None, 0] - bs[None, :, 0], bs[:, None, 1] - bs[None, :, 1])
    off_diag = d[~np.eye(len(bs), dtype=bool)]
    assert off_diag.min() >= 300.0 - 1e-6


def test_deterministic(tmp_path):
    p = _fake_city(tmp_path)
    a = rooftop_sites(p, 6, 1000.0, 6.0, seed=3)
    b = rooftop_sites(p, 6, 1000.0, 6.0, seed=3)
    assert np.allclose(a, b)
