"""Link features, diffraction baselines and the hurdle calibration."""
import numpy as np
import pytest

pytest.importorskip("rasterio")
from shapely.geometry import box  # noqa: E402

from terranet.data.link_features import (  # noqa: E402
    HeightMap,
    deygout_loss_db,
    free_space_db,
    knife_edge_loss_db,
    link_features,
    uma_tr38901_db,
)
from terranet.experiments.calibration import calibrate, estimate_kappa  # noqa: E402


def _scene(h=30.0):
    # one 40 m wide building of height h centred on the origin, scene 600 m
    parts = np.asarray([box(-20, -200, 20, 200)], dtype=object)
    return HeightMap(parts, np.array([h]), 600.0)


def test_knife_edge_monotone_and_zero_below_threshold():
    nu = np.array([-2.0, -0.78, 0.0, 1.0, 3.0])
    j = knife_edge_loss_db(nu)
    assert j[0] == 0.0 and j[1] == 0.0
    assert np.all(np.diff(j[2:]) > 0) and abs(j[2] - 6.0) < 0.5


def test_obstructed_link_sees_the_building():
    hm = _scene()
    px, py = np.array([200.0]), np.array([0.0])
    f = link_features(hm, px, py, 1.5, np.array([-200.0]), np.array([0.0]), np.array([20.0]),
                      3.5e9)
    blocked, n_cross, max_exc, ke = f[0, 4], f[0, 5], f[0, 7], f[0, 8]
    assert blocked > 0 and n_cross == 1 and 15 < max_exc < 30 and ke > 10
    # a site above the roof is in line of sight
    g = link_features(hm, px, py, 1.5, np.array([-200.0]), np.array([0.0]), np.array([80.0]),
                      3.5e9)
    assert g[0, 4] == 0 and g[0, 8] == 0


def test_deygout_at_least_single_edge_and_zero_in_los():
    hm = _scene()
    px, py = np.array([200.0, 200.0]), np.array([0.0, 150.0])
    sx, sy, sz = np.array([-200.0]), np.array([0.0]), np.array([20.0])
    f = link_features(hm, px, py, 1.5, sx, sy, sz, 3.5e9)
    d = deygout_loss_db(hm, px, py, 1.5, sx, sy, sz, 3.5e9)
    assert d[0] >= f[0, 8] - 1e-3            # principal edge plus non-negative sub-edges
    g = link_features(hm, px, py, 1.5, sx, sy, np.array([80.0]), 3.5e9)
    e = deygout_loss_db(hm, px, py, 1.5, sx, sy, np.array([80.0]), 3.5e9)
    assert g[0, 8] == 0 and e[0] == 0


def test_uma_nlos_never_below_los_and_free_space_reference():
    d3 = np.array([100.0, 500.0, 2000.0])
    d2 = np.sqrt(d3 ** 2 - 25 ** 2)
    los = uma_tr38901_db(d3, d2, 26.5, 3.5e9, np.array([True] * 3))
    nlos = uma_tr38901_db(d3, d2, 26.5, 3.5e9, np.array([False] * 3))
    assert np.all(nlos >= los) and np.all(np.diff(los) > 0)
    assert abs(free_space_db(np.array([100.0]), 3.5e9)[0] - 83.3) < 0.2


def test_kappa_large_when_sites_agree_and_small_when_they_differ():
    rng = np.random.default_rng(0)
    site = rng.integers(0, 10, 4000)
    same = rng.normal(0, 3, 4000)
    assert estimate_kappa(same, site, 10) > 100
    shifted = same + rng.normal(0, 5, 10)[site]
    assert estimate_kappa(shifted, site, 10) < 5


def test_calibrate_never_copies_truth_and_removes_site_offsets():
    rng = np.random.default_rng(1)
    P, S = 2000, 6

    class C:
        pass
    c = C()
    c.tile = rng.integers(0, 40, P)
    off = rng.normal(0, 6, S)
    c.true_pl = 100 + rng.normal(0, 2, (P, S)) + off[None, :]
    PLr = np.full((P, S), 100.0)
    keep = np.ones((P, S), bool)
    M, R, kappa = calibrate(c, PLr, keep, np.arange(20))
    surveyed = np.isin(c.tile, np.arange(20))
    assert not np.allclose(M[surveyed], c.true_pl[surveyed])       # no truth copied in
    err_before = np.sqrt(((PLr - c.true_pl)[~surveyed] ** 2).mean())
    err_after = np.sqrt(((R - c.true_pl)[~surveyed] ** 2).mean())
    assert err_after < 0.6 * err_before and kappa < 50
