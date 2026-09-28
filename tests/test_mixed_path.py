import numpy as np
import scipy.sparse as sp

from terranet.evaluation import mixed_path as mp


def _paths(L, Q=None, first=None):
    L = sp.csr_matrix(L)
    Q = sp.csr_matrix(L.shape) if Q is None else sp.csr_matrix(Q)
    first = np.zeros(L.shape[0], int) if first is None else np.asarray(first)
    return mp.Paths(L, Q, first)


def _random_paths(rng, n=4000, T=6):
    """Random pairs whose straight paths cross one to three tiles."""
    L, Q, first = np.zeros((n, T)), np.zeros((n, T)), np.zeros(n, int)
    for k in range(n):
        tiles = rng.choice(T, size=rng.integers(1, 4), replace=False)
        L[k, tiles] = rng.uniform(20, 150, len(tiles))
        Q[k, tiles] = rng.uniform(0, 3, len(tiles))
        first[k] = tiles[0]
    return _paths(L, Q, first)


def test_average_reduces_to_log_distance_in_one_tile():
    L = np.zeros((2, 4))
    L[0, 2], L[1, 2] = 80.0, 120.0
    x = np.array([3.0, 7.0])
    A, G = np.arange(4.0) + 100, np.arange(4.0) + 2
    np.testing.assert_allclose(mp.predict("average", _paths(L), x, A, G), A[2] + G[2] * x)


def test_sum_adds_one_loss_per_crossed_tile():
    L = np.array([[10.0, 100.0]])
    A, G = np.array([5.0, 7.0]), np.array([2.0, 3.0])
    want = (5 + 10 * 2 * 1) + (7 + 10 * 3 * 2)
    np.testing.assert_allclose(mp.predict("sum", _paths(L), np.zeros(1), A, G), [want])


def test_piecewise_is_multi_slope():
    # site in tile 0; tile 0 covers 0-200 m, tile 1 covers 200-800 m; d0 = 100 m
    L = np.array([[200.0, 600.0]])
    Q = np.array([[10 * np.log10(200 / 100), 10 * np.log10(800 / 200)]])
    A, G = np.array([90.0, 50.0]), np.array([2.0, 4.0])
    want = 90 + 2 * 10 * np.log10(2) + 4 * 10 * np.log10(4)
    got = mp.predict("piecewise", _paths(L, Q, [0]), np.zeros(1), A, G)
    np.testing.assert_allclose(got, [want])


def test_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    paths = _random_paths(rng)
    x = rng.uniform(0, 12, paths.shape[0])
    A, G = rng.uniform(90, 110, 6), rng.uniform(2, 4, 6)
    for model in mp.MODELS:
        y = mp.predict(model, paths, x, A, G)
        Ah, Gh, n = mp.fit(model, paths, x, y, ridge=1e-6)
        np.testing.assert_allclose(mp.predict(model, paths, x, Ah, Gh), y, atol=1e-3)
        assert (n > 0).all()
