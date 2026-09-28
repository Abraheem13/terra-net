import numpy as np
import scipy.sparse as sp

from terranet.evaluation import mixed_path as mp


def _pairs(rng, n=4000, T=6):
    """Random pairs whose straight paths cross one to three tiles."""
    rows, cols, vals = [], [], []
    for k in range(n):
        tiles = rng.choice(T, size=rng.integers(1, 4), replace=False)
        for t in tiles:
            rows.append(k)
            cols.append(t)
            vals.append(rng.uniform(20, 150))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, T))


def test_average_reduces_to_log_distance_in_one_tile():
    L = sp.csr_matrix(([80.0, 120.0], ([0, 1], [2, 2])), shape=(2, 4))
    x = np.array([3.0, 7.0])
    A, G = np.arange(4.0) + 100, np.arange(4.0) + 2
    np.testing.assert_allclose(mp.predict("average", L, x, A, G), A[2] + G[2] * x)


def test_sum_adds_one_loss_per_crossed_tile():
    L = sp.csr_matrix(([10.0, 100.0], ([0, 0], [0, 1])), shape=(1, 2))
    A, G = np.array([5.0, 7.0]), np.array([2.0, 3.0])
    want = (5 + 10 * 2 * 1) + (7 + 10 * 3 * 2)
    np.testing.assert_allclose(mp.predict("sum", L, np.zeros(1), A, G), [want])


def test_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    L = _pairs(rng)
    x = rng.uniform(0, 12, L.shape[0])
    A, G = rng.uniform(90, 110, 6), rng.uniform(2, 4, 6)
    for model in mp.MODELS:
        y = mp.predict(model, L, x, A, G)
        Ah, Gh, n = mp.fit(model, L, x, y, ridge=1e-6)
        np.testing.assert_allclose(mp.predict(model, L, x, Ah, Gh), y, atol=1e-3)
        assert (n > 0).all()
