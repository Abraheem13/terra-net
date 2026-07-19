import numpy as np

from terranet.models.baselines.log_distance import (
    fit_global, fit_tiles_mixed_path, predict,
)


def test_global_fit_recovers_params():
    rng = np.random.default_rng(0)
    d = rng.uniform(10, 500, 2000)
    pl = predict(d, gamma=2.8, pl0=30.0) + rng.normal(0, 0.5, d.size)
    g, p = fit_global(d, pl)
    assert abs(g - 2.8) < 0.05 and abs(p - 30.0) < 1.0


def test_mixed_path_two_tiles():
    rng = np.random.default_rng(1)
    true_g, true_p = np.array([2.0, 3.5]), np.array([20.0, 35.0])
    segments, pl = [], []
    for _ in range(3000):
        l0, l1 = rng.uniform(20, 200), rng.uniform(20, 200)
        total = l0 + l1
        val = (true_p[0] * l0 / total + 10 * true_g[0] * np.log10(l0)
               + true_p[1] * l1 / total + 10 * true_g[1] * np.log10(l1))
        segments.append([(0, l0), (1, l1)])
        pl.append(val + rng.normal(0, 0.1))
    g, p, counts = fit_tiles_mixed_path(segments, np.array(pl), 2, damp=1e-6)
    assert np.allclose(g, true_g, atol=0.1)
    assert np.allclose(p, true_p, atol=2.0)
    assert counts.tolist() == [3000, 3000]
