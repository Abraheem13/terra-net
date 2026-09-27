import numpy as np

from terranet.data.labels import dersimonian_laird, fit_tiles, log_distance


def _sim(rng, n_tiles=200, per=120, n_bs=10, rho_bs=0.0, sd=6.0):
    true_g = rng.normal(3.5, 0.4, n_tiles)
    true_p = rng.normal(80.0, 3.0, n_tiles)
    t = np.repeat(np.arange(n_tiles), per)
    bs = rng.integers(0, n_bs, t.size)
    d = rng.uniform(80, 900, t.size)
    x = log_distance(d)
    shock = rng.normal(0, sd * np.sqrt(rho_bs), (n_tiles, n_bs))[t, bs]
    # base-station-specific slope: the serving-geometry effect
    shock += rng.normal(0, 3.0 * rho_bs, (n_tiles, n_bs))[t, bs] * (x - x.mean())
    y = true_p[t] + true_g[t] * x + shock + rng.normal(0, sd * np.sqrt(1 - rho_bs), t.size)
    return t, x, y, bs, true_g, true_p


def test_shrinkage_beats_ols_and_recovers_tau():
    rng = np.random.default_rng(0)
    t, x, y, bs, g, _ = _sim(rng)
    f = fit_tiles(t, x, y, bs, 200, min_count=30)
    assert abs(np.sqrt(f.tau2) - 0.4) < 0.1
    mse_eb = np.mean((f.gamma - g) ** 2)
    mse_ols = np.mean((f.gamma_ols - g) ** 2)
    assert mse_eb < mse_ols
    assert np.all((f.shrink >= 0) & (f.shrink <= 1))


def test_convex_form_exact():
    rng = np.random.default_rng(1)
    t, x, y, bs, _, _ = _sim(rng, n_tiles=50)
    f = fit_tiles(t, x, y, bs, 50)
    lhs = f.gamma
    rhs = (1 - f.shrink) * f.gamma_ols + f.shrink * f.mu
    assert np.allclose(lhs, rhs)


def test_analytic_se_matches_monte_carlo_iid():
    rng = np.random.default_rng(2)
    t, x, y, bs, g, p = _sim(rng, n_tiles=1, per=200)
    f0 = fit_tiles(t, x, y, bs, 1, mu=3.5, tau2=0.16)
    est = []
    for _ in range(600):
        yy = p[0] + g[0] * x + rng.normal(0, 6.0, x.size)
        est.append(fit_tiles(t, x, yy, bs, 1, mu=3.5, tau2=0.16).gamma[0])
    assert abs(np.std(est) / f0.gamma_se[0] - 1) < 0.15


def test_cluster_se_detects_correlation():
    rng = np.random.default_rng(3)
    t, x, y, bs, _, _ = _sim(rng, n_tiles=300, per=150, rho_bs=0.8)
    f = fit_tiles(t, x, y, bs, 300)
    assert np.nanmedian(f.gamma_se_cl / f.gamma_se) > 2.0


def test_degenerate_tile_falls_back_to_prior():
    t = np.zeros(40, int)
    x = np.full(40, 5.0)
    y = np.full(40, 100.0)
    f = fit_tiles(t, x, y, np.zeros(40), 1, mu=3.0, tau2=0.1)
    assert f.gamma[0] == 3.0 and f.shrink[0] == 1.0
    assert np.isclose(f.pl0[0], 100.0 - 3.0 * 5.0)


def test_dersimonian_laird_zero_heterogeneity():
    rng = np.random.default_rng(4)
    v = np.full(500, 0.05)
    b = rng.normal(2.0, np.sqrt(v))
    mu, tau2 = dersimonian_laird(b, v)
    assert abs(mu - 2.0) < 0.05 and tau2 < 0.01


def test_cluster_variance_drives_shrinkage_under_site_correlation():
    rng = np.random.default_rng(5)
    t, x, y, bs, g, _ = _sim(rng, n_tiles=300, per=150, rho_bs=0.8)
    f = fit_tiles(t, x, y, bs, 300)
    assert f.design_effect > 2.0
    assert np.mean((f.gamma - g) ** 2) < 0.8 * np.mean((f.gamma_ols - g) ** 2)


def test_cr2_matches_direct_computation():
    """Vectorised CR2 equals the textbook matrix formula on small tiles."""
    from terranet.data.labels import cr2_cov
    rng = np.random.default_rng(3)
    n_tiles, n_b = 3, 4
    t = rng.integers(0, n_tiles, 400)
    b = rng.integers(0, n_b, 400)
    x = rng.normal(size=400)
    e0 = rng.normal(size=400) + rng.normal(size=n_b)[b]
    n = np.bincount(t, minlength=n_tiles).astype(float)
    xbar = np.bincount(t, weights=x, minlength=n_tiles) / n
    xc = x - xbar[t]
    sxx = np.bincount(t, weights=xc * xc, minlength=n_tiles)
    lam = np.array([0.0, 5.0, 50.0])
    va, vg, cag = cr2_cov(t, t * n_b + b, xc, e0, n, sxx + lam, n_tiles, n_b)
    for k in range(n_tiles):
        m = t == k
        X = np.c_[np.ones(m.sum()), xc[m]]
        M = np.diag([1 / n[k], 1 / (sxx[k] + lam[k])])
        meat = np.zeros((2, 2))
        for g in range(n_b):
            mg = b[m] == g
            if not mg.any():
                continue
            Xg = X[mg]
            H = Xg @ M @ Xg.T
            w, V = np.linalg.eigh(np.eye(mg.sum()) - H)
            A = V @ np.diag(np.where(w > 1e-10, 1 / np.sqrt(np.maximum(w, 1e-10)), 0)) @ V.T
            u = Xg.T @ A @ e0[m][mg]
            meat += np.outer(u, u)
        C = M @ meat @ M
        assert np.allclose([va[k], vg[k], cag[k]], [C[0, 0], C[1, 1], C[0, 1]], rtol=1e-8)
