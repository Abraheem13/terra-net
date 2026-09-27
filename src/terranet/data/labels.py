"""Per-tile log-distance labels (gamma, PL0) with empirical-Bayes shrinkage.

Model inside tile t, for link i (receiver in t, served by base station b_i):

    PL_i = PL0_t + gamma_t * x_i + e_i,     x_i = 10 log10(d_i / d0)

with d_i the 3-D transmitter-receiver distance and d0 the reference
distance (default 100 m, the tile scale; see `D0_M`).

Estimator (random-effects / James-Stein form).  With x centred inside the
tile, the OLS slope is b_t = Sxy_t / Sxx_t. Its sampling variance v_t is the
cluster-robust variance with the bias-reduced linearisation of Bell and
McCaffrey (CR2) and the serving sites as clusters (links served
by the same site share propagation paths, so the iid value s_t^2 / Sxx_t
understates it); it is never taken below the iid value, and tiles served by a
single site use the iid value times the city's median design effect. Slopes
are shrunk toward the precision-weighted city mean mu with between-tile
variance tau^2 estimated by DerSimonian-Laird:

    gamma_t = (1 - w_t) b_t + w_t mu,      w_t = v_t / (v_t + tau^2)

which is exactly ridge on the centred slope with lambda_t = Sxx_t v_t / tau^2,
i.e. the per-tile risk-optimal penalty of the manuscript.  The intercept is
not penalised: PL0_t = ybar_t - gamma_t xbar_t.

Standard errors are reported two ways for every tile:
  * analytic (iid residuals)
  * cluster-robust (CR2), clusters = serving base stations
Both treat (mu, tau^2) as fixed.  Everything is vectorised with bincount,
so the full 24M-link corpus is processed in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

D0_M = 100.0
EARTH_RADIUS_M = 6_371_000.0


def link_distance_m(meas: pd.DataFrame) -> np.ndarray:
    """3-D Tx-Rx distance in metres.

    Uses the solver-provided `dist_m` when present (Sionna, LWM); otherwise
    horizontal great-circle distance combined with the antenna heights.
    """
    if "dist_m" in meas.columns and meas["dist_m"].notna().all():
        return meas["dist_m"].to_numpy(float)
    la1, lo1 = np.radians(meas.tx_lat.to_numpy(float)), np.radians(meas.tx_lon.to_numpy(float))
    la2, lo2 = np.radians(meas.rx_lat.to_numpy(float)), np.radians(meas.rx_lon.to_numpy(float))
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    horiz = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    dh = meas.tx_h.to_numpy(float) - meas.rx_h.to_numpy(float)
    return np.sqrt(horiz ** 2 + dh ** 2)


def free_space_db(d_m: np.ndarray, freq_ghz: float) -> np.ndarray:
    """Free-space path loss (Friis, isotropic antennas) in dB."""
    return 20 * np.log10(np.maximum(d_m, 1e-3)) + 20 * np.log10(freq_ghz * 1e9) - 147.55


def load_links(path, cfg) -> tuple[pd.DataFrame, int]:
    """Ray-traced links of one city after the physical-consistency check.

    A link whose path loss lies more than `cfg.qc_fspl_margin_db` below free
    space at its 3-D distance cannot arise from an incoherent sum of a few
    propagation paths; such values are rare heavy-tailed outliers of the
    Monte Carlo radio-map estimator (diffracted-ray weights) and are removed.
    Without the key the links are returned unchanged. Returns (links, n_removed).
    """
    meas = pd.read_parquet(path)
    margin = cfg.get("qc_fspl_margin_db", None) if hasattr(cfg, "get") else None
    if margin is None:
        return meas, 0
    fs = free_space_db(link_distance_m(meas), float(cfg.freq_ghz))
    keep = meas.pathloss_db.to_numpy(float) >= fs - float(margin)
    return meas[keep].reset_index(drop=True), int((~keep).sum())


def log_distance(d_m: np.ndarray, d0: float = D0_M) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(d_m, 1.0) / d0)


@dataclass
class TileFit:
    gamma: np.ndarray
    pl0: np.ndarray
    n: np.ndarray
    n_bs: np.ndarray
    shrink: np.ndarray          # w_t in [0, 1]
    gamma_ols: np.ndarray
    fit_rmse_db: np.ndarray
    gamma_se: np.ndarray        # analytic, iid
    pl0_se: np.ndarray
    gamma_se_cl: np.ndarray     # cluster-robust CR2 over base stations
    pl0_se_cl: np.ndarray
    mu: float
    tau2: float
    design_effect: float = 1.0      # median v_cl / v_iid of the OLS slope

    def frame(self) -> pd.DataFrame:
        cols = ["gamma", "pl0", "n", "n_bs", "shrink", "gamma_ols", "fit_rmse_db",
                "gamma_se", "pl0_se", "gamma_se_cl", "pl0_se_cl"]
        return pd.DataFrame({c: getattr(self, c) for c in cols})


def dersimonian_laird(b: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Random-effects mean and between-unit variance (DerSimonian & Laird 1986)."""
    w = 1.0 / v
    mu_fe = float((w * b).sum() / w.sum())
    q = float((w * (b - mu_fe) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (len(b) - 1)) / c) if c > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float((ws * b).sum() / ws.sum())
    return mu, tau2


def cr2_cov(t: np.ndarray, key: np.ndarray, xc: np.ndarray, e: np.ndarray,
            n: np.ndarray, sxx_pen: np.ndarray, n_tiles: int, n_b: int):
    """CR2 covariance of the centred per-tile coefficients (a_t, g_t).

    Per tile the design is X = [1, x - xbar_t] with bread M = diag(1/n, 1/sxx_pen)
    (sxx_pen = Sxx for OLS, Sxx + lambda for the shrunk slope).  CR2 replaces the
    residual vector e_g of cluster g by (I - H_gg)^(-1/2) e_g with
    H_gg = X_g M X_g'.  Using X_g' f(X_g M X_g') = f(G_g M) X_g' with the 2 x 2
    Gram G_g = X_g' X_g, the adjusted score is
        u_g = M^(-1/2) f(S_g) M^(1/2) X_g' e_g,   S_g = M^(1/2) G_g M^(1/2),
    f(s) = (1 - s)^(-1/2) (Moore-Penrose: 0 on eigenvalues equal to 1), so only
    2 x 2 eigen-decompositions are needed.  Returns var_a, var_g, cov_ag.
    """
    m = n_tiles * n_b

    def c(v):
        return np.bincount(key, weights=v, minlength=m)

    ng_, s1, s2 = c(np.ones_like(xc)), c(xc), c(xc * xc)
    z = np.stack([c(e), c(xc * e)], -1)                                   # (m, 2)
    tt = np.repeat(np.arange(n_tiles), n_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.stack([1 / np.sqrt(n[tt]), 1 / np.sqrt(sxx_pen[tt])], -1)  # M^(1/2)
    h = np.where(np.isfinite(h), h, 0.0)
    G = np.stack([np.stack([ng_, s1], -1), np.stack([s1, s2], -1)], -2)  # (m, 2, 2)
    S = h[:, :, None] * G * h[:, None, :]
    lam, W = np.linalg.eigh(S)
    one_m = 1.0 - lam
    fl = np.where(one_m > 1e-10, 1.0 / np.sqrt(np.maximum(one_m, 1e-10)), 0.0)
    fS = np.einsum("mij,mj,mkj->mik", W, fl, W)
    v = np.einsum("mij,mj->mi", fS, h * z)                                # f(S) M^(1/2) z
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(h > 0, v / h, 0.0)                                   # M^(-1/2) ...
    u = u.reshape(n_tiles, n_b, 2)
    meat = np.einsum("tbi,tbj->tij", u, u)
    Mdiag = np.where(h.reshape(n_tiles, n_b, 2)[:, 0, :] > 0,
                     h.reshape(n_tiles, n_b, 2)[:, 0, :] ** 2, np.nan)
    cov = Mdiag[:, :, None] * meat * Mdiag[:, None, :]
    return cov[:, 0, 0], cov[:, 1, 1], cov[:, 0, 1]


def fit_tiles(tile_idx: np.ndarray, x: np.ndarray, y: np.ndarray, bs: np.ndarray,
              n_tiles: int, min_count: int = 30,
              mu: float | None = None, tau2: float | None = None) -> TileFit:
    """Fit every tile at once.

    tile_idx : int array, -1 for links outside every tile.
    x        : 10 log10(d / d0).
    bs       : serving base-station label per link (any hashable dtype).
    mu, tau2 : optional fixed prior (else estimated from this data).
    """
    ok = tile_idx >= 0
    t, x, y = tile_idx[ok].astype(np.int64), x[ok], y[ok]
    _, b_code = np.unique(np.asarray(bs)[ok], return_inverse=True)
    n_b = int(b_code.max()) + 1 if b_code.size else 0

    def s(v):
        return np.bincount(t, weights=v, minlength=n_tiles)

    n = np.bincount(t, minlength=n_tiles).astype(float)
    sx, sy = s(x), s(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar, ybar = sx / n, sy / n
        sxx = s(x * x) - n * xbar ** 2
        sxy = s(x * y) - n * xbar * ybar
        syy = s(y * y) - n * ybar ** 2
        b_ols = sxy / sxx
        rss_ols = np.maximum(syy - b_ols * sxy, 0.0)
        s2 = rss_ols / np.maximum(n - 2, 1)
        v = s2 / sxx

    key = t * n_b + b_code

    ng = (np.bincount(key, minlength=n_tiles * n_b).reshape(n_tiles, n_b) > 0).sum(1)
    xc = x - xbar[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        # cluster-robust (CR2) sampling variance of the OLS slope
        e_ols = y - ybar[t] - b_ols[t] * xc
        _, v_cl, _ = cr2_cov(t, key, xc, e_ols, n, sxx, n_tiles, n_b)
        v_cl = np.where(ng > 1, v_cl, np.nan)

    usable = (n >= min_count) & (sxx > 1e-9) & np.isfinite(v) & (v > 0)
    # sampling variance used for shrinkage: cluster-robust where >= 2 sites
    # serve the tile (never below the iid value); single-site tiles get the
    # iid variance inflated by the median design effect of the city.
    clu = usable & (ng >= 2) & np.isfinite(v_cl)
    deff = float(np.median(np.maximum(v_cl[clu] / v[clu], 1.0))) if clu.any() else 1.0
    v_used = np.where(clu, np.maximum(v_cl, v), v * deff)
    if mu is None or tau2 is None:
        mu, tau2 = dersimonian_laird(b_ols[usable], v_used[usable])
    tau2 = max(float(tau2), 1e-12)

    fitted = n >= min_count
    w = np.where(usable, v_used / (v_used + tau2), 1.0)     # degenerate span -> prior
    b = np.where(usable, b_ols, mu)
    gamma = np.where(fitted, (1 - w) * b + w * mu, np.nan)
    pl0 = np.where(fitted, ybar - gamma * xbar, np.nan)
    # equivalent ridge penalty on the centred slope: w = lam / (lam + Sxx)
    lam = np.where(usable, sxx * v_used / tau2, np.inf)

    # residuals of the shrunk fit
    e = y - pl0[t] - gamma[t] * x
    rmse = np.sqrt(s(e * e) / np.maximum(n, 1))

    # influence of each link on (gamma, pl0): linear statistics sum_i g_i e_i
    denom = np.where(usable, sxx + np.where(np.isfinite(lam), lam, 0.0), np.inf)
    # analytic (iid)
    var_g = np.where(usable, s2 * sxx / denom ** 2, 0.0)
    s2_used = np.where(usable, s2, s(e * e) / np.maximum(n - 1, 1))
    var_p = s2_used / np.maximum(n, 1) + xbar ** 2 * var_g
    # cluster-robust (CR2) over (tile, site), residuals and bread of the shrunk
    # fit; pl0 = a - gamma * xbar in the centred parametrisation
    with np.errstate(invalid="ignore", divide="ignore"):
        va, vg, cag = cr2_cov(t, key, xc, e, n, denom, n_tiles, n_b)
    var_g_cl = np.where(ng > 1, vg, np.nan)
    var_p_cl = np.where(ng > 1, va - 2 * xbar * cag + xbar ** 2 * vg, np.nan)

    nan_unless = lambda a: np.where(fitted, a, np.nan)  # noqa: E731
    return TileFit(
        gamma=gamma, pl0=pl0, n=n, n_bs=ng.astype(float),
        shrink=nan_unless(w), gamma_ols=nan_unless(np.where(usable, b_ols, np.nan)),
        fit_rmse_db=nan_unless(rmse),
        gamma_se=nan_unless(np.sqrt(var_g)), pl0_se=nan_unless(np.sqrt(var_p)),
        gamma_se_cl=nan_unless(np.sqrt(var_g_cl)), pl0_se_cl=nan_unless(np.sqrt(var_p_cl)),
        mu=float(mu), tau2=float(tau2), design_effect=deff,
    )


def fit_global(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """City-wide OLS (gamma, PL0) on the same x convention."""
    A = np.c_[x, np.ones_like(x)]
    (g, p), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(g), float(p)
