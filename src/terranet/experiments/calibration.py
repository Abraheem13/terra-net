"""Per-site calibration of a link model from surveyed tiles, and survey design.

The links observed in the surveyed tiles carry their serving site, as
minimisation-of-drive-tests (MDT) reports do. With independent site offsets
of prior variance tau^2 and link noise sigma^2 (kappa = sigma^2 / tau^2), the
posterior mean offset of site s is (n_s rbar_s + kappa rbar) / (n_s + kappa)
and the reduction of its posterior variance is tau^2 n_s / (n_s + kappa).

kappa is estimated from the surveyed links themselves by the DerSimonian-Laird
moment estimator over the site means (empirical Bayes): when the site means
scatter no more than their sampling error, tau^2 -> 0, kappa -> inf and the
calibration reduces to the overall offset; a fixed kappa can be passed for
the ablation. The surveyed pixels are never replaced by their true values:
the calibrated map is a model everywhere, and the evaluation restricts the
metrics to the unsurveyed pixels (`network.unsurveyed_mask`).

Two designs choose the surveyed tiles before any measurement: the greedy
maximisation of the variance reduction F(T) (monotone submodular), and a
stratified variant that first partitions the tiles into k descriptor
strata (k-means) and then takes, in each stratum, the tile with the largest
marginal gain of F, so that the survey stays representative of the city.
"""
from __future__ import annotations

import numpy as np

from terranet.data.labels import dersimonian_laird
from terranet.experiments.common import LINK_KAPPA

KAPPA_MAX = 1e6


def estimate_kappa(res, site, n_sites, kappa_min=1.0, kappa_max=KAPPA_MAX):
    """Empirical-Bayes kappa = sigma^2 / tau^2 from residuals `res` of links
    served by `site`: sigma^2 is the pooled within-site variance and tau^2 the
    DerSimonian-Laird between-site variance of the site means."""
    n = np.bincount(site, minlength=n_sites)
    ok = n >= 2
    if ok.sum() < 2:
        return float(kappa_max)
    mean = np.bincount(site, weights=res, minlength=n_sites) / np.maximum(n, 1)
    ss = np.bincount(site, weights=(res - mean[site]) ** 2, minlength=n_sites)
    sigma2 = float(ss[ok].sum() / max((n[ok] - 1).sum(), 1))
    if sigma2 <= 0:
        return float(kappa_max)
    _, tau2 = dersimonian_laird(mean[ok], sigma2 / n[ok])
    if tau2 <= 1e-12:
        return float(kappa_max)
    return float(np.clip(sigma2 / tau2, kappa_min, kappa_max))


def site_counts(tile_of_pixel, tile_ids, pred_obs):
    """(n_tiles, S) predicted number of received links of each site per tile."""
    S = pred_obs.shape[1]
    counts = np.zeros((len(tile_ids), S))
    pos = {t: i for i, t in enumerate(tile_ids)}
    rows = np.array([pos.get(t, -1) for t in tile_of_pixel])
    ok = rows >= 0
    np.add.at(counts, rows[ok], pred_obs[ok].astype(float))
    return counts


def design_survey(tile_of_pixel, tile_ids, pred_obs, k, kappa=LINK_KAPPA):
    """Greedy maximisation of F(T) = sum_s n_s(T) / (n_s(T) + kappa)."""
    counts = site_counts(tile_of_pixel, tile_ids, pred_obs)
    n = np.zeros(counts.shape[1])
    chosen = []
    for _ in range(min(k, len(tile_ids))):
        f0 = (n / (n + kappa)).sum()
        gain = ((n + counts) / (n + counts + kappa)).sum(1) - f0
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        n += counts[j]
    return np.array(chosen)


def design_survey_stratified(tile_of_pixel, tile_ids, pred_obs, k, X, kappa=LINK_KAPPA,
                             seed=0):
    """Stratified design: k-means strata of the normalised descriptors X (one
    row per tile in `tile_ids`), one tile per stratum, chosen by the largest
    marginal gain of F(T) in the order of decreasing stratum size."""
    from sklearn.cluster import KMeans
    counts = site_counts(tile_of_pixel, tile_ids, pred_obs)
    k = min(k, len(tile_ids))
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    order = np.argsort(-np.bincount(lab, minlength=k))
    n = np.zeros(counts.shape[1])
    chosen = []
    for c in order:
        members = np.flatnonzero(lab == c)
        f0 = (n / (n + kappa)).sum()
        gain = ((n + counts[members]) / (n + counts[members] + kappa)).sum(1) - f0
        j = int(members[np.argmax(gain)])
        chosen.append(j)
        n += counts[j]
    return np.array(chosen)


def calibrate(city, PLr, keep, surveyed_tiles, kappa=None):
    """Per-site offsets from the links observed in the surveyed tiles.

    PLr: regression map; keep: hurdle decision (predicted below the cut-off);
    kappa: fixed prior weight in links, or None for the empirical-Bayes
    estimate. Returns (network map, calibrated regression map, kappa used).
    Nothing is copied from the truth: the surveyed pixels keep their model
    prediction and are excluded from the evaluation by the caller."""
    P, S = PLr.shape
    surveyed = np.isin(city.tile, surveyed_tiles)
    obs = np.isfinite(city.true_pl) & surveyed[:, None]
    res = (city.true_pl - PLr)[obs]
    site = np.broadcast_to(np.arange(S), (P, S))[obs]
    if kappa is None:
        kappa = estimate_kappa(res, site, S) if res.size else KAPPA_MAX
    rbar = res.mean() if res.size else 0.0
    n_s = np.bincount(site, minlength=S)
    sum_s = np.bincount(site, weights=res, minlength=S)
    off = (sum_s + kappa * rbar) / (n_s + kappa)
    R = PLr + off[None, :]
    return np.where(keep, R, np.inf), R, float(kappa)
