"""Per-site calibration of a link model from surveyed tiles, and survey design.

The links observed in the surveyed tiles carry their serving site, as
minimisation-of-drive-tests (MDT) reports do. With independent site offsets
of prior variance tau^2 and link noise sigma^2 (kappa = sigma^2 / tau^2), the
posterior mean offset of site s is (n_s rbar_s + kappa rbar) / (n_s + kappa)
and the reduction of its posterior variance is tau^2 n_s / (n_s + kappa); the
design maximises the sum of the latter greedily (monotone submodular).
"""
from __future__ import annotations

import numpy as np

from terranet.experiments.common import LINK_KAPPA


def design_survey(tile_of_pixel, tile_ids, pred_obs, k, kappa=LINK_KAPPA):
    """Greedy maximisation of F(T) = sum_s n_s(T) / (n_s(T) + kappa)."""
    S = pred_obs.shape[1]
    counts = np.zeros((len(tile_ids), S))
    pos = {t: i for i, t in enumerate(tile_ids)}
    rows = np.array([pos.get(t, -1) for t in tile_of_pixel])
    ok = rows >= 0
    np.add.at(counts, rows[ok], pred_obs[ok].astype(float))
    n = np.zeros(S)
    chosen = []
    for _ in range(k):
        f0 = (n / (n + kappa)).sum()
        gain = ((n + counts) / (n + counts + kappa)).sum(1) - f0
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        n += counts[j]
    return np.array(chosen)


def calibrate(city, PLr, keep, surveyed_tiles, kappa=LINK_KAPPA):
    """Per-site offsets from the links observed in the surveyed tiles.

    PLr: regression map; keep: hurdle decision (predicted below the cut-off).
    Returns (network map, calibrated regression map). In the surveyed tiles the
    links are known, including which sites are not received at all."""
    P, S = PLr.shape
    surveyed = np.isin(city.tile, surveyed_tiles)
    obs = np.isfinite(city.true_pl) & surveyed[:, None]
    res = (city.true_pl - PLr)[obs]
    site = np.broadcast_to(np.arange(S), (P, S))[obs]
    rbar = res.mean() if res.size else 0.0
    n_s = np.bincount(site, minlength=S)
    sum_s = np.bincount(site, weights=res, minlength=S)
    off = (sum_s + kappa * rbar) / (n_s + kappa)
    R = PLr + off[None, :]
    M = np.where(keep, R, np.inf)
    M[surveyed] = city.true_pl[surveyed]            # surveyed links are known
    return M, R
