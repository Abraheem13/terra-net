"""Few-shot correction of tile parameters from k surveyed (anchor) tiles.

The anchors' labels are noisy (their cluster-robust standard error is of the
order of the between-city spread), so the correction weights every anchor by
its precision and shrinks the result by empirical Bayes:

    w_i = 1 / (se_i^2 + s^2),   c = sum_i w_i r_i / sum_i w_i,   V = 1 / sum_i w_i,
    c_eb = tau^2 / (tau^2 + V) c,

with r_i the anchor residual (label minus zero-shot prediction), se_i the
label's standard error, s^2 the within-city error variance of the operator
and tau^2 the between-city variance of its zero-shot bias, both estimated on
the inner validation cities of the fold (outputs/tables/inner_bias.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SE_COLS = {"gamma": ("gamma_se_cl", "gamma_se"), "pl0": ("pl0_se_cl", "pl0_se")}


def label_se(te: pd.DataFrame) -> np.ndarray:
    """(n_tiles, 2) standard errors of the labels: cluster-robust, else iid."""
    out = []
    for cl, iid in SE_COLS.values():
        s = te[cl].to_numpy(float)
        s = np.where(np.isfinite(s), s, te[iid].to_numpy(float))
        out.append(np.where(np.isfinite(s), s, np.nanmedian(s)))
    return np.column_stack(out)


def eb_prior(inner_bias: pd.DataFrame, fold: str, op: str):
    """(s^2, tau^2) per target of operator `op` in `fold` from the inner cities."""
    g = inner_bias[(inner_bias.fold == fold) & (inner_bias.operator == op)]
    b = g[["bias_gamma", "bias_pl0"]].to_numpy()
    r2 = g[["rmse_gamma", "rmse_pl0"]].to_numpy() ** 2
    return (np.maximum((r2 - b ** 2).mean(0), 1e-6), np.maximum(b.var(0, ddof=1), 1e-6))


def eb_correction(resid, se, s2, tau2):
    """Precision-weighted mean residual shrunk by empirical Bayes (per target);
    returns (correction, anchor weights)."""
    w = 1.0 / (se ** 2 + s2[None, :])
    c = (w * resid).sum(0) / w.sum(0)
    V = 1.0 / w.sum(0)
    return tau2 / (tau2 + V) * c, w
