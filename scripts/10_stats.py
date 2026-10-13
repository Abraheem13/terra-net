#!/usr/bin/env python
"""Fold-level inference: the city is the unit of analysis (11 folds).

For each comparison the per-fold difference of the metric (seed-averaged) is
summarised by its mean, a 95% percentile bootstrap CI (10,000 resamples of
folds), the exact two-sided Wilcoxon signed-rank p-value and the number of
folds in which the first operator is better. Holm-Bonferroni is applied
within each family.

Families
  zero_shot  every operator vs the source median (gamma RMSE)
  pairs      gbdt vs kernel, gbdt vs ridge, kernel vs knn, gp_ard vs kernel,
             gbdt_iw vs gbdt, coral vs ridge
  few_shot   k = 10 (random anchors, empirical-Bayes correction) vs k = 0 per
             operator, the facility-location design vs random anchors at
             k = 10, and the empirical-Bayes vs the plain bias correction
  network    path loss, serving site, SINR, coverage and planning regret (K = 5)
             on the 20-site networks: gbdt vs median, link vs gbdt, link vs
             oracle, link+k25 vs link, link+k25f (fixed kappa) vs link+k25,
             link+k25d and link+k25s (designs) vs link+k25
  decomp     the receiver-tile variants and the non-learned link baselines of
             19_tile_variants.py: the hurdle, the LoS state, and link transfer
             against the best tile variant and against the physics baselines
  band2      the same network metrics at 7.5 GHz (when available): link vs
             gbdt, link vs oracle, and the 3.5 GHz link model transferred
             across bands (link_x_phys) vs the model trained at 7.5 GHz

Output: outputs/tables/stats.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from terranet.evaluation.statistics import holm_bonferroni

N_BOOT = 10_000
NET_METRICS = [("pl_rmse_db", True), ("assoc_acc", False), ("sinr_mae_db", True),
               ("cov_acc_130", False), ("regret", True)]


def compare(a: pd.Series, b: pd.Series, lower_is_better=True):
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = float(wilcoxon(d, method="exact").pvalue) if np.any(d != 0) else 1.0
    better = int((d < 0).sum() if lower_is_better else (d > 0).sum())
    return dict(diff=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi), p_wilcoxon=p,
                n_better=better, n_folds=len(d))


def fold_table(net, plan, key):
    """Per-(operator, fold) means of the network metrics plus the K = 5 regret."""
    full = plan.sites.max()
    net = net[net.sites.fillna(full) == full]
    plan = plan[(plan.K == 5) & (plan.sites == full)]
    net = net.merge(plan[key + ["regret"]], on=key)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="outputs/tables")
    ap.add_argument("--band2", default="outputs/sionna_7p5/tables")
    args = ap.parse_args()
    T = Path(args.tables)
    m = pd.concat([pd.read_csv(T / "loco_metrics.csv"), pd.read_csv(T / "neural_metrics.csv")])
    fold_op = m.groupby(["fold", "operator"]).gamma_rmse.mean().unstack()
    rows = []

    fam = [(op, "median") for op in fold_op.columns if op != "median"]
    rows += [dict(family="zero_shot", a=a, b=b, metric="gamma_rmse",
                  **compare(fold_op[a], fold_op[b])) for a, b in fam]
    rows += [dict(family="pairs", a=a, b=b, metric="gamma_rmse",
                  **compare(fold_op[a], fold_op[b]))
             for a, b in [("gbdt", "kernel"), ("gbdt", "ridge"), ("kernel", "knn"),
                          ("gp_ard", "kernel"), ("gbdt_iw", "gbdt"), ("coral", "ridge")]]

    k = pd.read_csv(T / "kshot.csv")
    kk = k.groupby(["mode", "sampling", "operator", "fold", "k"]).gamma_rmse.mean().unstack()
    for op in sorted(k.operator.unique()):
        eb_r = kk.loc[("eb", "random", op)]
        rows.append(dict(family="few_shot", a=f"{op}@k10", b=f"{op}@k0", metric="gamma_rmse",
                         **compare(eb_r[10], eb_r[0])))
        rows.append(dict(family="few_shot", a=f"{op}@k10d", b=f"{op}@k10", metric="gamma_rmse",
                         **compare(kk.loc[("eb", "design", op)][10], eb_r[10])))
        rows.append(dict(family="few_shot", a=f"{op}@k10", b=f"{op}@k10bias", metric="gamma_rmse",
                         **compare(eb_r[10], kk.loc[("bias", "random", op)][10])))

    def network_family(T, family, pairs, extra=None):
        net = pd.concat([pd.read_csv(T / "network_metrics.csv"),
                         pd.read_csv(T / "network_link.csv")] + ([extra] if extra is not None
                                                                 else []))
        plan = pd.concat([pd.read_csv(T / "planning.csv"),
                          pd.read_csv(T / "planning_link.csv")])
        key = ["fold", "operator", "seed", "draw"]
        if extra is not None:
            full = plan.sites.max()
            net = net[net.sites.fillna(full) == full]
            plan = plan[(plan.K == 5) & (plan.sites == full)]
            net = net.drop(columns="regret", errors="ignore").merge(
                pd.concat([plan[key + ["regret"]], extra[key + ["regret"]]]), on=key)
        else:
            net = fold_table(net, plan, key)
        for metric, lib in NET_METRICS:
            fo = net.groupby(["fold", "operator"])[metric].mean().unstack()
            for a, b in pairs:
                if a in fo and b in fo:
                    rows.append(dict(family=family, a=a, b=b, metric=metric,
                                     **compare(fo[a], fo[b], lower_is_better=lib)))

    network_family(T, "network", [("gbdt", "median"), ("link", "gbdt"), ("link", "oracle"),
                                  ("link+k25", "link"), ("link+k25f", "link+k25"),
                                  ("link+k25d", "link+k25"), ("link+k25s", "link+k25")])

    if (T / "network_variants.csv").exists():
        net = fold_table(pd.read_csv(T / "network_variants.csv"),
                         pd.read_csv(T / "planning_variants.csv"), ["fold", "operator", "seed"])
        for metric, lib in NET_METRICS:
            fo = net.groupby(["fold", "operator"])[metric].mean().unstack()
            for a, b in [("gbdt+h", "gbdt"), ("gbdt+los", "gbdt"), ("gbdt+los+h", "gbdt+h"),
                         ("oracle+los+h", "oracle"), ("link", "gbdt+los+h"),
                         ("link", "oracle+los+h"), ("link", "fspl_ke"), ("link", "deygout"),
                         ("link", "uma_geo"), ("deygout", "fspl_ke")]:
                if a in fo and b in fo:
                    rows.append(dict(family="decomp", a=a, b=b, metric=metric,
                                     **compare(fo[a], fo[b], lower_is_better=lib)))

    T2 = Path(args.band2)
    if (T2 / "network_link.csv").exists() and (T / "band_transfer.csv").exists():
        bt = pd.read_csv(T / "band_transfer.csv").rename(columns={"regret5": "regret"})
        bt["draw"] = 0
        network_family(T2, "band2", [("link", "gbdt"), ("link", "oracle"),
                                     ("link_x_phys", "link")], extra=bt)

    df = pd.DataFrame(rows)
    df["holm_reject"] = False
    for _f, g in df.groupby("family"):
        rej = holm_bonferroni({i: p for i, p in zip(g.index, g.p_wilcoxon, strict=False)})
        df.loc[list(rej), "holm_reject"] = list(rej.values())
    df.to_csv(T / "stats.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
