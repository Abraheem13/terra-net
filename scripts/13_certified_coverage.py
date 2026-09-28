#!/usr/bin/env python
"""Certified coverage maps from leave-one-city-out residuals.

For a transfer model and a held-out city c, let r_p be the true minus the
predicted serving path loss of pixel p (the smallest path loss over the sites),
and q_c(alpha) the (1 - alpha) quantile of r over the pixels of c. Every q_c comes
from a model that never saw city c, so under exchangeable cities the n values
q_1..q_n are exchangeable. For a new city t the bound

    qhat_t = max_{c != t} q_c(alpha)

therefore satisfies P(q_t(alpha) <= qhat_t) >= (n - 1) / n: with probability
at least (n-1)/n over the draw of the city, at least a fraction 1 - alpha of
its pixels have r_p <= qhat_t. More generally, with qhat_t the j-th largest of
the other n - 1 values the probability is at least (n - j) / n; j = 1 and 2
are reported (column `rank`). A pixel is CLAIMED covered at level L when
PLhat_serv(p) + qhat_t <= L; then r_p <= qhat_t implies that the true serving
path loss is at most L. False claims are thus confined to the pixels with
r_p > qhat_t.

Inputs: outputs/predictions/serving_tile.parquet (09), serving_link.parquet (11)
Output: outputs/tables/certified.csv  one row per (operator, seed, city, alpha, L)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.experiments.common import load_cfg
from terranet.utils.logging import get_logger

log = get_logger("certified")
ALPHAS = (0.05, 0.10)
LEVELS = (130.0, 140.0)
OPERATORS = ("oracle", "median", "uma_nlos", "kernel", "ridge", "gbdt", "link", "link_reg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    args = ap.parse_args()
    _, base = load_cfg(args.config)
    pr = Path(base.paths.outputs) / "predictions"
    S = pd.concat([pd.read_parquet(pr / "serving_tile.parquet"),
                   pd.read_parquet(pr / "serving_link.parquet")], ignore_index=True)
    S = S[S.operator.isin(OPERATORS)]
    rows = []
    for (op, seed), g in S.groupby(["operator", "seed"]):
        cities = sorted(g.fold.unique())
        by = {c: g[g.fold == c] for c in cities}
        for a in ALPHAS:
            q = {c: float(np.quantile((by[c].true_serv_pl - by[c].pred_serv_pl)
                                      .to_numpy(np.float64), 1 - a, method="higher"))
                 for c in cities}
            for t, j in [(t, j) for t in cities for j in (1, 2)]:
                qhat = sorted((q[c] for c in cities if c != t), reverse=True)[j - 1]
                d = by[t]
                ps = d.pred_serv_pl.to_numpy(np.float64)
                ts = d.true_serv_pl.to_numpy(np.float64)
                r = ts - ps
                for L in LEVELS:
                    cov = ts <= L
                    claim, naive = ps + qhat <= L, ps <= L
                    rows.append(dict(
                        operator=op, seed=seed, city=t, alpha=a, level=L, rank=j,
                        n_cities=len(cities),
                        q_city=q[t], qhat=qhat, valid=bool(q[t] <= qhat),
                        resid_cov=float((r <= qhat).mean()),
                        true_cov=float(cov.mean()),
                        claimed=float(claim.mean()), false_claim=float((claim & ~cov).mean()),
                        naive_claimed=float(naive.mean()),
                        naive_false=float((naive & ~cov).mean())))
    df = pd.DataFrame(rows)
    out = Path(base.paths.outputs) / "tables" / "certified.csv"
    df.to_csv(out, index=False)
    s = df[(df.alpha == 0.10) & (df.level == 140.0) & (df["rank"] == 1)].groupby("operator")[
        ["qhat", "resid_cov", "valid", "claimed", "false_claim", "naive_false", "true_cov"]].mean()
    print(s.round(3))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
