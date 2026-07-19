#!/usr/bin/env python
"""Phase 4: aggregate eval.json across folds/seeds -> paper tables with
paired bootstrap CIs and Holm-Bonferroni-corrected significance."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from terranet.evaluation.statistics import holm_bonferroni, paired_bootstrap_ci, paired_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="glob of run dirs")
    ap.add_argument("--metric", default="gamma_rmse")
    args = ap.parse_args()
    rows = []
    for run in glob.glob(args.runs):
        f = Path(run) / "eval.json"
        if f.exists():
            r = json.loads(f.read_text())
            r["run"] = Path(run).name
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no eval.json found")
    df["model"] = df["run"].str.rsplit("_loco_", n=1).str[0]
    df["fold"] = df["run"].str.extract(r"loco_([^_]+)")
    piv = df.pivot_table(index="fold", columns="model", values=args.metric, aggfunc="mean")
    print(piv.round(4))
    models = list(piv.columns)
    if len(models) >= 2:
        ref = models[0]
        pvals = {}
        for m in models[1:]:
            a, b = piv[ref].to_numpy(), piv[m].to_numpy()
            stats_ = paired_test(b, a)
            ci = paired_bootstrap_ci(b, a)
            pvals[m] = stats_["p_t"]
            print(f"{m} vs {ref}: diff={ci['diff_mean']:.4f} "
                  f"CI=[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}] dz={stats_['cohens_dz_folds']:.2f}")
        print("Holm-Bonferroni:", holm_bonferroni(pvals))
    out = Path("outputs/tables") / f"summary_{args.metric}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    piv.to_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
