#!/usr/bin/env python
"""Independent cross-check of the manuscript numbers.

    python paper/scripts/verify_numbers.py [--since ISO-timestamp]

This script does not import make_assets.py. It re-derives a set of headline
quantities from outputs/tables/*.csv with plain pandas, formats them with the
manuscript conventions (ddof = 1, per-fold mean then mean over folds) and
compares them with the macros in paper/generated/numbers.tex. It also checks
that every CSV the manuscript depends on was written after --since (so that no
stale result from an earlier run survives), that every number in the cover
letter and the highlights appears in numbers.tex, and that the PDF contains
the macro values. Exit status 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
T = REPO / "outputs" / "tables"
BAD = []


def macros():
    txt = (HERE / "generated" / "numbers.tex").read_text()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", txt, re.M):
        out[m.group(1)] = m.group(2).replace("\\ensuremath{-}", "-").replace("{,}", ",")
    return out


def check(name, expected, N):
    got = N.get(name, "<missing>")
    if str(expected) != got:
        BAD.append(f"{name}: numbers.tex has {got!r}, recomputed {expected!r}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def nets():
    n = pd.concat([pd.read_csv(T / "network_metrics.csv"), pd.read_csv(T / "network_link.csv")])
    pl = pd.concat([pd.read_csv(T / "planning.csv"), pd.read_csv(T / "planning_link.csv")])
    n = n[n.sites == n.sites.max()]
    pl = pl[(pl.sites == pl.sites.max()) & (pl.K == 5)]
    per = n.groupby(["operator", "fold"]).mean(numeric_only=True)
    per["regret"] = pl.groupby(["operator", "fold"]).regret.mean()
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO time; every CSV must be newer")
    args = ap.parse_args()
    N = macros()

    # ---- freshness ---------------------------------------------------------
    if args.since:
        since = dt.datetime.fromisoformat(args.since).timestamp()
        stale = [p.name for p in sorted(T.glob("*.csv")) if p.stat().st_mtime < since]
        if stale:
            BAD.append("stale result tables: " + ", ".join(stale))
        for p in sorted((HERE / "generated").glob("*.tex")):
            if p.stat().st_mtime < since:
                BAD.append(f"stale generated file {p.name}")

    # ---- corpus --------------------------------------------------------------
    c = pd.read_csv(T / "corpus.csv")
    fits = pd.read_csv(T / "fits.csv")
    check("nCities", len(c), N)
    check("nSites", int(c.transmitters.sum()), N)
    check("nSitesPerCity", int(c.transmitters.median()), N)
    check("nSamples", f"{int(c.samples.sum()):,}", N)
    check("nTiles", f"{int(c.tiles.sum()):,}", N)
    check("impMin", pct(c.frac_imputed.min(), 0), N)
    check("impMax", pct(c.frac_imputed.max(), 0), N)
    check("betweenSd", f"{fits.gamma_median.std(ddof=1):.2f}", N)
    check("gammaMedMin", f"{fits.gamma_median.min():.2f}", N)
    check("gammaMedMax", f"{fits.gamma_median.max():.2f}", N)

    # ---- label precision -----------------------------------------------------
    ln = pd.read_csv(T / "label_noise.csv")

    def rms(col):
        return float(np.sqrt((ln[col] ** 2).mean()))
    check("seRatio", f"{rms('gamma_se_cluster') / rms('gamma_se_analytic'):.1f}", N)
    check("seSplitDiff", f"{np.sqrt(2) * rms('gamma_se_half'):.2f}", N)
    check("seIid", f"{rms('gamma_se_analytic'):.3f}", N)
    check("seCluster", f"{rms('gamma_se_cluster'):.3f}", N)

    # ---- zero-shot tile transfer ---------------------------------------------
    m = pd.concat([pd.read_csv(T / "loco_metrics.csv"), pd.read_csv(T / "neural_metrics.csv")])
    g = m.groupby(["fold", "operator"]).gamma_rmse.mean().unstack()
    for op, key in [("median", "Median"), ("kernel", "Kernel"), ("gbdt", "Gbdt"),
                    ("ridge", "Ridge"), ("mlp", "Mlp"), ("gp_ard", "GpArd"), ("knn", "Knn"),
                    ("coral", "Coral"), ("encoder", "Encoder"), ("gbdt_iw", "GbdtIw"),
                    ("uma_los", "UmaLos"), ("uma_nlos", "UmaNlos"), ("cost231", "Cost")]:
        if op in g:
            check(f"rmse{key}", f"{g[op].mean():.3f}", N)
            check(f"sd{key}", f"{g[op].std(ddof=1):.3f}", N)
    check("gainGbdtMedian", pct(1 - g["gbdt"].mean() / g["median"].mean()), N)
    check("nGbdtMedian", str(int((g["gbdt"] < g["median"]).sum())), N)
    check("nKernelMedian", str(int((g["kernel"] < g["median"]).sum())), N)

    # ---- statistics ----------------------------------------------------------
    s = pd.read_csv(T / "stats.csv")
    check("nFolds", int(s.n_folds.max()), N)
    zs = s[s.family == "zero_shot"]
    check("nZeroShotSigBetter", int((zs.holm_reject & (zs["diff"] < 0)).sum()), N)
    check("nZeroShotTotal", len(zs), N)
    row = s[(s.a == "gbdt") & (s.b == "median") & (s.family == "zero_shot")].iloc[0]
    check("holmGbdtMedian", "significant" if row.holm_reject else "not significant", N)
    check("pGbdtMedian", "$<$0.001" if row.p_wilcoxon < 1e-3 else f"{row.p_wilcoxon:.3f}", N)
    row = s[(s.a == "link") & (s.b == "gbdt") & (s.metric == "pl_rmse_db")
            & (s.family == "network")].iloc[0]
    check("nNetLinkGbdt", f"{row.n_better}", N)
    check("holmNetLinkGbdt", "significant" if row.holm_reject else "not significant", N)

    # ---- network level -------------------------------------------------------
    per = nets()
    mean = per.groupby("operator").mean()
    for op, key in [("oracle", "Oracle"), ("median", "Median"), ("gbdt", "Gbdt"),
                    ("link", "Link"), ("link_reg", "LinkReg"), ("uma_nlos", "UmaNlos"),
                    ("kernel", "Kernel"), ("ridge", "Ridge"), ("link+k25", "LinkTwentyFive")]:
        if op in mean.index:
            check(f"net{key}", f"{mean.loc[op, 'pl_rmse_db']:.1f}", N)
            check(f"assoc{key}", pct(mean.loc[op, "assoc_acc"]), N)
            check(f"regret{key}", pct(mean.loc[op, "regret"]), N)
            check(f"edge{key}", f"{mean.loc[op, 'edge_disp_m']:.0f}", N) if key in (
                "Oracle", "Median", "Gbdt", "Link", "UmaNlos", "LinkReg") else None
    lk = per.loc["link"]
    check("nLinkBetterGbdtPl", int((lk.pl_rmse_db < per.loc["gbdt"].pl_rmse_db).sum()), N)
    check("nLinkBetterOraclePl", int((lk.pl_rmse_db < per.loc["oracle"].pl_rmse_db).sum()), N)
    check("nLinkBetterOracleRegret", int((lk.regret < per.loc["oracle"].regret).sum()), N)
    check("linkMin", f"{lk.pl_rmse_db.min():.1f}", N)
    check("linkMax", f"{lk.pl_rmse_db.max():.1f}", N)
    check("reachLink", pct(mean.loc["link", "reach_acc"]), N)
    check("covTarget", "90", N)

    # ---- receiver-tile variants and physics baselines -------------------------
    v = pd.read_csv(T / "network_variants.csv")
    pv = pd.read_csv(T / "planning_variants.csv")
    perv = v.groupby(["operator", "fold"]).mean(numeric_only=True)
    perv["regret"] = pv[pv.K == 5].groupby(["operator", "fold"]).regret.mean()
    mv = perv.groupby("operator").mean()
    for op, key in [("gbdt+h", "GbdtH"), ("gbdt+los", "GbdtLos"), ("gbdt+los+h", "GbdtLosH"),
                    ("oracle+los+h", "OracleLosH"), ("deygout", "Deygout"), ("fspl_ke", "Ke"),
                    ("uma_geo", "UmaGeo"), ("link", "Link"), ("gbdt", "Gbdt")]:
        if op in mv.index:
            check(f"var{key}Net", f"{mv.loc[op, 'pl_rmse_db']:.1f}", N)
            check(f"var{key}Assoc", pct(mv.loc[op, "assoc_acc"]), N)
            check(f"var{key}Regret", pct(mv.loc[op, "regret"]), N)
    phys = [o for o in ["fspl_ke", "deygout", "uma_geo"] if o in mv.index]
    bp = mv.loc[phys, "pl_rmse_db"].idxmin()
    check("varBestPhysNetVal", f"{mv.loc[bp, 'pl_rmse_db']:.1f}", N)
    check("nLinkBetterPhysNet",
          int((perv.loc["link"].pl_rmse_db < perv.loc[bp].pl_rmse_db).sum()), N)
    check("nVarFolds", int(perv.loc["link"].shape[0]), N)

    # ---- certificate ---------------------------------------------------------
    ce = pd.read_csv(T / "certified.csv")
    sub = ce[(ce.alpha == 0.10) & (ce.level == 140.0) & (ce["rank"] == 1)]
    mc = sub.groupby(["operator", "city"]).mean(numeric_only=True).groupby("operator").mean()
    for op, key in [("link", "Link"), ("gbdt", "Gbdt"), ("oracle", "Oracle"),
                    ("median", "Median")]:
        check(f"certClaim{key}", pct(mc.loc[op, "claimed"]), N)
        check(f"certNaiveFalse{key}", pct(mc.loc[op, "naive_false"], 2), N)
        check(f"certFalse{key}", pct(mc.loc[op, "false_claim"], 2), N)

    # ---- few shot --------------------------------------------------------------
    k = pd.read_csv(T / "kshot.csv")
    pc = k[(k["mode"] == "eb") & (k.sampling == "random") & (k.operator == "gbdt")] \
        .groupby(["fold", "k"]).gamma_rmse.mean().unstack()
    check("fsImprovedTen", int((pc[10] < pc[0]).sum()), N)
    check("fsImprovedFifty", int((pc[50] < pc[0]).sum()), N)

    # ---- censoring -----------------------------------------------------------
    cs = pd.read_csv(T / "censoring.csv")
    x = cs.dgamma_median_140.median()
    got = N.get("censDgOneForty", "nan")
    if abs(float(got.replace("$", "")) - x) > 5e-4:
        BAD.append(f"censDgOneForty: numbers.tex has {got!r}, recomputed {x:.3f}")

    # ---- band 2 ------------------------------------------------------------------
    T2 = REPO / "outputs" / "sionna_7p5" / "tables"
    if (T2 / "network_link.csv").exists():
        n2 = pd.concat([pd.read_csv(T2 / "network_metrics.csv"),
                        pd.read_csv(T2 / "network_link.csv")])
        n2 = n2[n2.sites == n2.sites.max()]
        m2 = n2.groupby(["operator", "fold"]).mean(numeric_only=True).groupby("operator").mean()
        for op, key in [("link", "Link"), ("gbdt", "Gbdt"), ("oracle", "Oracle")]:
            if f"bandNet{key}" in N:
                check(f"bandNet{key}", f"{m2.loc[op, 'pl_rmse_db']:.1f}", N)

    # ---- the text around the numbers -----------------------------------------------
    vals = set(N.values())
    for name in ("cover_letter.md", "highlights.txt"):
        p = HERE / name
        if not p.exists():
            BAD.append(f"missing {name}")
            continue
        txt = p.read_text()
        for num in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|dB|m\b|,\d{3})", txt):
            n0 = num
            if n0 in {"3.5", "7.5", "100", "1000", "90", "95", "150", "130", "140", "25", "10",
                      "50", "2026", "257", "108620", "20", "6", "12", "45", "220", "80"}:
                continue
            if n0 not in vals and f"{n0}" not in " ".join(vals):
                BAD.append(f"{name}: number {n0} is not a manuscript macro value")

    # ---- the PDF carries the macro values --------------------------------------------
    pdf = HERE / "main.pdf"
    if pdf.exists():
        try:
            txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                                 text=True, check=True).stdout
            txt = re.sub(r"\s+", " ", txt)
            for key in ("netLink", "netOracle", "assocLink", "assocOracle", "regretLink",
                        "certClaimLink", "seRatio", "rmseGbdt", "rmseMedian"):
                if N[key].replace("-", "") not in txt:
                    BAD.append(f"PDF does not contain the value of \\{key} = {N[key]}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"(pdftotext unavailable: {e})")

    if BAD:
        print("\n".join(BAD))
        sys.exit(1)
    print(f"verify_numbers: {len(N)} macros read, all recomputed quantities agree")


if __name__ == "__main__":
    main()
