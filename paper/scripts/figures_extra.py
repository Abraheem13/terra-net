"""Result figures of the manuscript beyond the five drawn in make_assets.py.

Imported by make_assets.py (same directory). Every function reads only the
result tables it is given and writes one PDF into paper/figures. Palette: three
hues validated for every pairwise contrast under colour-vision deficiency
(blue, orange, aqua), achromatic tones and an ordinal blue ramp; the identity
of a series is always carried by a marker, a line style or a hatch as well as
by its colour, and legends are placed outside the data area.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_BLACK, C_GREY, C_LIGHT, C_PALE = "#1a1a1a", "#7a7a7a", "#c9c9c9", "#cde2fb"
BLUES = ["#86b6ef", "#2a78d6", "#104281"]
W1, W2 = 3.46, 7.16          # single / double column width (in)


def _top_legend(fig, handles, labels, ncol):
    fig.legend(handles, labels, frameon=False, ncol=ncol, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2,
               handletextpad=0.4)


def _plain(label):
    """Plain-text form of a LaTeX label for matplotlib (no math needed)."""
    return (str(label).replace("\\'E", "\u00c9").replace("\\,", " ")
            .replace("\\&", "&").replace("~", " "))


def _panels(ax):
    for i, a in enumerate(np.atleast_1d(ax)):
        a.set_title(f"({'abcdef'[i]})", loc="left")


def _save(fig, path):
    fig.tight_layout(w_pad=1.4)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
def fig_convergence(T, F, nice):
    """RMS difference to the 1e9-ray reference and share of reached cells at
    140 dB against the ray budget, for every convergence site."""
    c = pd.read_csv(T / "rt_convergence.csv")
    cities = list(dict.fromkeys(c.city))
    hue = [C_BLUE, C_ORANGE, C_AQUA, C_GREY]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for i, city in enumerate(cities):
        for j, (site, g) in enumerate(c[c.city == city].groupby("site", sort=True)):
            g = g.sort_values("rays")
            g = g[g.rays < g.ref_rays]
            kw = dict(color=hue[i % len(hue)], ls="-" if j == 0 else "--",
                      marker="o" if j == 0 else "s", ms=3.2, mfc=hue[i % len(hue)] if j == 0
                      else "white", label=f"{_plain(nice(city))}, site {int(site)}")
            ax[0].plot(g.rays, g.rms_diff_140, **kw)
            ax[1].plot(g.rays, 100 * g.frac_of_ref_140, **kw)
    budget = c.rays.loc[c.rays.sub(1e8).abs().idxmin()]
    for a in ax:
        a.set_xscale("log")
        a.axvline(budget, color=C_GREY, lw=0.6, ls=":")
        a.set_xlabel("rays per site")
    ax[0].set_ylabel(r"RMS difference at $\leq$140 dB (dB)")
    ax[1].set_ylabel(r"cells reached at $\leq$140 dB (% of reference)")
    ax[0].text(budget, ax[0].get_ylim()[1], " corpus budget", fontsize=6, color=C_GREY,
               ha="left", va="top")
    _panels(ax)
    h, lab = ax[0].get_legend_handles_labels()
    _top_legend(fig, h, lab, ncol=3)
    _save(fig, F / "fig_convergence.pdf")


def fig_noise(T, F, nice):
    """Standard error of the tile exponent per city under the three estimators."""
    d = pd.read_csv(T / "label_noise.csv").sort_values("gamma_se_cluster")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for j, (col, lab) in enumerate([("gamma_se_analytic", "independent residuals"),
                                    ("gamma_se_cluster", "cluster-robust (sites)"),
                                    ("gamma_se_split_full", "split-half, extrapolated")]):
        ax.bar(x + (j - 1) * 0.27, d[col], 0.26, color=BLUES[j], edgecolor="none", label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([_plain(nice(c)) for c in d.city], rotation=60, ha="right")
    ax.set_ylabel(r"standard error of $\hat\gamma_t$ (RMS over tiles)")
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              handlelength=1.4, handletextpad=0.4)
    ax.set_ylim(0, 1.35 * d.gamma_se_split_full.max())
    _save(fig, F / "fig_noise.pdf")


def fig_city_results(per, F, nice):
    """Per-city path loss, serving site and regret of the three reference models."""
    ops = [("oracle", C_GREY, "Tile labels (oracle)"), ("gbdt", C_AQUA, "GBDT tile transfer"),
           ("link", C_BLUE, "Link transfer")]
    cities = list(per.loc["oracle"].pl_rmse_db.sort_values(ascending=False).index)
    x = np.arange(len(cities))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4))
    metrics = [("pl_rmse_db", "path-loss RMSE (dB)", 1.0),
               ("assoc_acc", "serving-site accuracy (%)", 100.0),
               ("regret", r"planning regret, $K{=}5$ (%)", 100.0)]
    for a, (mtr, lab, sc) in zip(ax, metrics, strict=True):
        for j, (o, col, name) in enumerate(ops):
            a.bar(x + (j - 1) * 0.27, sc * per.loc[o].reindex(cities)[mtr], 0.26, color=col,
                  edgecolor="none", label=name)
        a.set_xticks(x)
        a.set_xticklabels([_plain(nice(c)) for c in cities], rotation=60, ha="right")
        a.set_ylabel(lab)
    _panels(ax)
    h, lab = ax[0].get_legend_handles_labels()
    _top_legend(fig, h, lab, ncol=3)
    _save(fig, F / "fig_cityres.pdf")


def fig_decomp(T, F, op_name):
    """The receiver-tile variants, the physics baselines and link transfer."""
    v = pd.read_csv(T / "network_variants.csv")
    pl = pd.read_csv(T / "planning_variants.csv")
    per = v.groupby(["operator", "fold"]).mean(numeric_only=True)
    per["regret"] = pl[pl.K == 5].groupby(["operator", "fold"]).regret.mean()
    m = per.groupby("operator").mean()
    groups = [("oracle", C_GREY), ("oracle+h", C_GREY), ("oracle+los", C_GREY),
              ("oracle+los+h", C_GREY), ("median", C_LIGHT), ("median+h", C_LIGHT),
              ("median+los+h", C_LIGHT), ("gbdt", C_AQUA), ("gbdt+h", C_AQUA),
              ("gbdt+los", C_AQUA), ("gbdt+los+h", C_AQUA), ("fspl_ke", C_ORANGE),
              ("deygout", C_ORANGE), ("uma_geo", C_ORANGE), ("link", C_BLUE)]
    ops = [g for g in groups if g[0] in m.index]
    y = np.arange(len(ops))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    metrics = [("pl_rmse_db", "path-loss RMSE (dB)", 1.0),
               ("assoc_acc", "serving-site accuracy (%)", 100.0),
               ("regret", r"planning regret, $K{=}5$ (%)", 100.0)]
    for a, (mtr, lab, sc) in zip(ax, metrics, strict=True):
        a.barh(y, [sc * m.loc[o, mtr] for o, _ in ops], color=[c for _, c in ops],
               edgecolor="none", height=0.72)
        a.set_yticks(y)
        a.invert_yaxis()
        a.set_xlabel(lab)
        a.set_yticklabels([])
    ax[0].set_yticklabels([_plain(op_name[o]) for o, _ in ops])
    _panels(ax)
    _top_legend(fig, [Patch(color=c) for c in (C_GREY, C_LIGHT, C_AQUA, C_ORANGE, C_BLUE)],
                ["exact tile labels", "source median", "GBDT transfer", "per-link physics",
                 "link transfer"], ncol=5)
    _save(fig, F / "fig_decomp.pdf")


def fig_survey(n, pl, F):
    """Per-site calibration of the link model against the number of surveyed tiles."""
    reg = pl.groupby(["operator", "fold"]).regret.mean().groupby("operator").mean()
    m = n.groupby(["operator", "fold"])[["pl_rmse_db", "assoc_acc"]].mean() \
        .groupby("operator").mean()
    ks = [0, 10, 25, 50]
    variants = [("", C_BLUE, "-", "o", r"random tiles, EB $\kappa$"),
                ("f", C_BLUE, "--", "s", r"random tiles, fixed $\kappa$"),
                ("d", C_ORANGE, "-", "^", "variance design"),
                ("s", C_AQUA, "-", "D", "stratified design")]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    for suf, col, ls, mk, lab in variants:
        names = ["link"] + [f"link+k{k}{suf}" for k in ks[1:]]
        if not all(o in m.index for o in names):
            continue
        kw = dict(color=col, ls=ls, marker=mk, ms=3.2, mfc=col if ls == "-" else "white",
                  label=lab)
        ax[0].plot(ks, [m.loc[o, "pl_rmse_db"] for o in names], **kw)
        ax[1].plot(ks, [100 * m.loc[o, "assoc_acc"] for o in names], **kw)
        ax[2].plot(ks, [100 * reg.loc[o] for o in names], **kw)
    for a, lab in zip(ax, ["path-loss RMSE (dB)", "serving-site accuracy (%)",
                           r"planning regret, $K{=}5$ (%)"], strict=True):
        a.set_xticks(ks)
        a.set_xlabel("surveyed tiles $k$")
        a.set_ylabel(lab)
    _panels(ax)
    h, lab = ax[0].get_legend_handles_labels()
    _top_legend(fig, h, lab, ncol=4)
    _save(fig, F / "fig_survey.pdf")


def fig_certified(T, F, op_name):
    """Claimed and falsely claimed coverage under the certificate and the naive claim."""
    c = pd.read_csv(T / "certified.csv")
    sub = c[(c.alpha == 0.10) & (c.level == 140.0) & (c["rank"] == 1)]
    m = sub.groupby(["operator", "city"]).mean(numeric_only=True).groupby("operator").mean()
    ops = [o for o in ["oracle", "median", "uma_nlos", "kernel", "ridge", "gbdt", "link_reg",
                       "link"] if o in m.index]
    y = np.arange(len(ops))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax[0].barh(y - 0.19, 100 * m.loc[ops, "claimed"], 0.36, color=C_BLUE, edgecolor="none",
               label="certified claim")
    ax[0].barh(y + 0.19, 100 * m.loc[ops, "naive_claimed"], 0.36, color=C_LIGHT,
               edgecolor="none", label="naive claim")
    tc = 100 * m.loc[ops[0], "true_cov"]
    ax[0].axvline(tc, color=C_BLACK, ls=":", lw=0.8)
    ax[0].text(tc, -0.7, " true covered share", fontsize=6, ha="right", va="bottom")
    ax[0].set_xlabel("pixels claimed covered (%)")
    ax[1].barh(y - 0.19, 100 * m.loc[ops, "false_claim"], 0.36, color=C_BLUE, edgecolor="none")
    ax[1].barh(y + 0.19, 100 * m.loc[ops, "naive_false"], 0.36, color=C_LIGHT, edgecolor="none")
    ax[1].axvline(10, color=C_BLACK, ls=":", lw=0.8)
    ax[1].text(10, -0.7, r" $\alpha=10\%$", fontsize=6, ha="left", va="bottom")
    ax[1].set_xlabel("pixels claimed but not covered (%)")
    for a in ax:
        a.set_yticks(y)
        a.invert_yaxis()
    ax[0].set_yticklabels([_plain(op_name[o]) for o in ops])
    ax[1].set_yticklabels([])
    _panels(ax)
    h, lab = ax[0].get_legend_handles_labels()
    _top_legend(fig, h, lab, ncol=2)
    _save(fig, F / "fig_certified.pdf")


def fig_replicates(T, F, per):
    """The main corpus and the three replicates of the four re-traced cities."""
    s = pd.read_csv(T / "siting.csv")
    reps = [("main", "Main corpus"), ("sionna_randsite", "Random rooftops"),
            ("sionna_capped", "Mid-rise rooftops"), ("sionna_mat", "Random materials")]
    reps = [r for r in reps if r[0] == "main" or r[0] in set(s.replicate)]
    cities = sorted(s.city.unique())
    mm = per.loc[per.index.get_level_values(1).isin(cities)].groupby("operator").mean()
    series = [("oracle", "oracle_rand", "Tile labels of the network", C_GREY),
              ("gbdt", "gbdt", "GBDT tile transfer", C_AQUA),
              ("link", "link", "Link transfer", C_BLUE)]
    vals = {}
    for rep, _ in reps:
        if rep == "main":
            vals[rep] = {o: (mm.loc[o, "pl_rmse_db"], mm.loc[o, "assoc_acc"], mm.loc[o, "regret"])
                         for o, _, _, _ in series}
        else:
            m = s[s.replicate == rep].groupby(["operator", "city"]).mean(numeric_only=True) \
                .groupby("operator").mean()
            vals[rep] = {o: (m.loc[r, "pl_rmse_db"], m.loc[r, "assoc_acc"], m.loc[r, "regret5"])
                         for o, r, _, _ in series}
    x = np.arange(len(reps))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    labels = ["path-loss RMSE (dB)", "serving-site accuracy (%)", r"planning regret, $K{=}5$ (%)"]
    for i, (a, lab) in enumerate(zip(ax, labels, strict=True)):
        for j, (o, _, name, col) in enumerate(series):
            v = [vals[rep][o][i] * (1.0 if i == 0 else 100.0) for rep, _ in reps]
            a.bar(x + (j - 1) * 0.27, v, 0.26, color=col, edgecolor="none", label=name)
        a.set_xticks(x)
        a.set_xticklabels([n for _, n in reps], rotation=25, ha="right")
        a.set_ylabel(lab)
    _panels(ax)
    h, lab = ax[0].get_legend_handles_labels()
    _top_legend(fig, h, lab, ncol=3)
    _save(fig, F / "fig_replicates.pdf")


def fig_band(T, T2, F, nice, op_name, f1, f2):
    """Tile exponents of the two bands per city and the network path-loss error at the
    second band, including the cross-band link models."""
    lab = pd.read_csv(T / "band_labels.csv").sort_values("gamma_median_1")
    bt = pd.read_csv(T / "band_transfer.csv")
    n2 = pd.concat([pd.read_csv(T2 / "network_metrics.csv"), pd.read_csv(T2 / "network_link.csv")])
    n2 = n2[n2.sites == n2.sites.max()]
    m = pd.concat([n2, bt]).groupby(["operator", "fold"]).pl_rmse_db.mean() \
        .groupby("operator").mean()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(lab))
    ax[0].bar(x - 0.18, lab.gamma_median_1, 0.34, color=C_BLUE, edgecolor="none",
              label=f"{f1:g} GHz")
    ax[0].bar(x + 0.18, lab.gamma_median_2, 0.34, color=C_ORANGE, edgecolor="none",
              label=f"{f2:g} GHz")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([_plain(nice(c)) for c in lab.city], rotation=60, ha="right")
    ax[0].set_ylabel(r"median tile exponent $\hat\gamma_t$")
    ax[0].legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    ops = [o for o in ["oracle", "median", "gbdt", "link", "link_x_fs", "link_x_phys"]
           if o in m.index]
    col = {"oracle": C_GREY, "median": C_LIGHT, "gbdt": C_AQUA, "link": C_BLUE,
           "link_x_fs": C_BLUE, "link_x_phys": C_BLUE}
    hatch = {"link_x_fs": "//", "link_x_phys": "xx"}
    y = np.arange(len(ops))
    for i, o in enumerate(ops):
        ax[1].barh(i, m.loc[o], color=col[o], edgecolor="white" if o in hatch else "none",
                   hatch=hatch.get(o, ""), height=0.72, lw=0.0)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([_plain(op_name[o]) for o in ops])
    ax[1].invert_yaxis()
    ax[1].set_xlabel(f"path-loss RMSE at {f2:g} GHz (dB)")
    _panels(ax)
    _save(fig, F / "fig_band.pdf")


def fig_mixed(T, F, per, sizes):
    """Compositions of the tile model at 100 m and across tile sizes."""
    mx = pd.read_csv(T / "mixed_path.csv")
    mxf = mx.groupby(["model", "operator", "fold"]).pl_rmse_db.mean() \
        .groupby(["model", "operator"]).mean()
    link = per.loc["link"].pl_rmse_db.mean()
    comps = [("Receiver tile", per.loc["oracle"].pl_rmse_db.mean(),
              per.loc["gbdt"].pl_rmse_db.mean()),
             ("Path-averaged", mxf.loc[("average", "oracle")], mxf.loc[("average", "gbdt")]),
             ("Piecewise", mxf.loc[("piecewise", "oracle")], mxf.loc[("piecewise", "gbdt")])]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    x = np.arange(len(comps))
    ax[0].bar(x - 0.18, [c[1] for c in comps], 0.34, color=C_GREY, edgecolor="none",
              label="exact tile labels")
    ax[0].bar(x + 0.18, [c[2] for c in comps], 0.34, color=C_AQUA, edgecolor="none",
              label="GBDT transfer")
    ax[0].axhline(link, color=C_BLUE, ls="--", lw=1.0, label="link transfer")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([c[0] for c in comps])
    ax[0].set_ylabel("path-loss RMSE (dB)")
    ax[0].set_ylim(0, None)
    series = []
    for gs in sizes:
        tag = "" if gs == 100 else f"_G{gs}"
        d = pd.read_csv(T / f"mixed_path{tag}.csv")
        series.append(d.groupby(["model", "operator", "fold"]).pl_rmse_db.mean()
                      .groupby(["model", "operator"]).mean())
    lines = [("average", "oracle", C_GREY, "-", "o", "path-averaged, labels"),
             ("average", "gbdt", C_AQUA, "-", "o", "path-averaged, GBDT"),
             ("piecewise", "oracle", C_GREY, "--", "s", "piecewise, labels"),
             ("piecewise", "gbdt", C_AQUA, "--", "s", "piecewise, GBDT")]
    for model, o, col, ls, mk, lab in lines:
        ax[1].plot(sizes, [s_.loc[(model, o)] for s_ in series], color=col, ls=ls, marker=mk,
                   ms=3.2, mfc=col if ls == "-" else "white", label=lab)
    ax[1].axhline(link, color=C_BLUE, ls="--", lw=1.0, label="link transfer")
    ax[1].set_xscale("log")
    ax[1].set_xticks(sizes)
    ax[1].set_xticklabels([str(s_) for s_ in sizes])
    ax[1].set_xlabel("tile side $G_s$ (m)")
    ax[1].set_ylabel("path-loss RMSE (dB)")
    _panels(ax)
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, lab in list(zip(h0, l0, strict=True)) + list(zip(h1, l1, strict=True)):
        if lab not in seen:
            seen.add(lab)
            hh.append(h)
            ll.append(lab)
    _top_legend(fig, hh, ll, ncol=4)
    _save(fig, F / "fig_mixed.pdf")


def fig_external(T, F, nice, op_name):
    """Path loss and serving site on the external scenes, per scene."""
    e = pd.read_csv(T / "external.csv")
    e = e.groupby(["scene", "operator"], sort=False).mean(numeric_only=True)
    scenes = list(dict.fromkeys(e.index.get_level_values(0)))
    ops = [("oracle", C_GREY, ""), ("gbdt", C_AQUA, ""), ("link", C_BLUE, ""),
           ("link+k25", C_BLUE, "//")]
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for a, (mtr, lab, sc) in zip(ax, [("pl_rmse_db", "path-loss RMSE (dB)", 1.0),
                                       ("assoc_acc", "serving-site accuracy (%)", 100.0)],
                                 strict=True):
        for j, (o, col, hatch) in enumerate(ops):
            a.bar(x + (j - 1.5) * 0.2, [sc * e.loc[(s_, o), mtr] for s_ in scenes], 0.19,
                  color=col, edgecolor="white" if hatch else "none", hatch=hatch, lw=0.0,
                  label=_plain(op_name[o]))
        a.set_xticks(x)
        a.set_xticklabels([_plain(nice(s_)) for s_ in scenes])
        a.set_ylabel(lab)
    _panels(ax)
    h, lab = ax[0].get_legend_handles_labels()
    _top_legend(fig, h, lab, ncol=4)
    _save(fig, F / "fig_external.pdf")
