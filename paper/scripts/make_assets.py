#!/usr/bin/env python
"""Generate every data-dependent element of the manuscript.

    python paper/scripts/make_assets.py            (run from the repo root)

Reads only outputs/tables/*.csv and data/processed (tile tables), and writes
  paper/generated/numbers.tex   \\newcommand macros for every number in the text
  paper/generated/tab_*.tex     tables
  paper/figures/fig_*.pdf       figures (monochrome)
Nothing in the manuscript is typed by hand, so text, tables and figures
cannot disagree with the results. Statements of the text that depend on the
direction of a result are driven by macros as well (e.g. counts of cities,
"significant"/"not significant"), never asserted.

Conventions: standard deviations across cities use ddof = 1; "mean" over
seeds is taken per fold first, then across folds.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

NICE = {"newyork": "New York", "etoile": "\\'Etoile"}
OPS = ["median", "uma_los", "uma_nlos", "cost231", "kernel", "gp_ard", "knn", "ridge", "coral",
       "mlp", "encoder", "gbdt", "gbdt_iw"]
OP_NAME = {"median": "Source median", "uma_los": "3GPP UMa LoS", "uma_nlos": "3GPP UMa NLoS",
           "cost231": "COST-231 Hata", "kernel": "Kernel (NW)", "gp_ard": "GP-ARD",
           "knn": "$k$-NN", "ridge": "Ridge", "coral": "CORAL + ridge", "mlp": "MLP",
           "encoder": "Neural encoder", "gbdt": "GBDT", "gbdt_iw": "GBDT-IW",
           "oracle": "Tile labels (oracle)",
           "gbdt+k10": "GBDT + 10 tiles", "gbdt+k10d": "GBDT + 10 designed tiles",
           "link": "Link transfer", "link_reg": "Link, regressor only",
           "link+k10": "Link + 10 tiles", "link+k10f": "Link + 10 tiles, fixed $\\kappa$",
           "link+k10d": "Link + 10 tiles, variance design",
           "link+k10s": "Link + 10 tiles, stratified design",
           "link+k25": "Link + 25 tiles", "link+k25f": "Link + 25 tiles, fixed $\\kappa$",
           "link+k25d": "Link + 25 tiles, variance design",
           "link+k25s": "Link + 25 tiles, stratified design",
           "link+k50": "Link + 50 tiles", "link+k50f": "Link + 50 tiles, fixed $\\kappa$",
           "link+k50d": "Link + 50 tiles, variance design",
           "link+k50s": "Link + 50 tiles, stratified design",
           "link_x_phys": r"Link, 3.5\,GHz model",
           "link_x_fs": r"Link, 3.5\,GHz model, offset only",
           "oracle_rand": "Tile labels of this network", "oracle_main": "Tile labels, main corpus",
           # receiver-tile variants and physics baselines
           "oracle+h": "Tile labels + hurdle", "oracle+los": "Tile labels, dual slope",
           "oracle+los+h": "Tile labels, dual slope + hurdle",
           "median+h": "Source median + hurdle", "median+los": "Source median, dual slope",
           "median+los+h": "Source median, dual slope + hurdle",
           "gbdt+h": "GBDT + hurdle", "gbdt+los": "GBDT, dual slope",
           "gbdt+los+h": "GBDT, dual slope + hurdle",
           "fspl_ke": "Free space + knife edge", "deygout": "Free space + Deygout",
           "uma_geo": "3GPP UMa, geometric LoS"}
SHORT = {"median": "Median", "uma_los": "UMa-L", "uma_nlos": "UMa-N", "cost231": "COST",
         "kernel": "Kernel", "gp_ard": "GP", "knn": "$k$-NN", "ridge": "Ridge",
         "coral": "CORAL", "mlp": "MLP", "encoder": "Enc.", "gbdt": "GBDT", "gbdt_iw": "IW"}
MACRO = {"median": "Median", "uma_los": "UmaLos", "uma_nlos": "UmaNlos", "cost231": "Cost",
         "kernel": "Kernel", "gp_ard": "GpArd", "knn": "Knn", "ridge": "Ridge", "coral": "Coral",
         "mlp": "Mlp", "encoder": "Encoder", "gbdt": "Gbdt", "gbdt_iw": "GbdtIw"}
FS_OPS = ["kernel", "ridge", "mlp", "gbdt"]
KWORD = {10: "Ten", 25: "TwentyFive", 50: "Fifty"}
SUFFIX = {"": "R", "f": "F", "d": "D", "s": "S"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.linewidth": 0.5, "lines.linewidth": 0.9,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "axes.unicode_minus": True})
W1, W2 = 3.46, 7.16          # single / double column width (in)
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]

LINK_NAME = {"ke_loss_db": "knife-edge loss", "max_excess": "max.\\ excess height",
             "n_cross": "buildings crossed", "log_d3": "log distance",
             "blocked_frac": "blocked fraction", "built_len": "length over buildings",
             "d2": "horizontal distance", "dens_50": "built fraction, 50\\,m",
             "rx_bld_h": "receiver building height", "tx_h": "site height",
             "rx_indoor": "receiver indoors", "elev_deg": "elevation angle",
             "hmean_50": "mean height, 50\\,m", "tx_dens_100": "site built fraction, 100\\,m"}
H_CLASS = ["<5 m", "5\u201310 m", "10\u201330 m", "30\u201360 m", "\u226560 m"]
H_CLASS_TEX = ["$<$5\\,m", "5--10\\,m", "10--30\\,m", "30--60\\,m", "$\\geq$60\\,m"]


def desc_name(n, tex=False):
    """Readable name of a built-environment descriptor (without the d_ prefix)."""
    H = H_CLASS_TEX if tex else H_CLASS
    m = re.fullmatch(r"base_area_ratio_h(\d)", n)
    if m:
        return f"footprint ratio, {H[int(m.group(1))]}"
    m = re.fullmatch(r"n_buildings_h(\d)", n)
    if m:
        return f"building count, {H[int(m.group(1))]}"
    m = re.fullmatch(r"plane(\d+)_n", n)
    if m:
        return f"buildings reaching {m.group(1)}" + ("\\,m" if tex else " m")
    m = re.fullmatch(r"plane(\d+)_dist_(mean|std)", n)
    if m:
        unit = "\\,m" if tex else " m"
        return f"spacing at {m.group(1)}{unit} ({'mean' if m.group(2) == 'mean' else 's.d.'})"
    return {"base_area_mean": "mean footprint", "base_area_std": "s.d. footprint",
            "height_mean": "mean height", "height_std": "s.d. height",
            "volume_mean": "mean volume", "volume_std": "s.d. volume",
            "building_density": "footprint density"}[n]


def feat_name(c):
    if c in LINK_NAME:
        return LINK_NAME[c]
    return "tile " + desc_name(c.replace("d_", "", 1), tex=True)


def nice(c):
    return NICE.get(c, str(c).capitalize())


_MINUS = re.compile(r"(?<![\w\-\\{}.])-(?=\d)")


def minus(text: str) -> str:
    """Typeset negative numbers with a true minus sign (text and math mode)."""
    return _MINUS.sub(r"\\ensuremath{-}", text)


class Numbers:
    def __init__(self):
        self.lines = ["% generated by paper/scripts/make_assets.py -- do not edit"]
        self.keys = set()

    def __setitem__(self, k, v):
        assert k.isalpha() and k not in self.keys, k
        self.keys.add(k)
        self.lines.append(f"\\newcommand{{\\{k}}}{{{minus(str(v))}}}")

    def raw(self, k, v):
        """Macro without minus-sign typesetting (for use inside siunitx \\SI)."""
        assert k.isalpha() and k not in self.keys, k
        self.keys.add(k)
        self.lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")

    def write(self, p):
        p.write_text("\n".join(self.lines) + "\n")


def sci(n):
    """siunitx number in scientific notation, e.g. 100000000 -> \\num{1e8}."""
    m, e = f"{float(n):.2e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"\\num{{{m}e{int(e)}}}"


def fs(x, d=2):
    """Signed number without a negative zero."""
    x = 0.0 if round(float(x), d) == 0 else float(x)
    return f"{x:+.{d}f}"


def f(x, d=3):
    return f"{x:.{d}f}"


def thou(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pval(p):
    return r"$<$0.001" if p < 1e-3 else f"{p:.3f}"


def save_table(out, name, body):
    (out / f"{name}.tex").write_text("% generated by paper/scripts/make_assets.py\n"
                                     + minus(body))


def tabular(cols, header, rows):
    return "\n".join([r"\begin{tabular}{" + cols + "}", r"\toprule", *header, r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


def nets(T):
    """Network tables of the tile operators and the link model on the full
    network, plus the K = 5 regret, as per-(operator, fold) means."""
    n = pd.concat([pd.read_csv(T / "network_metrics.csv"), pd.read_csv(T / "network_link.csv")])
    pl = pd.concat([pd.read_csv(T / "planning.csv"), pd.read_csv(T / "planning_link.csv")])
    n = n[n.sites == n.sites.max()]
    pl = pl[(pl.sites == pl.sites.max()) & (pl.K == 5)]
    per = n.groupby(["operator", "fold"]).mean(numeric_only=True)
    per["regret"] = pl.groupby(["operator", "fold"]).regret.mean()
    return n, pl, per


# ----------------------------------------------------------------------------
def corpus(T, G, N, fig_dir, tiles):
    c = pd.read_csv(T / "corpus.csv")
    fits = pd.read_csv(T / "fits.csv")
    N["nCities"] = len(c)
    N["nSamples"] = thou(c.samples.sum())
    N["nSamplesM"] = f"{c.samples.sum() / 1e6:.1f}"
    N["nSites"] = int(c.transmitters.sum())
    N["nSitesPerCity"] = int(c.transmitters.median())
    N["nTiles"] = thou(c.tiles.sum())
    N["nBuildings"] = thou(c.buildings.sum())
    N["qcRemoved"] = thou(c.qc_removed.sum())
    N["qcShare"] = f"{100 * c.qc_removed.sum() / (c.samples.sum() + c.qc_removed.sum()):.3f}"
    lo, hi = c.frac_imputed.idxmin(), c.frac_imputed.idxmax()
    N["impMin"], N["impMax"] = pct(c.frac_imputed[lo], 0), pct(c.frac_imputed[hi], 0)
    N["impMinCity"], N["impMaxCity"] = nice(c.city[lo]), nice(c.city[hi])
    rows = []
    for r in c.itertuples():
        rows.append(f"{nice(r.city)} & {thou(r.buildings)} & {pct(r.frac_tag, 0)} & "
                    f"{pct(r.frac_levels, 0)} & {pct(r.frac_imputed, 0)} & "
                    f"{r.median_height_m:.1f} & {r.site_h_min:.0f} & {r.site_h_median:.0f} & "
                    f"{r.site_h_max:.0f} & {thou(r.samples)} & {thou(r.tiles)} \\\\")
    save_table(G, "tab_corpus", tabular("lrrrrrrrrrr", [
        r"& & \multicolumn{3}{c}{Height source (\%)} & & "
        r"\multicolumn{3}{c}{Site height (m)} & & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){7-9}",
        r"City & Buildings & tag & storeys & imputed & Median (m) & min & med. & max & "
        r"Links & Tiles \\"], [*rows, r"\midrule",
        f"Total & {thou(c.buildings.sum())} & & & & & & & & {thou(c.samples.sum())} & "
        f"{thou(c.tiles.sum())} \\\\"]))
    N["siteHMedMin"] = f"{c.site_h_median.min():.0f}"
    N["siteHMedMax"] = f"{c.site_h_median.max():.0f}"
    N["siteHMedMinCity"] = nice(c.city[c.site_h_median.idxmin()])
    N["siteHMedMaxCity"] = nice(c.city[c.site_h_median.idxmax()])
    N["siteHMax"] = f"{c.site_h_max.max():.0f}"
    N["siteHMaxCity"] = nice(c.city[c.site_h_max.idxmax()])
    N["siteHMedianAll"] = f"{c.site_h_median.median():.0f}"

    rows = []
    for r in fits.itertuples():
        rows.append(f"{nice(r.city)} & {r.gamma_global:.2f} & {r.pl0_global:.1f} & "
                    f"{r.gamma_median:.2f} & {r.gamma_sd:.2f} & {r.pl0_median:.1f} & "
                    f"{r.tau:.2f} & {r.shrink_median:.2f} & {r.fit_rmse_db:.1f} \\\\")
    save_table(G, "tab_fits", tabular("lrrrrrrrr", [
        r"& \multicolumn{2}{c}{City-wide fit} & \multicolumn{3}{c}{Tile labels} & & & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"City & $\gamma$ & $P_{L_0}$ & med.\ $\gamma$ & s.d.\ $\gamma$ & med.\ $P_{L_0}$ & "
        r"$\hat\tau$ & med.\ $w_t$ & RMSE \\"], rows))
    N["betweenSd"] = f(fits.gamma_median.std(ddof=1), 2)
    N["withinSdMin"], N["withinSdMax"] = f(fits.gamma_sd.min(), 2), f(fits.gamma_sd.max(), 2)
    N["betweenWithinRatio"] = f(fits.gamma_median.std(ddof=1) / fits.gamma_sd.median(), 1)
    lo, hi = fits.gamma_median.idxmin(), fits.gamma_median.idxmax()
    N["gammaMedMin"], N["gammaMedMax"] = f(fits.gamma_median[lo], 2), f(fits.gamma_median[hi], 2)
    N["gammaMedMinCity"], N["gammaMedMaxCity"] = nice(fits.city[lo]), nice(fits.city[hi])
    N["olsMin"], N["olsMax"] = f(fits.gamma_ols_min.min(), 1), f(fits.gamma_ols_max.max(), 1)
    N["ebMin"], N["ebMax"] = f(fits.gamma_min.min(), 2), f(fits.gamma_max.max(), 2)
    N["fitRmseMin"], N["fitRmseMax"] = f(fits.fit_rmse_db.min(), 1), f(fits.fit_rmse_db.max(), 1)
    N["shrinkMedian"] = f(fits.shrink_median.median(), 2)
    N["tauMin"], N["tauMax"] = f(fits.tau.min(), 2), f(fits.tau.max(), 2)
    N["cityGammaMin"] = f(fits.gamma_global.min(), 2)
    N["cityGammaMinCity"] = nice(fits.city[fits.gamma_global.idxmin()])
    N["nCityGammaBelowTwo"] = int((fits.gamma_global < 2).sum())
    N["medianSitesPerTile"] = f(fits.median_n_bs.median(), 0)

    from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS, DESCRIPTOR_NAMES
    order = fits.sort_values("gamma_median").city.tolist()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw={"width_ratios": [1.15, 1]})
    data = [tiles[tiles.city == c].gamma.dropna().to_numpy() for c in order]
    bp = ax[0].boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                       medianprops=dict(color="k", lw=1.0), whiskerprops=dict(lw=0.6),
                       capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for b in bp["boxes"]:
        b.set_facecolor("0.85")
    ax[0].set_xticks(range(1, len(order) + 1))
    ax[0].set_xticklabels([nice(c) for c in order], rotation=45, ha="right")
    ax[0].set_ylabel(r"tile exponent $\gamma$")
    ax[0].set_title("(a)", loc="left")
    D = tiles[DESCRIPTOR_COLUMNS].to_numpy(float)
    g = tiles.gamma.to_numpy()
    live = [j for j in range(D.shape[1]) if D[:, j].std() > 0]
    r = np.array([np.corrcoef(D[:, j], g)[0, 1] for j in live])
    top = np.argsort(-np.abs(r))[:10]
    y = np.arange(len(top))
    ax[1].barh(y, r[top], color=["0.25" if v > 0 else "0.7" for v in r[top]],
               edgecolor="k", lw=0.4)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([desc_name(DESCRIPTOR_NAMES[live[j]]) for j in top])
    ax[1].invert_yaxis()
    ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_xlabel(r"Pearson $r$ with $\gamma$ (all tiles)")
    ax[1].set_title("(b)", loc="left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(fig_dir / "fig_corpus.pdf")
    plt.close(fig)
    N["nDescLive"] = len(live)
    N["topDesc"] = desc_name(DESCRIPTOR_NAMES[live[top[0]]], tex=True)
    N["topDescR"] = f"{r[top[0]]:+.2f}"
    N["maxAbsR"] = f(np.abs(r).max(), 2)


# ----------------------------------------------------------------------------
def loco(T, G, N, fig_dir):
    m = pd.concat([pd.read_csv(T / "loco_metrics.csv"), pd.read_csv(T / "neural_metrics.csv")])
    hp = pd.read_csv(T / "loco_hparams.csv")
    lh = pd.read_csv(T / "link_hparams.csv").set_index("fold")
    per = m.groupby(["fold", "operator"])[["gamma_rmse", "pl0_rmse", "gamma_pearson"]].mean()
    g = per.gamma_rmse.unstack()[[o for o in OPS if o in m.operator.unique()]]
    p = per.pl0_rmse.unstack()[g.columns]
    ops = list(g.columns)
    rows = []
    for city in g.index:
        best = g.loc[city].idxmin()
        rows.append(nice(city) + " & " + " & ".join(
            (rf"\textbf{{{g.loc[city, o]:.3f}}}" if o == best else f"{g.loc[city, o]:.3f}")
            for o in ops) + r" \\")
    mean, sd = g.mean(), g.std(ddof=1)
    bm = mean.idxmin()
    save_table(G, "tab_loco", "\n".join([
        r"\begin{tabular}{l" + "r" * len(ops) + "}", r"\toprule",
        "Target city & " + " & ".join(SHORT[o] for o in ops) + r" \\", r"\midrule", *rows,
        r"\midrule",
        "Mean & " + " & ".join((rf"\textbf{{{mean[o]:.3f}}}" if o == bm else f"{mean[o]:.3f}")
                               for o in ops) + r" \\",
        "S.d. & " + " & ".join(f"{sd[o]:.3f}" for o in ops) + r" \\",
        r"\midrule",
        r"$P_{L_0}$ (dB) & " + " & ".join(f"{p[o].mean():.1f}" for o in ops) + r" \\",
        r"\bottomrule", r"\end{tabular}"]) + "\n")
    for o in ops:
        k = MACRO[o]
        N[f"rmse{k}"] = f(mean[o])
        N[f"sd{k}"] = f(sd[o])
        N[f"plRmse{k}"] = f(p[o].mean(), 1)
    N["bestZeroShot"] = OP_NAME[bm]
    N["gainGbdtMedian"] = pct(1 - mean["gbdt"] / mean["median"])
    N["gainKernelMedian"] = pct(1 - mean["kernel"] / mean["median"])
    const = ["uma_los", "uma_nlos", "cost231"]
    learned = g.drop(columns=[c for c in const if c in g])
    hard = learned.min(axis=1)
    N["hardCity"] = nice(hard.idxmax())
    N["hardCityBest"] = f(hard.max())
    N["hardCityBestOp"] = OP_NAME[learned.loc[hard.idxmax()].idxmin()]
    N["sigmaMin"], N["sigmaMax"] = f"{hp.sigma.min():g}", f"{hp.sigma.max():g}"
    N["hBsMedian"] = f(hp.h_bs_m.median(), 0)
    N["iwEssMin"], N["iwEssMax"] = pct((hp.iw_ess / hp.n_train).min(), 0), \
        pct((hp.iw_ess / hp.n_train).max(), 0)
    N["nBetterThanMedian"] = int((mean.drop("median") < mean["median"]).sum())
    N["nOps"] = len(ops)
    N["knnKMedian"] = f"{hp.knn_k.median():g}"
    N["nSigmaLargest"] = int((hp.sigma == hp.sigma.max()).sum())
    N["linkTreesMedian"] = f"{lh.trees_reg.median():.0f}"
    N["linkClfTreesMedian"] = f"{lh.trees_clf.median():.0f}"

    rows = [f"{nice(r.fold)} & {thou(r.n_train)} & {r.sigma:g} & {r.knn_k} & {r.ridge_alpha:g} & "
            f"{r.mlp_alpha:g} & {r.mlp_epochs} & {r.gbdt_trees_gamma} & "
            f"{lh.loc[r.fold, 'trees_reg']} & {lh.loc[r.fold, 'trees_clf']} \\\\"
            for r in hp.itertuples()]
    save_table(G, "tab_hparams", tabular("lrrrrrrrrr", [
        r"& & & & & \multicolumn{2}{c}{MLP} & GBDT & \multicolumn{2}{c}{Link model} \\",
        r"\cmidrule(lr){6-7}\cmidrule(lr){9-10}",
        r"Target & $N$ & $\sigma$ & $k$ & $\alpha_{\mathrm{ridge}}$ & $\alpha$ & epochs & "
        r"rounds & rounds & rounds (clf.) \\"], rows))

    fig, ax = plt.subplots(figsize=(W1, 2.4))
    show = [o for o in ["median", "kernel", "gp_ard", "knn", "ridge", "mlp", "encoder", "gbdt"]
            if o in ops]
    order = g["median"].sort_values().index
    for i, o in enumerate(show):
        ax.plot(g.loc[order, o].to_numpy(), np.arange(len(order)) + (i - len(show) / 2) * 0.09,
                MARK[i], ms=3.2, mfc="none" if i % 2 else "k", mec="k", mew=0.6,
                ls="none", label=SHORT[o])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([nice(c) for c in order])
    ax.set_xlabel(r"zero-shot $\gamma$ RMSE")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_loco.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
SHARE_METRICS = {"assoc_acc", "regret", "cov_acc_130"}


def scaled(metric, *vals):
    """Differences of shares in percentage points (1 decimal), of dB quantities to
    2 decimals, of exponent errors to 3 decimals."""
    if metric in SHARE_METRICS:
        return tuple(fs(100 * v, 1) for v in vals)
    if metric.endswith("_db"):
        return tuple(fs(v, 2) for v in vals)
    return tuple(fs(v, 3) for v in vals)


def stats(T, G, N):
    s = pd.read_csv(T / "stats.csv")
    label = {**OP_NAME, **{f"{o}@k10": f"{OP_NAME[o]}, $k{{=}}10$" for o in FS_OPS},
             **{f"{o}@k0": f"{OP_NAME[o]}, $k{{=}}0$" for o in FS_OPS},
             **{f"{o}@k10d": f"{OP_NAME[o]}, $k{{=}}10$ designed" for o in FS_OPS},
             **{f"{o}@k10bias": f"{OP_NAME[o]}, $k{{=}}10$ plain bias" for o in FS_OPS}}
    metric = {"gamma_rmse": r"$\gamma$ RMSE", "pl_rmse_db": "path-loss RMSE (dB)",
              "sinr_mae_db": "SINR MAE (dB)", "cov_acc_130": "coverage agreement",
              "assoc_acc": "serving site", "regret": "regret, $K{=}5$"}
    fam_name = {"zero_shot": "Zero-shot vs source median", "pairs": "Zero-shot pairs",
                "few_shot": "Few-shot correction", "network": "Network level",
                "decomp": "Decomposition and physics baselines",
                "band2": "Network level, second band"}

    def table(fams, name):
        rows = []
        for fam in [f_ for f_ in fams if f_ in set(s.family)]:
            sub = s[s.family == fam]
            rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{fam_name[fam]}}}}} \\")
            for r in sub.itertuples():
                d, lo, hi = scaled(r.metric, r.diff, r.ci_lo, r.ci_hi)
                rows.append(f"{label.get(r.a, r.a)} vs {label.get(r.b, r.b)} & "
                            f"{metric[r.metric]} & {d} & [{lo}, {hi}] & "
                            f"{pval(r.p_wilcoxon)} & {r.n_better}/{r.n_folds} & "
                            f"{'yes' if r.holm_reject else 'no'} \\\\")
        save_table(G, name, tabular("llrrrrc", [
            r"Comparison & Metric & $\Delta$ & 95\% CI & $p$ & better & Holm \\"], rows))
    table(["zero_shot", "pairs", "few_shot"], "tab_stats_param")
    table(["network", "decomp", "band2"], "tab_stats_net")

    def get(a, b, mtr, family):
        q = s[(s.a == a) & (s.b == b) & (s.metric == mtr) & (s.family == family)]
        return q.iloc[0] if len(q) else None
    keys = {"GbdtMedian": ("gbdt", "median", "gamma_rmse", "zero_shot"),
            "KernelMedian": ("kernel", "median", "gamma_rmse", "zero_shot"),
            "RidgeMedian": ("ridge", "median", "gamma_rmse", "zero_shot"),
            "MlpMedian": ("mlp", "median", "gamma_rmse", "zero_shot"),
            "GpMedian": ("gp_ard", "median", "gamma_rmse", "zero_shot"),
            "EncMedian": ("encoder", "median", "gamma_rmse", "zero_shot"),
            "GbdtKernel": ("gbdt", "kernel", "gamma_rmse", "pairs"),
            "GpKernel": ("gp_ard", "kernel", "gamma_rmse", "pairs"),
            "IwGbdt": ("gbdt_iw", "gbdt", "gamma_rmse", "pairs"),
            "CoralRidge": ("coral", "ridge", "gamma_rmse", "pairs"),
            "FewGbdt": ("gbdt@k10", "gbdt@k0", "gamma_rmse", "few_shot"),
            "FewRidge": ("ridge@k10", "ridge@k0", "gamma_rmse", "few_shot"),
            "FewDesignGbdt": ("gbdt@k10d", "gbdt@k10", "gamma_rmse", "few_shot"),
            "FewEbGbdt": ("gbdt@k10", "gbdt@k10bias", "gamma_rmse", "few_shot"),
            "NetGbdtMedian": ("gbdt", "median", "pl_rmse_db", "network"),
            "NetLinkGbdt": ("link", "gbdt", "pl_rmse_db", "network"),
            "NetLinkOracle": ("link", "oracle", "pl_rmse_db", "network"),
            "NetLinkAssoc": ("link", "gbdt", "assoc_acc", "network"),
            "NetLinkOracleAssoc": ("link", "oracle", "assoc_acc", "network"),
            "NetLinkRegret": ("link", "gbdt", "regret", "network"),
            "NetLinkOracleRegret": ("link", "oracle", "regret", "network"),
            "NetLinkSinr": ("link", "gbdt", "sinr_mae_db", "network"),
            "NetLinkCov": ("link", "gbdt", "cov_acc_130", "network"),
            "NetLinkTwentyFive": ("link+k25", "link", "pl_rmse_db", "network"),
            "NetFixedKappa": ("link+k25f", "link+k25", "pl_rmse_db", "network"),
            "NetDesignVar": ("link+k25d", "link+k25", "pl_rmse_db", "network"),
            "NetDesignStrat": ("link+k25s", "link+k25", "pl_rmse_db", "network"),
            "DecHurdle": ("gbdt+h", "gbdt", "pl_rmse_db", "decomp"),
            "DecHurdleRegret": ("gbdt+h", "gbdt", "regret", "decomp"),
            "DecLos": ("gbdt+los", "gbdt", "pl_rmse_db", "decomp"),
            "DecLosAssoc": ("gbdt+los", "gbdt", "assoc_acc", "decomp"),
            "DecLosH": ("gbdt+los+h", "gbdt+h", "pl_rmse_db", "decomp"),
            "DecLosHRegret": ("gbdt+los+h", "gbdt+h", "regret", "decomp"),
            "DecLinkBestTile": ("link", "gbdt+los+h", "pl_rmse_db", "decomp"),
            "DecLinkBestTileAssoc": ("link", "gbdt+los+h", "assoc_acc", "decomp"),
            "DecLinkBestTileRegret": ("link", "gbdt+los+h", "regret", "decomp"),
            "DecLinkOracleLosH": ("link", "oracle+los+h", "pl_rmse_db", "decomp"),
            "DecLinkOracleLosHAssoc": ("link", "oracle+los+h", "assoc_acc", "decomp"),
            "DecLinkOracleLosHRegret": ("link", "oracle+los+h", "regret", "decomp"),
            "DecLinkKe": ("link", "fspl_ke", "pl_rmse_db", "decomp"),
            "DecLinkKeAssoc": ("link", "fspl_ke", "assoc_acc", "decomp"),
            "DecLinkDeygout": ("link", "deygout", "pl_rmse_db", "decomp"),
            "DecLinkDeygoutAssoc": ("link", "deygout", "assoc_acc", "decomp"),
            "DecLinkDeygoutRegret": ("link", "deygout", "regret", "decomp"),
            "DecLinkUma": ("link", "uma_geo", "pl_rmse_db", "decomp"),
            "DecLinkUmaAssoc": ("link", "uma_geo", "assoc_acc", "decomp"),
            "DecLinkUmaRegret": ("link", "uma_geo", "regret", "decomp"),
            "BandLinkGbdt": ("link", "gbdt", "pl_rmse_db", "band2"),
            "BandLinkOracle": ("link", "oracle", "pl_rmse_db", "band2"),
            "BandCross": ("link_x_phys", "link", "pl_rmse_db", "band2"),
            "BandCrossAssoc": ("link_x_phys", "link", "assoc_acc", "band2"),
            "BandCrossRegret": ("link_x_phys", "link", "regret", "band2")}
    for key, (a, b, mtr, fam) in keys.items():
        r = get(a, b, mtr, fam)
        if r is None:
            continue
        N[f"d{key}"], N[f"lo{key}"], N[f"hi{key}"] = scaled(mtr, r["diff"], r.ci_lo, r.ci_hi)
        N[f"p{key}"] = pval(r.p_wilcoxon)
        N[f"n{key}"] = f"{r.n_better}"
        N[f"holm{key}"] = "significant" if r.holm_reject else "not significant"
    N["nFolds"] = int(s.n_folds.max())
    zs = s[s.family == "zero_shot"]
    N["nZeroShotTotal"] = len(zs)
    better = zs[zs.holm_reject & (zs["diff"] < 0)]
    worse = zs[zs.holm_reject & (zs["diff"] > 0)]
    N["nZeroShotSigBetter"] = len(better)
    N["zeroShotSigBetterList"] = ", ".join(label.get(a, a) for a in better.a) or "none"
    N["zeroShotSigBetterParen"] = (" (" + ", ".join(label.get(a, a) for a in better.a) + ")"
                                   if len(better) else "")
    N["nZeroShotSigWorse"] = len(worse)
    N["zeroShotSigWorseList"] = ", ".join(label.get(a, a) for a in worse.a) or "none"
    N["zeroShotSigBetterWord"] = ("none" if len(better) == 0 else
                                  "one" if len(better) == 1 else str(len(better)))
    nb = zs[zs["diff"] < 0].n_better
    N["learnedBetterMin"], N["learnedBetterMax"] = int(nb.min()), int(nb.max())
    N["nStatsRows"] = len(s)
    dc = s[(s.family == "decomp") & (s.a == "link")]
    N["nDecompLinkAllCities"] = int((dc.n_better == dc.n_folds).sum())
    N["nDecompLinkTotal"] = len(dc)
    N["nDecompLinkHolm"] = int(dc.holm_reject.sum())


# ----------------------------------------------------------------------------
def degeneracy(T, G, N):
    d = pd.read_csv(T / "degeneracy.csv")
    rows = [f"{nice(r.fold)} & {r.sigma:g} & {r.eps_median:.3g} & {pct(r.frac_eps_lt1, 0)} & "
            f"{100 * r.neff_frac:.2f} & {r.wmax_median:.3f} & {r.pred_spread_ratio:.2f} \\\\"
            for r in d.itertuples()]
    save_table(G, "tab_degeneracy", tabular("lrrrrrr", [
        r"Target & $\sigma$ & med.\ $\epsilon_u$ & $\epsilon_u{<}1$ (\%) & "
        r"$n_{\mathrm{eff}}/N$ (\%) & med.\ $w_{\max}$ & spread \\"], rows))
    N["epsMedianMin"], N["epsMedianMax"] = f"{d.eps_median.min():.3g}", f"{d.eps_median.max():.3g}"
    N["fracEpsRegime"] = pct(d.frac_eps_lt1.mean(), 1)
    N["neffFracMax"] = f"{100 * d.neff_frac.max():.0f}"
    N["wmaxMedian"] = f(d.wmax_median.median(), 2)
    N["spreadMin"] = f(d.pred_spread_ratio.min(), 2)
    N["spreadMax"] = f(d.pred_spread_ratio.max(), 2)
    N["spreadMedian"] = f(d.pred_spread_ratio.median(), 2)


# ----------------------------------------------------------------------------
def fewshot(T, G, N, fig_dir):
    k = pd.read_csv(T / "kshot.csv")

    def mean_table(mode, sampling):
        q = k[(k["mode"] == mode) & (k.sampling == sampling)]
        return q.groupby(["operator", "fold", "k"]).gamma_rmse.mean() \
            .groupby(["operator", "k"]).mean().unstack(0)
    tab = mean_table("eb", "random")[FS_OPS]
    bias = mean_table("bias", "random")
    resid = mean_table("residual", "random")
    patch = mean_table("eb", "patch")
    design = mean_table("eb", "design")
    rows = []
    for kk in tab.index:
        rows.append(f"{kk} & " + " & ".join(f"{tab.loc[kk, o]:.3f}" for o in FS_OPS)
                    + f" & {bias.loc[kk, 'gbdt']:.3f} & {resid.loc[kk, 'gbdt']:.3f}"
                    + f" & {patch.loc[kk, 'gbdt']:.3f} & {design.loc[kk, 'gbdt']:.3f} \\\\")
    save_table(G, "tab_kshot", tabular("rrrrrrrrr", [
        r"& \multicolumn{4}{c}{EB correction, random tiles} & "
        r"\multicolumn{4}{c}{GBDT} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"$k$ & Kernel & Ridge & MLP & GBDT & plain bias & residual & contig. & design \\"], rows))
    N["fsGbdtDesignTen"] = f(design.loc[10, "gbdt"])
    dbetter = [kk for kk in design.index if kk > 0 and design.loc[kk, "gbdt"] < tab.loc[0, "gbdt"]]
    N["fsDesignFirstBetterK"] = int(dbetter[0]) if dbetter else "--"
    N["fsDesignEverBetter"] = "yes" if dbetter else "no"
    best0, best10 = tab.loc[0].idxmin(), tab.loc[10].idxmin()
    N["fsBestZero"], N["fsBestTen"] = OP_NAME[best0], OP_NAME[best10]
    for o, key in zip(FS_OPS, ["Kernel", "Ridge", "Mlp", "Gbdt"], strict=True):
        N[f"fsZero{key}"], N[f"fsTen{key}"] = f(tab.loc[0, o]), f(tab.loc[10, o])
        N[f"fsMax{key}"] = f(tab.loc[tab.index.max(), o])
        N[f"fsGain{key}"] = pct(1 - tab.loc[10, o] / tab.loc[0, o])
        N[f"fsGainMax{key}"] = pct(1 - tab.loc[tab.index.max(), o] / tab.loc[0, o])
    N["fsKmax"] = int(tab.index.max())
    better = [kk for kk in tab.index if kk > 0 and tab.loc[kk, "gbdt"] < tab.loc[0, "gbdt"]]
    N["fsFirstBetterK"] = int(better[0]) if better else "--"
    N["fsFirstBetterGbdt"] = f(tab.loc[better[0], "gbdt"]) if better else "--"
    N["fsGbdtPatchTen"] = f(patch.loc[10, "gbdt"])
    N["fsGbdtBiasTen"] = f(bias.loc[10, "gbdt"])
    N["fsGbdtBiasThree"] = f(bias.loc[3, "gbdt"])
    N["fsGbdtEbThree"] = f(tab.loc[3, "gbdt"])
    N["fsGbdtResidTen"] = f(resid.loc[10, "gbdt"])
    N["fsGbdtResidThree"] = f(resid.loc[3, "gbdt"])
    worse_plain = [kk for kk in bias.index if kk > 0 and bias.loc[kk, "gbdt"] > tab.loc[0, "gbdt"]]
    N["fsPlainWorseKs"] = ", ".join(str(int(v)) for v in worse_plain) or "none"
    worse_eb = [kk for kk in tab.index if kk > 0 and tab.loc[kk, "gbdt"] > tab.loc[0, "gbdt"]]
    N["fsEbWorseKs"] = ", ".join(str(int(v)) for v in worse_eb) or "none"
    N["fsEbNeverWorse"] = "at no value of $k$" if not worse_eb else "at $k=" + ", ".join(
        str(int(v)) for v in worse_eb) + "$"
    pc = k[(k["mode"] == "eb") & (k.sampling == "random") & (k.operator == "gbdt")] \
        .groupby(["fold", "k"]).gamma_rmse.mean().unstack()
    N["fsImprovedTen"] = int((pc[10] < pc[0]).sum())
    N["fsImprovedFifty"] = int((pc[50] < pc[0]).sum())
    N["fsSdZero"], N["fsSdTen"] = f(pc[0].std(ddof=1)), f(pc[10].std(ddof=1))
    rows = [f"{nice(c)} & {pc.loc[c, 0]:.3f} & {pc.loc[c, 10]:.3f} & "
            f"{pc.loc[c, 50] if 50 in pc else float('nan'):.3f} & "
            f"{100 * (pc.loc[c, 10] / pc.loc[c, 0] - 1):+.0f} \\\\" for c in pc.index]
    save_table(G, "tab_kshot_city", tabular("lrrrr", [
        r"Target & $k{=}0$ & $k{=}10$ & $k{=}50$ & $\Delta_{10}$ (\%) \\"], [*rows, r"\midrule",
        f"Mean & {pc[0].mean():.3f} & {pc[10].mean():.3f} & {pc[50].mean():.3f} & \\\\",
        f"S.d. & {pc[0].std(ddof=1):.3f} & {pc[10].std(ddof=1):.3f} & "
        f"{pc[50].std(ddof=1):.3f} & \\\\"]))

    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    for i, o in enumerate(FS_OPS):
        ax[0].plot(tab.index, tab[o], marker=MARK[i], ms=3, color="k",
                   mfc="none" if i % 2 else "k", ls=["-", "--", ":", "-."][i], label=OP_NAME[o])
    ax[0].plot(bias.index, bias["gbdt"], color="0.55", marker="x", ms=3, ls="-",
               label="GBDT, plain bias")
    ax[0].plot(patch.index, patch["gbdt"], color="0.55", marker="+", ms=3, ls="--",
               label="GBDT, contiguous tiles")
    ax[0].set_xscale("symlog", linthresh=3)
    ax[0].set_xticks(tab.index)
    ax[0].set_xticklabels([str(int(v)) for v in tab.index])
    ax[0].set_xlabel("surveyed target tiles $k$")
    ax[0].set_ylabel(r"$\gamma$ RMSE (unsurveyed tiles)")
    lo_, hi_ = ax[0].get_ylim()
    ax[0].set_ylim(lo_, hi_ + 0.6 * (hi_ - lo_))
    ax[0].legend(frameon=False, ncol=2, loc="upper center")
    ax[0].set_title("(a)", loc="left")
    order = pc[0].sort_values(ascending=False).index
    x = np.arange(len(order))
    for j, (kk, col) in enumerate([(0, "0.2"), (10, "0.6"), (50, "0.9")]):
        if kk in pc:
            ax[1].bar(x + (j - 1) * 0.27, pc.loc[order, kk], 0.27, color=col, edgecolor="k",
                      lw=0.4, label=f"$k={kk}$")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([nice(c) for c in order], rotation=45, ha="right")
    ax[1].set_ylabel(r"$\gamma$ RMSE, GBDT")
    ax[1].set_ylim(0, 1.3 * ax[1].get_ylim()[1])
    ax[1].legend(frameon=False, ncol=3, loc="upper center")
    ax[1].set_title("(b)", loc="left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(fig_dir / "fig_kshot.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
def label_noise(T, G, N):
    d = pd.read_csv(T / "label_noise.csv")
    rows = [f"{nice(r.city)} & {r.median_n_bs:.0f} & {r.gamma_se_analytic:.3f} & "
            f"{r.gamma_se_cluster:.3f} & {r.gamma_se_split_full:.3f} & {r.pl0_se_analytic:.2f} & "
            f"{r.pl0_se_cluster:.2f} & {r.pl0_se_split_full:.2f} \\\\" for r in d.itertuples()]

    def rms(col):
        return float(np.sqrt((d[col] ** 2).mean()))
    save_table(G, "tab_noise", tabular("lrrrrrrr", [
        r"& & \multicolumn{3}{c}{$\mathrm{se}(\gamma)$} & "
        r"\multicolumn{3}{c}{$\mathrm{se}(P_{L_0})$ (dB)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"City & sites/tile & iid & cluster & split & iid & cluster & split \\"], [
        *rows, r"\midrule",
        f"RMS & & {rms('gamma_se_analytic'):.3f} & {rms('gamma_se_cluster'):.3f} & "
        f"{rms('gamma_se_split_full'):.3f} & {rms('pl0_se_analytic'):.2f} & "
        f"{rms('pl0_se_cluster'):.2f} & {rms('pl0_se_split_full'):.2f} \\\\"]))
    a, c, s = rms("gamma_se_analytic"), rms("gamma_se_cluster"), rms("gamma_se_split_full")
    N["seIid"], N["seCluster"], N["seSplit"] = f(a), f(c), f(s)
    N["seRatio"] = f(c / a, 1)
    N["seSplitHalf"] = f(rms("gamma_se_half"))
    N["seSplitDiff"] = f(np.sqrt(2) * rms("gamma_se_half"), 2)
    N["plSeIid"], N["plSeCluster"] = f(rms("pl0_se_analytic"), 2), f(rms("pl0_se_cluster"), 2)


# ----------------------------------------------------------------------------
def network(T, G, N, fig_dir):
    n, pl, per = nets(T)
    mean = per.groupby("operator").mean()
    show = [o for o in ["oracle", "median", "uma_nlos", "cost231", "kernel", "gp_ard", "knn",
                        "ridge", "coral", "mlp", "encoder", "gbdt", "gbdt_iw", "gbdt+k10",
                        "gbdt+k10d", "link", "link_reg", "link+k25", "link+k25s"]
            if o in set(n.operator)]
    rows = []
    for o in show:
        m = mean.loc[o]
        if o == "link":
            rows.append(r"\midrule")
        rows.append(f"{OP_NAME[o]} & {m.pl_rmse_db:.2f} & {fs(m.pl_bias_db)} & "
                    f"{m.pl_rmse_cens_db:.2f} & {pct(m.assoc_acc)} & {pct(m.cov_acc_130)} & "
                    f"{pct(m.cov_err_130)} & {m.sinr_mae_db:.2f} & {m.sinr_ks:.3f} & "
                    f"{fs(m.se_err, 3)} & {pct(m.regret)} \\\\")
        if o == "oracle":
            rows.append(r"\midrule")
    save_table(G, "tab_network", tabular("lrrrrrrrrrr", [
        r"& \multicolumn{3}{c}{Path loss (dB)} & Serving & "
        r"\multicolumn{2}{c}{Coverage 130\,dB (\%)} & "
        r"\multicolumn{2}{c}{SINR} & SE & Regret \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Predictor & RMSE & bias & cens. & site (\%) & agree & $\Delta$ & MAE (dB) & KS & "
        r"$\Delta$ & $K{=}5$ (\%) \\"], rows))
    keys = [("oracle", "Oracle"), ("median", "Median"), ("kernel", "Kernel"),
            ("ridge", "Ridge"), ("gbdt", "Gbdt"), ("gbdt+k10", "GbdtTen"),
            ("gbdt+k10d", "GbdtTenD"), ("uma_nlos", "UmaNlos"), ("cost231", "Cost"),
            ("gp_ard", "GpArd"), ("link", "Link"), ("link_reg", "LinkReg"),
            ("link+k25", "LinkTwentyFive"), ("link+k25s", "LinkTwentyFiveS")]
    for o, key in keys:
        if o in mean.index:
            N[f"net{key}"] = f(mean.loc[o, "pl_rmse_db"], 1)
            N[f"cens{key}"] = f(mean.loc[o, "pl_rmse_cens_db"], 1)
            N[f"assoc{key}"] = pct(mean.loc[o, "assoc_acc"])
            N[f"cov{key}"] = pct(mean.loc[o, "cov_acc_130"])
            N[f"sinr{key}"] = f(mean.loc[o, "sinr_mae_db"], 1)
            N[f"regret{key}"] = pct(mean.loc[o, "regret"])
    N["covTrue"] = pct(mean.loc["oracle", "cov_true_130"])
    N["covTrueOneForty"] = pct(mean.loc["oracle", "cov_true_140"])
    N["seTrue"] = f(mean.loc["oracle", "se_true"], 2)
    N["unreachedShare"] = pct(mean.loc["oracle", "n_unreached"] / mean.loc["oracle", "n_pixels"])
    ur = per.loc["oracle"].n_unreached / per.loc["oracle"].n_pixels
    N["unreachedMin"], N["unreachedMax"] = pct(ur.min()), pct(ur.max())
    N["gainLinkOracle"] = f(mean.loc["oracle", "pl_rmse_db"] - mean.loc["link", "pl_rmse_db"], 1)
    N["gainLinkGbdt"] = f(mean.loc["gbdt", "pl_rmse_db"] - mean.loc["link", "pl_rmse_db"], 1)
    lk = per.loc["link"].pl_rmse_db
    N["linkMin"], N["linkMax"] = f(lk.min(), 1), f(lk.max(), 1)
    N["linkMinCity"], N["linkMaxCity"] = nice(lk.idxmin()), nice(lk.idxmax())
    # hurdle quality
    rows = []
    for o in ["oracle", "gbdt", "link", "link_reg"]:
        m = mean.loc[o]
        rows.append(f"{OP_NAME[o]} & {pct(m.reach_acc)} & {pct(m.link_precision)} & "
                    f"{pct(m.link_recall)} & {m.pl_rmse_cens_db:.2f} \\\\")
    save_table(G, "tab_hurdle", tabular("lrrrr", [
        r"Predictor & reach (\%) & precision (\%) & recall (\%) & cens.\ RMSE (dB) \\"],
        rows))
    for o, key in [("link", "Link"), ("gbdt", "Gbdt"), ("link_reg", "LinkReg")]:
        N[f"reach{key}"] = pct(mean.loc[o, "reach_acc"])
        N[f"linkPrec{key}"] = pct(mean.loc[o, "link_precision"])
        N[f"linkRec{key}"] = pct(mean.loc[o, "link_recall"])
    # per-city table of the link model against GBDT and the oracle
    rows = []
    for c in sorted(per.loc["link"].index):
        a, b, o = per.loc[("link", c)], per.loc[("gbdt", c)], per.loc[("oracle", c)]
        rows.append(f"{nice(c)} & {o.pl_rmse_db:.2f} & {b.pl_rmse_db:.2f} & {a.pl_rmse_db:.2f} & "
                    f"{pct(o.assoc_acc)} & {pct(a.assoc_acc)} & {pct(o.regret)} & "
                    f"{pct(b.regret)} & {pct(a.regret)} \\\\")
    save_table(G, "tab_link_city", tabular("lrrrrrrrr", [
        r"& \multicolumn{3}{c}{Path loss (dB)} & \multicolumn{2}{c}{Serving (\%)} & "
        r"\multicolumn{3}{c}{Regret $K{=}5$ (\%)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
        r"City & oracle & GBDT & link & oracle & link & oracle & GBDT & link \\"], rows))
    lb = per.loc["link"]
    N["nLinkBetterGbdtPl"] = int((lb.pl_rmse_db < per.loc["gbdt"].pl_rmse_db).sum())
    N["nLinkBetterOraclePl"] = int((lb.pl_rmse_db < per.loc["oracle"].pl_rmse_db).sum())
    N["nLinkBetterOracleRegret"] = int((lb.regret < per.loc["oracle"].regret).sum())
    mobility(n, pd.concat([pd.read_csv(T / "planning.csv"), pd.read_csv(T / "planning_link.csv")]),
             G, N)
    survey(n, pl, G, N)
    density(T, G, N)

    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    ops = [o for o in ["oracle", "median", "kernel", "ridge", "gbdt", "link", "link+k25"]
           if o in mean.index]
    y = np.arange(len(ops))
    shade = {"oracle": "0.35", "link": "0.1", "link+k25": "0.1"}
    ax[0].barh(y, [mean.loc[o, "pl_rmse_db"] for o in ops],
               color=[shade.get(o, "0.8") for o in ops], edgecolor="k", lw=0.4)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([OP_NAME[o] for o in ops])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("path-loss RMSE on the target city (dB)")
    ax[0].set_title("(a)", loc="left")
    reg = pl.groupby(["operator", "K", "fold"]).regret.mean().groupby(["operator", "K"]).mean()
    pl_all = pd.concat([pd.read_csv(T / "planning.csv"), pd.read_csv(T / "planning_link.csv")])
    pl_all = pl_all[(pl_all.sites == pl_all.sites.max()) & (pl_all.K > 0)]
    reg = pl_all.groupby(["operator", "K", "fold"]).regret.mean().groupby(["operator", "K"]).mean()
    for i, o in enumerate([o for o in ["oracle", "median", "kernel", "gbdt", "link"]
                           if o in mean.index]):
        r = reg.loc[o]
        ax[1].plot(r.index, 100 * r.to_numpy(), marker=MARK[i], ms=3, color="k",
                   mfc="none" if i % 2 else "k", ls=["-", "--", ":", "-.", "-"][i],
                   lw=1.8 if o == "link" else 0.9, label=OP_NAME[o])
    ax[1].set_xlabel("sites selected $K$")
    ax[1].set_ylabel("planning regret (% of pixels)")
    ax[1].set_ylim(0, 1.45 * ax[1].get_ylim()[1])
    ax[1].legend(frameon=False, ncol=2, loc="upper center")
    ax[1].set_title("(b)", loc="left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(fig_dir / "fig_network.pdf")
    plt.close(fig)

    imp = pd.read_csv(T / "link_importance.csv")
    top = imp.head(10)
    rows = [f"{feat_name(r.feature)} & {pct(r.gain_share)} \\\\" for r in top.itertuples()]
    save_table(G, "tab_importance", tabular("lr", [r"Feature & gain share (\%) \\"], rows))
    link_share = imp[~imp.feature.str.startswith("d_")].gain_share.sum()
    N["linkFeatShare"] = pct(link_share, 0)
    N["topLinkFeat"] = feat_name(imp.feature.iloc[0])
    N["topLinkFeatShare"] = pct(imp.gain_share.iloc[0], 0)
    N["topThreeLinkShare"] = pct(imp.gain_share.iloc[:3].sum(), 0)


def mobility(n, pl, G, N):
    """Cell edges and dimensioning (sites for the coverage target)."""
    from terranet.evaluation.network import COV_TARGET
    pl = pl[pl.sites == pl.sites.max()]
    d = pl[pl.K == -1].groupby(["operator", "fold"])[
        ["sites_pred", "sites_true", "cov_true_of_pred_plan"]].mean().groupby("operator").mean()
    e = n.groupby(["operator", "fold"])[["edge_disp_m", "handover_err"]].mean() \
        .groupby("operator").mean()
    show = [o for o in ["oracle", "median", "uma_nlos", "kernel", "gp_ard", "ridge", "gbdt",
                        "gbdt+k10", "link", "link_reg", "link+k25"] if o in e.index]
    rows = []
    for o in show:
        rows.append(f"{OP_NAME[o]} & {e.loc[o, 'edge_disp_m']:.0f} & "
                    f"{100 * e.loc[o, 'handover_err']:+.0f} & {d.loc[o, 'sites_pred']:.1f} & "
                    f"{pct(d.loc[o, 'cov_true_of_pred_plan'])} \\\\")
        if o in ("oracle", "gbdt+k10"):
            rows.append(r"\midrule")
    save_table(G, "tab_mobility", tabular("lrrrr", [
        r"& \multicolumn{2}{c}{Cell edges} & \multicolumn{2}{c}{Plan for "
        + pct(COV_TARGET, 0) + r"\%} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Predictor & disp.\ (m) & length (\%) & sites & true cov.\ (\%) \\"], [
        *rows, r"\midrule",
        f"Truth & 0 & 0 & {d.loc['oracle', 'sites_true']:.1f} & {pct(COV_TARGET, 0)}+ \\\\"]))
    N["covTarget"] = pct(COV_TARGET, 0)
    for o, key in [("oracle", "Oracle"), ("median", "Median"), ("gbdt", "Gbdt"),
                   ("link", "Link"), ("uma_nlos", "UmaNlos"), ("link_reg", "LinkReg")]:
        N[f"edge{key}"] = f"{e.loc[o, 'edge_disp_m']:.0f}"
        N[f"edgeLen{key}"] = f"{100 * e.loc[o, 'handover_err']:+.0f}"
        v = d.loc[o, "sites_pred"]
        N[f"sitesNeed{key}"] = "a single site" if round(v, 1) == 1.0 else f"{v:.1f} sites"
        N[f"sitesCov{key}"] = pct(d.loc[o, "cov_true_of_pred_plan"])
    N["sitesNeedTrue"] = f"{d.loc['oracle', 'sites_true']:.1f}"


def survey(n, pl, G, N):
    """Few-shot site calibration of the link model: random tiles with EB or fixed
    kappa, and the two designs; unsurveyed pixels only."""
    reg = pl.groupby(["operator", "fold"]).regret.mean().groupby("operator").mean()
    m = n.groupby(["operator", "fold"])[["pl_rmse_db", "assoc_acc", "kappa"]].mean() \
        .groupby("operator").mean()
    rows = []
    for k in (10, 25, 50):
        cells = []
        for suf in ("", "f", "d", "s"):
            o = f"link+k{k}{suf}"
            if o not in m.index:
                continue
            cells.append(f"{m.loc[o, 'pl_rmse_db']:.2f} & {pct(m.loc[o, 'assoc_acc'])} & "
                         f"{pct(reg.loc[o])}")
            N[f"svPl{SUFFIX[suf]}{KWORD[k]}"] = f(m.loc[o, "pl_rmse_db"], 2)
            N[f"svReg{SUFFIX[suf]}{KWORD[k]}"] = pct(reg.loc[o])
            N[f"svAssoc{SUFFIX[suf]}{KWORD[k]}"] = pct(m.loc[o, "assoc_acc"])
        rows.append(f"{k} & " + " & ".join(cells) + r" \\")
    save_table(G, "tab_survey", tabular("rrrrrrrrrrrrr", [
        r"& \multicolumn{3}{c}{Random, EB $\kappa$} & \multicolumn{3}{c}{Random, fixed $\kappa$} & "
        r"\multicolumn{3}{c}{Variance design} & \multicolumn{3}{c}{Stratified design} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r"$k$ & PL & serv. & regret & PL & serv. & regret & PL & serv. & regret & "
        r"PL & serv. & regret \\"], rows))
    N["svPlZero"] = f(m.loc["link", "pl_rmse_db"], 2)
    N["svAssocZero"] = pct(m.loc["link", "assoc_acc"])
    N["svRegZero"] = pct(reg.loc["link"])
    kap = n[n.operator == "link+k25"].kappa
    N["kappaMedianTwentyFive"] = f"{np.median(kap):.0f}"
    N["kappaShareLargeTwentyFive"] = pct((kap >= 1e5).mean(), 0)
    # best survey variant at k = 25 in path loss
    cand = {s_: m.loc[f"link+k25{s_}", "pl_rmse_db"] for s_ in ("", "f", "d", "s")
            if f"link+k25{s_}" in m.index}
    best = min(cand, key=cand.get)
    N["svBestTwentyFive"] = {"": "random tiles with the EB $\\kappa$",
                             "f": "random tiles with the fixed $\\kappa$",
                             "d": "the variance design", "s": "the stratified design"}[best]


def density(T, G, N):
    """The same predictors on the 20-site and on the 10-site network."""
    n_all = pd.concat([pd.read_csv(T / "network_metrics.csv"),
                       pd.read_csv(T / "network_link.csv")])
    pl_all = pd.concat([pd.read_csv(T / "planning.csv"), pd.read_csv(T / "planning_link.csv")])
    smax, smin = int(n_all.sites.max()), int(n_all.sites.min())
    out = {}
    for S in (smax, smin):
        nn = n_all[n_all.sites == S]
        pp = pl_all[(pl_all.sites == S) & (pl_all.K == 5)]
        out[S] = (nn.groupby(["operator", "fold"])[["pl_rmse_db", "assoc_acc", "sinr_mae_db"]]
                  .mean().groupby("operator").mean(),
                  pp.groupby(["operator", "fold"]).regret.mean().groupby("operator").mean())
    show = [o for o in ["oracle", "median", "uma_nlos", "kernel", "gp_ard", "ridge", "gbdt",
                        "link"] if o in out[smin][0].index]
    rows = []
    for o in show:
        c = []
        for S in (smax, smin):
            m, r = out[S]
            c.append(f"{m.loc[o, 'pl_rmse_db']:.2f} & {pct(m.loc[o, 'assoc_acc'])} & "
                     f"{m.loc[o, 'sinr_mae_db']:.2f} & {pct(r.loc[o])}")
        rows.append(f"{OP_NAME[o]} & " + " & ".join(c) + " \\\\")
        if o == "oracle":
            rows.append(r"\midrule")
    save_table(G, "tab_density", tabular("lrrrrrrrr", [
        rf"& \multicolumn{{4}}{{c}}{{{smax} sites}} & \multicolumn{{4}}{{c}}{{{smin} sites}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Predictor & PL (dB) & serv.\ (\%) & SINR (dB) & regret (\%) & "
        r"PL (dB) & serv.\ (\%) & SINR (dB) & regret (\%) \\"], rows))
    N["nSitesSmall"] = smin
    m, r = out[smin]
    for o, key in [("oracle", "Oracle"), ("gbdt", "Gbdt"), ("link", "Link"),
                   ("median", "Median")]:
        N[f"smallNet{key}"] = f(m.loc[o, "pl_rmse_db"], 1)
        N[f"smallAssoc{key}"] = pct(m.loc[o, "assoc_acc"])
        N[f"smallRegret{key}"] = pct(r.loc[o])


# ----------------------------------------------------------------------------
def variants(T, G, N):
    """Receiver-tile variants (hurdle, dual slope) and the physics baselines."""
    v = pd.read_csv(T / "network_variants.csv")
    pl = pd.read_csv(T / "planning_variants.csv")
    per = v.groupby(["operator", "fold"]).mean(numeric_only=True)
    per["regret"] = pl[pl.K == 5].groupby(["operator", "fold"]).regret.mean()
    m = per.groupby("operator").mean()
    groups = [("Receiver tile, exact labels",
               ["oracle", "oracle+h", "oracle+los", "oracle+los+h"]),
              ("Receiver tile, source median",
               ["median", "median+h", "median+los", "median+los+h"]),
              ("Receiver tile, GBDT transfer", ["gbdt", "gbdt+h", "gbdt+los", "gbdt+los+h"]),
              ("Per-link physics, no training", ["fspl_ke", "deygout", "uma_geo"]),
              ("Site-aware link model", ["link"])]
    rows = []
    for title, ops in groups:
        rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\")
        for o in ops:
            if o not in m.index:
                continue
            r = m.loc[o]
            rows.append(f"{OP_NAME[o]} & {r.pl_rmse_db:.2f} & {r.pl_rmse_cens_db:.2f} & "
                        f"{pct(r.reach_acc)} & {pct(r.assoc_acc)} & {pct(r.cov_acc_130)} & "
                        f"{r.sinr_mae_db:.2f} & {pct(r.regret)} \\\\")
    save_table(G, "tab_variants", tabular("lrrrrrrr", [
        r"& \multicolumn{2}{c}{Path loss (dB)} & & & & & \\", r"\cmidrule(lr){2-3}",
        r"Predictor & RMSE & cens. & reach (\%) & serv.\ (\%) & cov.\ (\%) & SINR (dB) & "
        r"regret (\%) \\"], rows))
    key = {"oracle": "Oracle", "oracle+h": "OracleH", "oracle+los": "OracleLos",
           "oracle+los+h": "OracleLosH", "median": "Median", "median+h": "MedianH",
           "median+los+h": "MedianLosH", "gbdt": "Gbdt", "gbdt+h": "GbdtH", "gbdt+los": "GbdtLos",
           "gbdt+los+h": "GbdtLosH", "fspl_ke": "Ke", "deygout": "Deygout", "uma_geo": "UmaGeo",
           "link": "Link"}
    for o, k in key.items():
        if o in m.index:
            N[f"var{k}Net"] = f(m.loc[o, "pl_rmse_db"], 1)
            N[f"var{k}Cens"] = f(m.loc[o, "pl_rmse_cens_db"], 1)
            N[f"var{k}Assoc"] = pct(m.loc[o, "assoc_acc"])
            N[f"var{k}Regret"] = pct(m.loc[o, "regret"])
            N[f"var{k}Reach"] = pct(m.loc[o, "reach_acc"])
            N[f"var{k}Sinr"] = f(m.loc[o, "sinr_mae_db"], 1)
    # best tile variant and best physics baseline, per metric
    tile_ops = [o for o in ["gbdt", "gbdt+h", "gbdt+los", "gbdt+los+h"] if o in m.index]
    phys = [o for o in ["fspl_ke", "deygout", "uma_geo"] if o in m.index]
    for mtr, name, lower in [("pl_rmse_db", "Net", True), ("assoc_acc", "Assoc", False),
                             ("regret", "Regret", True)]:
        bt = (m.loc[tile_ops, mtr].idxmin() if lower else m.loc[tile_ops, mtr].idxmax())
        bp = (m.loc[phys, mtr].idxmin() if lower else m.loc[phys, mtr].idxmax())
        N[f"varBestTile{name}"] = OP_NAME[bt]
        N[f"varBestPhys{name}"] = OP_NAME[bp]
        N[f"varBestPhys{name}Val"] = (f(m.loc[bp, mtr], 1) if lower and mtr.endswith("db")
                                      else pct(m.loc[bp, mtr]))
        lk, bst = per.loc["link"][mtr], per.loc[bp][mtr]
        N[f"nLinkBetterPhys{name}"] = int(((lk < bst) if lower else (lk > bst)).sum())
        lk2, bt2 = per.loc["link"][mtr], per.loc["gbdt+los+h"][mtr]
        N[f"nLinkBetterTile{name}"] = int(((lk2 < bt2) if lower else (lk2 > bt2)).sum())
    N["nVarFolds"] = int(per.loc["link"].shape[0])
    los = pd.read_csv(T / "los_labels.csv")
    rows = [f"{nice(r.city)} & {pct(r.los_links_share)} & {r.tiles_los}/{r.tiles} & "
            f"{r.tiles_nlos}/{r.tiles} & {r.gamma_los_median:.2f} & {r.gamma_nlos_median:.2f} & "
            f"{r.pl0_los_median:.1f} & {r.pl0_nlos_median:.1f} \\\\" for r in los.itertuples()]
    save_table(G, "tab_los", tabular("lrrrrrrr", [
        r"& LoS links & \multicolumn{2}{c}{Tiles with a pair} & "
        r"\multicolumn{2}{c}{med.\ $\gamma$} & \multicolumn{2}{c}{med.\ $P_{L_0}$ (dB)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"City & (\%) & LoS & NLoS & LoS & NLoS & LoS & NLoS \\"], rows))
    N["losShareMin"], N["losShareMax"] = pct(los.los_links_share.min(), 0), \
        pct(los.los_links_share.max(), 0)
    N["losGammaMedian"] = f(los.gamma_los_median.median(), 2)
    N["nlosGammaMedian"] = f(los.gamma_nlos_median.median(), 2)


# ----------------------------------------------------------------------------
def certified(T, G, N):
    c = pd.read_csv(T / "certified.csv")
    n = int(c.n_cities.max())
    N["certLevel"] = f"{n - 1}/{n}"
    N["certLevelTwo"] = f"{n - 2}/{n}"
    N["certLevelPct"] = pct((n - 1) / n, 0)
    show = [o for o in ["oracle", "median", "uma_nlos", "kernel", "ridge", "gbdt", "link",
                        "link_reg"] if o in set(c.operator)]
    rows = []
    for a, L, j in ((0.10, 140.0, 1), (0.10, 140.0, 2), (0.05, 140.0, 1)):
        sub = c[(c.alpha == a) & (c.level == L) & (c["rank"] == j)]
        per = sub.groupby(["operator", "city"]).mean(numeric_only=True)
        mean = per.groupby("operator").mean()
        valid = sub.groupby(["operator", "city"]).valid.mean().groupby("operator").sum()
        head = (rf"$\alpha={a:g}$, $L=\SI{{{L:g}}}{{\decibel}}$, bound holding with "
                rf"probability $\ge {n - j}/{n}$")
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{head}}}}} \\")
        for o in show:
            m = mean.loc[o]
            rows.append(f"{OP_NAME[o]} & {m.qhat:.1f} & {valid.loc[o]:g} & {pct(m.resid_cov)} & "
                        f"{pct(m.claimed)} & {pct(m.false_claim, 2)} & {pct(m.naive_claimed)} & "
                        f"{pct(m.naive_false, 2)} & {pct(m.true_cov)} \\\\")
            if a == 0.10 and j == 2 and o in ("link", "gbdt"):
                N[f"certClaimTwo{'Link' if o == 'link' else 'Gbdt'}"] = pct(m.claimed)
                N[f"certValidTwo{'Link' if o == 'link' else 'Gbdt'}"] = f"{valid.loc[o]:g}"
            if a == 0.10 and j == 1:
                k = {"oracle": "Oracle", "median": "Median", "uma_nlos": "UmaNlos",
                     "kernel": "Kernel", "ridge": "Ridge", "gbdt": "Gbdt", "link": "Link",
                     "link_reg": "LinkReg"}[o]
                N[f"certQ{k}"] = f(m.qhat, 1)
                N[f"certClaim{k}"] = pct(m.claimed)
                N[f"certFalse{k}"] = pct(m.false_claim, 2)
                N[f"certNaiveFalse{k}"] = pct(m.naive_false, 2)
                N[f"certNaiveClaim{k}"] = pct(m.naive_claimed)
                N[f"certValid{k}"] = f"{valid.loc[o]:g}"
                N[f"certCov{k}"] = pct(m.resid_cov)
        if a == 0.10 and j == 1:
            N["certTrueCov"] = pct(mean.loc["link"].true_cov)
            N["certMaxFalse"] = pct(mean.false_claim.max(), 2)
    save_table(G, "tab_certified", tabular("lrrrrrrrr", [
        r"& & & & \multicolumn{2}{c}{Certified claim} & \multicolumn{2}{c}{Naive claim} & \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Predictor & $\hat q_t$ (dB) & valid & $r_p\le\hat q_t$ (\%) & claimed (\%) & "
        r"false (\%) & claimed (\%) & false (\%) & covered (\%) \\"], rows))


def tile_size(T, G, N):
    d = pd.read_csv(T / "tile_size.csv")
    per = d.groupby(["G", "operator", "fold"]).mean(numeric_only=True)
    m = per.groupby(["G", "operator"]).mean()
    o_ = per.xs("oracle", level="operator")
    for c in ("se_iid", "se_cl"):
        num = (o_.n_tiles * o_[c] ** 2).groupby(level="G").sum()
        pooled = num / o_.n_tiles.groupby(level="G").sum()
        for g, v in np.sqrt(pooled).items():
            m.loc[(g, "oracle"), c] = v
    rows = []
    for g in sorted(d.G.unique()):
        o = m.loc[g]
        rows.append(f"{g:.0f} & {o.loc['oracle', 'n_tiles']:.0f} & "
                    f"{o.loc['oracle', 'shrink_median']:.2f} & "
                    f"{o.loc['oracle', 'se_iid']:.3f} & {o.loc['oracle', 'se_cl']:.3f} & "
                    f"{o.loc['median', 'gamma_rmse']:.3f} & {o.loc['gbdt', 'gamma_rmse']:.3f} & "
                    f"{o.loc['oracle', 'pl_rmse_db']:.2f} & {o.loc['gbdt', 'pl_rmse_db']:.2f} & "
                    f"{pct(o.loc['oracle', 'assoc_acc'])} \\\\")
    save_table(G, "tab_tilesize", tabular("rrrrrrrrrr", [
        r"& & & \multicolumn{2}{c}{$\mathrm{se}(\gamma)$} & \multicolumn{2}{c}{$\gamma$ RMSE} & "
        r"\multicolumn{2}{c}{PL RMSE (dB)} & \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"$G_s$ (m) & tiles/city & med.\ $w_t$ & iid & cluster & median & GBDT & oracle & GBDT & "
        r"serv.\ (\%) \\"], rows))
    acc = [m.loc[(g, "oracle"), "assoc_acc"] for g in sorted(d.G.unique())]
    lo_, hi_ = pct(min(acc)), pct(max(acc))
    N["tsAssocRange"] = lo_ if lo_ == hi_ else f"{lo_}--{hi_}"
    for g, key in ((50.0, "Fifty"), (100.0, "Hundred"), (200.0, "TwoHundred")):
        N[f"tsOracle{key}"] = f(m.loc[(g, "oracle"), "pl_rmse_db"], 1)
        N[f"tsGbdt{key}"] = f(m.loc[(g, "gbdt"), "pl_rmse_db"], 1)
        N[f"tsGamma{key}"] = f(m.loc[(g, "gbdt"), "gamma_rmse"], 3)
        N[f"tsSe{key}"] = f(m.loc[(g, "oracle"), "se_cl"], 3)
        N[f"tsAssoc{key}"] = pct(m.loc[(g, "oracle"), "assoc_acc"])


def censoring(T, G, N):
    c = pd.read_csv(T / "censoring.csv")
    rows = [f"{nice(r.city)} & {pct(r.censored_share)} & {pct(r.links_removed_140)} & "
            f"{fs(r.dgamma_median_140, 3)} & {fs(r.dgamma_median_130, 3)} & {r.tobit_tiles} & "
            f"{fs(r.dgamma_tobit_median, 3)} & {fs(r.dgamma_tobit_q90, 3)} \\\\"
            for r in c.itertuples()]
    save_table(G, "tab_censoring", tabular("lrrrrrrr", [
        r"& censored & \multicolumn{3}{c}{Lower cut-off} & "
        r"\multicolumn{3}{c}{Tobit vs least squares} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"City & pairs (\%) & links $>$140 (\%) & $\Delta\gamma_{140}$ & $\Delta\gamma_{130}$ & "
        r"tiles & med.\ $\Delta\gamma$ & q90 \\"], rows))
    N["censMin"], N["censMax"] = pct(c.censored_share.min()), pct(c.censored_share.max())
    N["censDgOneForty"] = fs(c.dgamma_median_140.median(), 3)
    N["censDgOneThirty"] = fs(c.dgamma_median_130.median(), 3)
    N["censTobitMin"] = f(c.dgamma_tobit_median.min(), 2)
    N["censTobitMax"] = f(c.dgamma_tobit_median.max(), 2)
    N["censTobitMedian"] = f(c.dgamma_tobit_median.median(), 2)
    N["censTobitTiles"] = thou(c.tobit_tiles.sum())


def siting(T, G, N):
    s = pd.read_csv(T / "siting.csv")
    reps = [r for r in ["sionna_randsite", "sionna_capped", "sionna_mat"]
            if r in set(s.replicate)]
    rep_name = {"sionna_randsite": "Random rooftops", "sionna_capped": "Mid-rise rooftops",
                "sionna_mat": "Random materials"}
    rep_key = {"sionna_randsite": "Rand", "sionna_capped": "Capped", "sionna_mat": "Mat"}
    _, _, per = nets(T)
    cities = sorted(s.city.unique())
    mm = per.loc[per.index.get_level_values(1).isin(cities)].groupby("operator").mean()
    rows = []
    for o, ref in (("oracle_rand", "oracle"), ("oracle_main", None), ("median", "median"),
                   ("gbdt", "gbdt"), ("link", "link")):
        refc = (f"{mm.loc[ref, 'pl_rmse_db']:.2f} & {pct(mm.loc[ref, 'assoc_acc'])} & "
                f"{pct(mm.loc[ref, 'regret'])}" if ref else "-- & -- & --")
        cells = []
        for rep in reps:
            m = s[s.replicate == rep].groupby(["operator", "city"]).mean(numeric_only=True) \
                .groupby("operator").mean()
            a = m.loc[o]
            cells.append(f"{a.pl_rmse_db:.2f} & {pct(a.assoc_acc)} & {pct(a.regret5)}")
        rows.append(f"{OP_NAME[o]} & {refc} & " + " & ".join(cells) + r" \\")
    head = r"& \multicolumn{3}{c}{Main corpus} & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{rep_name[r]}}}" for r in reps) + r" \\"
    cm = "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(reps) + 1))
    save_table(G, "tab_siting", tabular("l" + "rrr" * (len(reps) + 1), [
        head, cm, "Predictor & " + " & ".join(["PL & serv. & regret"] * (len(reps) + 1)) + r" \\"],
        rows))
    N["repList"] = ", ".join(rep_name[r].lower() for r in reps)
    N["nReplicates"] = len(reps)
    for rep in reps:
        m = s[s.replicate == rep].groupby(["operator", "city"]).mean(numeric_only=True) \
            .groupby("operator").mean()
        k = rep_key[rep]
        N[f"siteH{k}"] = f"{s[s.replicate == rep].site_h_median.median():.0f}"
        for o, key in [("oracle_rand", "OracleRand"), ("oracle_main", "OracleMain"),
                       ("gbdt", "Gbdt"), ("link", "Link"), ("median", "Median")]:
            N[f"rs{k}Net{key}"] = f(m.loc[o, "pl_rmse_db"], 1)
            N[f"rs{k}Assoc{key}"] = pct(m.loc[o, "assoc_acc"])
            N[f"rs{k}Regret{key}"] = pct(m.loc[o, "regret5"])
            N[f"rs{k}Bias{key}"] = fs(m.loc[o, "pl_bias_db"], 1)
        sr = s[s.replicate == rep]
        lk = sr[sr.operator == "link"].groupby("city").pl_rmse_db.mean()
        orc = sr[sr.operator == "oracle_rand"].groupby("city").pl_rmse_db.mean()
        N[f"rs{k}LinkBetterOracle"] = int((lk < orc.reindex(lk.index)).sum())
        N[f"rs{k}Cities"] = int(len(lk))
    N["rsNetLinkMain"] = f(mm.loc["link", "pl_rmse_db"], 1)
    N["rsNetOracleMainSiting"] = f(mm.loc["oracle", "pl_rmse_db"], 1)
    corp = pd.read_csv(T / "corpus.csv").set_index("city").loc[cities]
    N["rsSiteHMain"] = f"{corp.site_h_median.median():.0f}"


def band(T, T2, G, N, f2):
    lab = pd.read_csv(T / "band_labels.csv")
    bt = pd.read_csv(T / "band_transfer.csv")
    n2 = pd.concat([pd.read_csv(T2 / "network_metrics.csv"), pd.read_csv(T2 / "network_link.csv")])
    n2 = n2[n2.sites == n2.sites.max()]
    p2 = pd.concat([pd.read_csv(T2 / "planning.csv"), pd.read_csv(T2 / "planning_link.csv")])
    p2 = p2[(p2.sites == p2.sites.max()) & (p2.K == 5)]
    m = pd.concat([n2, bt]).groupby(["operator", "fold"])[
        ["pl_rmse_db", "assoc_acc", "sinr_mae_db"]].mean().groupby("operator").mean()
    reg = pd.concat([p2[["operator", "fold", "regret"]],
                     bt.rename(columns={"regret5": "regret"})[["operator", "fold", "regret"]]]) \
        .groupby(["operator", "fold"]).regret.mean().groupby("operator").mean()
    show = ["oracle", "median", "uma_nlos", "gbdt", "link", "link_x_phys", "link_x_fs"]
    rows = []
    for o in show:
        if o == "link":
            rows.append(r"\midrule")
        rows.append(f"{OP_NAME[o]} & {m.loc[o, 'pl_rmse_db']:.2f} & {pct(m.loc[o, 'assoc_acc'])} & "
                    f"{m.loc[o, 'sinr_mae_db']:.2f} & {pct(reg.loc[o])} \\\\")
        if o == "oracle":
            rows.append(r"\midrule")
    save_table(G, "tab_band", tabular("lrrrr", [
        r"Predictor & PL (dB) & serv.\ (\%) & SINR (dB) & regret (\%) \\"], rows))
    c2 = pd.read_csv(T2 / "corpus.csv")
    N["nSamplesTwo"] = thou(c2.samples.sum())
    N["nTilesTwo"] = thou(c2.tiles.sum())
    N["qcRemovedTwo"] = thou(c2.qc_removed.sum())
    N["bandGammaOne"] = f(lab.gamma_median_1.median(), 2)
    N["bandGammaTwo"] = f(lab.gamma_median_2.median(), 2)
    N["bandCorrMin"], N["bandCorrMax"] = f(lab.gamma_corr.min(), 2), f(lab.gamma_corr.max(), 2)
    N["bandPlShift"] = f(lab.pl0_shift_median.median(), 1)
    N["bandFsOffset"] = f(20 * np.log10(f2 / 3.5), 1)
    for o, key in [("oracle", "Oracle"), ("median", "Median"), ("gbdt", "Gbdt"),
                   ("link", "Link"), ("link_x_phys", "Cross"), ("link_x_fs", "CrossFs")]:
        N[f"bandNet{key}"] = f(m.loc[o, "pl_rmse_db"], 1)
        N[f"bandAssoc{key}"] = pct(m.loc[o, "assoc_acc"])
        N[f"bandRegret{key}"] = pct(reg.loc[o])


def external(T, G, N):
    e = pd.read_csv(T / "external.csv")
    e = e.groupby(["scene", "operator"], sort=False).mean(numeric_only=True).reset_index()
    ops = [o for o in ["oracle", "median", "ridge", "gbdt", "link", "link+k25", "link+k25d",
                       "link+k25s"] if o in set(e.operator)]
    rows = []
    for sc, g in e.groupby("scene", sort=False):
        g = g.set_index("operator")
        rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{nice(sc)}}} ({thou(g.links.iloc[0])} links, "
                    rf"{g.sites.iloc[0]:.0f} sites, {g.tiles.iloc[0]:.0f} tiles)}} \\")
        for o in ops:
            r = g.loc[o]
            rows.append(f"{OP_NAME[o]} & {r.pl_rmse_db:.2f} & {fs(r.pl_bias_db, 1)} & "
                        f"{pct(r.assoc_acc)} & {r.sinr_mae_db:.2f} & {r.edge_disp_m:.0f} & "
                        f"{pct(r.regret5)} \\\\")
    m = e.groupby("operator")[["pl_rmse_db", "pl_bias_db", "assoc_acc", "sinr_mae_db",
                               "edge_disp_m", "regret5"]].mean()
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{l}{\emph{Mean over scenes}} \\")
    for o in ops:
        r = m.loc[o]
        rows.append(f"{OP_NAME[o]} & {r.pl_rmse_db:.2f} & {fs(r.pl_bias_db, 1)} & "
                    f"{pct(r.assoc_acc)} & {r.sinr_mae_db:.2f} & {r.edge_disp_m:.0f} & "
                    f"{pct(r.regret5)} \\\\")
    save_table(G, "tab_external", tabular("lrrrrrr", [
        r"Predictor & PL (dB) & bias & serv.\ (\%) & SINR (dB) & edge (m) & regret (\%) \\"],
        rows))
    N["nExtScenes"] = e.scene.nunique()
    N["nExtLinks"] = thou(e.groupby("scene").links.first().sum())
    for o, key in [("oracle", "Oracle"), ("median", "Median"), ("gbdt", "Gbdt"),
                   ("link", "Link"), ("link+k10", "LinkTen"), ("link+k25", "LinkTwentyFive"),
                   ("link+k25d", "LinkTwentyFiveD"), ("link+k25s", "LinkTwentyFiveS")]:
        if o not in m.index:
            continue
        N[f"extBias{key}"] = fs(m.loc[o, "pl_bias_db"], 1)
        N[f"extNet{key}"] = f(m.loc[o, "pl_rmse_db"], 2 if o.startswith(("oracle", "link+")) else 1)
        N[f"extAssoc{key}"] = pct(m.loc[o, "assoc_acc"])
        N[f"extRegret{key}"] = pct(m.loc[o, "regret5"])
    k25 = e[e.operator == "link+k25"].set_index("scene").pl_rmse_db
    orc = e[e.operator == "oracle"].set_index("scene").pl_rmse_db
    N["extKBetterOracle"] = int((k25 < orc.reindex(k25.index)).sum())
    lk = e[e.operator == "link"].set_index("scene").pl_rmse_db
    N["extLinkMin"], N["extLinkMax"] = f(lk.min(), 1), f(lk.max(), 1)
    gb = e[e.operator == "gbdt"].set_index("scene").pl_rmse_db
    N["extLinkBetterGbdt"] = int((lk < gb.reindex(lk.index)).sum())
    N["extLinkBetterOracle"] = int((lk < orc.reindex(lk.index)).sum())
    bias = e[e.operator == "link"].pl_bias_db
    N["extLinkBiasMin"], N["extLinkBiasMax"] = fs(bias.min(), 1), fs(bias.max(), 1)
    N["extLinkBiasAllPositive"] = "every scene" if (bias > 0).all() else \
        f"{int((bias > 0).sum())} of {len(bias)} scenes"
    ed = e.groupby("operator").edge_disp_m.mean()
    N["extEdgeLink"], N["extEdgeGbdt"] = f"{ed['link']:.0f}", f"{ed['gbdt']:.0f}"
    N["extSinrLink"] = f(m.loc["link", "sinr_mae_db"], 1)
    N["extSinrGbdt"] = f(m.loc["gbdt", "sinr_mae_db"], 1)
    zs = e[~e.operator.str.contains(r"\+")]
    reg = zs.groupby("scene").regret5.agg(["min", "max"])
    N["extRegretSame"] = "identical" if (reg["max"] - reg["min"] < 1e-9).all() else "close"
    N["extRegretMin"], N["extRegretMax"] = pct(zs.regret5.min()), pct(zs.regret5.max())


def mixed(T, G, N):
    """Mixed-path composition of the tile model (scripts/18_mixed_path.py)."""
    mx = pd.read_csv(T / "mixed_path.csv")
    mpl = pd.read_csv(T / "mixed_path_plan.csv")
    fit = pd.read_csv(T / "mixed_path_labels.csv")
    _, _, rtf = nets(T)
    cols = ["pl_rmse_db", "assoc_acc", "sinr_mae_db", "edge_disp_m"]
    mxf = mx.groupby(["model", "operator", "fold"])[cols].mean()
    mxf["regret"] = mpl[mpl.K == 5].groupby(["model", "operator", "fold"]).regret.mean()
    groups = [("Receiver tile", [("oracle", "Tile labels (oracle)"), ("gbdt", "GBDT")], rtf),
              ("Mixed path, path-averaged", [("oracle", "Tile labels (oracle)"),
                                             ("median", "Source median"),
                                             ("kernel", "Kernel (NW)"), ("gbdt", "GBDT")],
               mxf.xs("average", level="model")),
              ("Mixed path, piecewise (multi-slope)", [("oracle", "Tile labels (oracle)"),
                                                       ("median", "Source median"),
                                                       ("kernel", "Kernel (NW)"),
                                                       ("gbdt", "GBDT")],
               mxf.xs("piecewise", level="model")),
              (r"Mixed path, summed (\ref{app:summed})", [("oracle", "Tile labels (oracle)"),
                                                           ("median", "Source median"),
                                                           ("kernel", "Kernel (NW)"),
                                                           ("gbdt", "GBDT")],
               mxf.xs("sum", level="model")),
              ("Site-aware link model", [("link", "Link transfer")], rtf)]
    rows = []
    for title, ops, d in groups:
        rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\")
        m = d.groupby(level="operator").mean()
        for o, name in ops:
            r = m.loc[o]
            rows.append(f"{name} & {r.pl_rmse_db:.2f} & {pct(r.assoc_acc)} & "
                        f"{r.sinr_mae_db:.2f} & {r.edge_disp_m:.0f} & {pct(r.regret)} \\\\")
    save_table(G, "tab_mixed", tabular("lrrrrr", [
        r"Predictor & PL (dB) & serv.\ (\%) & SINR (dB) & edge (m) & regret (\%) \\"], rows))
    for model, mk in (("average", "Avg"), ("sum", "Sum"), ("piecewise", "Pw")):
        m = mxf.xs(model, level="model").groupby(level="operator").mean()
        for o, ok in (("oracle", "Oracle"), ("median", "Median"), ("kernel", "Kernel"),
                      ("gbdt", "Gbdt")):
            N[f"mix{mk}Net{ok}"] = f(m.loc[o, "pl_rmse_db"], 1)
            N[f"mix{mk}Assoc{ok}"] = pct(m.loc[o, "assoc_acc"])
            N[f"mix{mk}Regret{ok}"] = pct(m.loc[o, "regret"])
            N[f"mix{mk}Edge{ok}"] = f"{m.loc[o, 'edge_disp_m']:.0f}"
        tr = m.loc[["median", "kernel", "gbdt"], "pl_rmse_db"]
        N[f"mix{mk}TransMin"], N[f"mix{mk}TransMax"] = f(tr.min(), 1), f(tr.max(), 1)
        fr = fit[fit.model == model].fit_rmse_db
        N[f"mix{mk}FitMin"], N[f"mix{mk}FitMax"] = f(fr.min(), 1), f(fr.max(), 1)
    lk = rtf.xs("link", level="operator")
    for model, mk in (("average", "Avg"), ("piecewise", "Pw"), ("sum", "Sum")):
        ao = mxf.xs((model, "oracle"), level=("model", "operator"))
        N[f"mix{mk}LinkBetterPl"] = int((lk.pl_rmse_db < ao.pl_rmse_db.reindex(lk.index)).sum())
        N[f"mix{mk}LinkBetterAssoc"] = int((lk.assoc_acc > ao.assoc_acc.reindex(lk.index)).sum())
        N[f"mix{mk}LinkBetterRegret"] = int((lk.regret <= ao.regret.reindex(lk.index)).sum())
    ag = mxf.xs(("average", "gbdt"), level=("model", "operator"))
    rg = rtf.xs("gbdt", level="operator")
    N["mixAvgBetterRecvAssoc"] = int((ag.assoc_acc > rg.assoc_acc.reindex(ag.index)).sum())
    N["mixAvgBetterRecvPl"] = int((ag.pl_rmse_db < rg.pl_rmse_db.reindex(ag.index)).sum())


MIX_SIZES = (100, 200, 500, 1000)


def mixed_sizes(T, G, N):
    """Mixed-path compositions at the tile sizes of the base study."""
    rows, stats_ = [], {}
    for gs in MIX_SIZES:
        tag = "" if gs == 100 else f"_G{gs}"
        mx = pd.read_csv(T / f"mixed_path{tag}.csv")
        pl = pd.read_csv(T / f"mixed_path_plan{tag}.csv")
        fit = pd.read_csv(T / f"mixed_path_labels{tag}.csv")
        per = mx.groupby(["model", "operator", "fold"])[["pl_rmse_db", "assoc_acc"]].mean()
        per["regret"] = pl[pl.K == 5].groupby(["model", "operator", "fold"]).regret.mean()
        m = per.groupby(level=["model", "operator"]).mean()
        tiles = fit.groupby("model").tiles.mean()
        stats_[gs] = m
        r = [f"{gs}", f"{tiles['average']:.0f}"]
        for model in ("sum", "piecewise", "average"):
            for o in ("oracle", "gbdt"):
                r += [f"{m.loc[(model, o), 'pl_rmse_db']:.1f}",
                      pct(m.loc[(model, o), "assoc_acc"])]
        rows.append(" & ".join(r) + r" \\")
    save_table(G, "tab_mixed_size", tabular("rrrrrrrrrrrrrr", [
        r"& & \multicolumn{4}{c}{Summed (\ref{app:summed})} & "
        r"\multicolumn{4}{c}{Piecewise} & \multicolumn{4}{c}{Path-averaged} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r"& & \multicolumn{2}{c}{labels} & \multicolumn{2}{c}{GBDT} & "
        r"\multicolumn{2}{c}{labels} & \multicolumn{2}{c}{GBDT} & "
        r"\multicolumn{2}{c}{labels} & \multicolumn{2}{c}{GBDT} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}"
        r"\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
        r"$G_s$ (m) & tiles & PL & serv. & PL & serv. & PL & serv. & PL & serv. & PL & serv. & "
        r"PL & serv. \\"], rows))
    big = [g for g in MIX_SIZES if g > 100]
    for model, mk in (("sum", "Sum"), ("average", "Avg"), ("piecewise", "Pw")):
        for o, ok in (("oracle", "Oracle"), ("gbdt", "Gbdt")):
            vals = [stats_[g].loc[(model, o), "pl_rmse_db"] for g in big]
            N[f"mixSize{mk}{ok}Min"], N[f"mixSize{mk}{ok}Max"] = f(min(vals), 1), f(max(vals), 1)
    aa = [stats_[g].loc[("average", "oracle"), "assoc_acc"] for g in big]
    N["mixSizeAvgAssocMin"], N["mixSizeAvgAssocMax"] = pct(min(aa)), pct(max(aa))
    best = min(stats_[g].loc[(mo, "oracle"), "pl_rmse_db"] for g in MIX_SIZES
               for mo in ("sum", "piecewise", "average"))
    bassoc = max(stats_[g].loc[(mo, "oracle"), "assoc_acc"] for g in MIX_SIZES
                 for mo in ("sum", "piecewise", "average"))
    N["mixSizeBestNet"], N["mixSizeBestAssoc"] = f(best, 1), pct(bassoc)
    _, _, per = nets(T)
    lm = per.loc["link"][["pl_rmse_db", "assoc_acc"]].mean()
    N["mixSizeLinkBetterBest"] = "better" if (best > lm.pl_rmse_db and bassoc < lm.assoc_acc) \
        else "not better"
    so100 = stats_[100].loc[("sum", "oracle"), "pl_rmse_db"]
    N["mixSumLargerWorse"] = "larger" if min(
        stats_[g].loc[("sum", "oracle"), "pl_rmse_db"] for g in big) > so100 else "not larger"
    for g, k in ((500, "FiveHundred"), (1000, "Thousand")):
        N[f"mixSumOracle{k}"] = f(stats_[g].loc[("sum", "oracle"), "pl_rmse_db"], 1)
        N[f"mixSumGbdt{k}"] = f(stats_[g].loc[("sum", "gbdt"), "pl_rmse_db"], 1)


def outliers(T, N):
    o = pd.read_csv(T / "rt_outliers.csv").set_index("run")
    N["outCity"] = nice(o.city.iloc[0])
    N["outDiffOn"] = int(o.loc["corpus settings", "below_6db"])
    N["outDiffOff"] = int(o.loc["diffraction off", "below_6db"])


def convergence(T, G, N):
    c = pd.read_csv(T / "rt_convergence.csv")
    if "city" not in c:                    # legacy single-site file
        c["city"], c["site"], c["site_h_m"], c["ref_rays"] = "amsterdam", 0, np.nan, c.rays.max()
    rows = []
    for (city, site), g in c.groupby(["city", "site"], sort=False):
        rows.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{nice(city)}, site {site}"
                    + (f", {g.site_h_m.iloc[0]:.0f}\\,m" if np.isfinite(g.site_h_m.iloc[0]) else "")
                    + rf"; reference {sci(g.ref_rays.iloc[0])} rays}}}} \\")
        for r in g.sort_values("rays").itertuples():
            rows.append(f"{sci(r.rays)} & {r.seconds:.0f} & " + " & ".join(
                f"{100 * getattr(r, f'frac_of_ref_{L}'):.1f} & {getattr(r, f'rms_diff_{L}'):.2f}"
                for L in (120, 130, 140, 150)) + " \\\\")
    save_table(G, "tab_convergence", tabular("rrrrrrrrrr", [
        r"& & \multicolumn{2}{c}{$\le120$\,dB} & \multicolumn{2}{c}{$\le130$\,dB} & "
        r"\multicolumn{2}{c}{$\le140$\,dB} & \multicolumn{2}{c}{$\le150$\,dB} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"Rays & s & cells & RMS & cells & RMS & cells & RMS & cells & RMS \\"], rows))
    used = c[c.rays == 10 ** 8]
    N["convNSites"] = int(len(used))
    N["convNCities"] = int(used.city.nunique())
    N["convRef"] = sci(c.ref_rays.max())
    N["convRmsOneTwentyMax"] = f(used.rms_diff_120.max(), 2)
    N["convRmsOneFortyMin"] = f(used.rms_diff_140.min(), 1)
    N["convRmsOneFortyMax"] = f(used.rms_diff_140.max(), 1)
    N["convRmsOneFiftyMax"] = f(used.rms_diff_150.max(), 1)
    N["convCellsOneFortyMin"] = pct(used.frac_of_ref_140.min())
    N["convCellsOneFiftyMin"] = pct(used.frac_of_ref_150.min())
    N["convBiasOneFortyMax"] = f(used.bias_140.abs().max(), 2)
    N["convSecondsMedian"] = f"{used.seconds.median():.0f}"
    N["convRefSecondsMax"] = f"{c[c.rays == c.ref_rays].seconds.max():.0f}"


def runtime(T, G, N):
    r = pd.read_csv(T / "runtime.csv")
    us = "\\_"
    def short(a):
        a = "" if pd.isna(a) else str(a)
        return a.replace("configs/data/", "").replace("--config ", "").replace("_", us)
    rows = [f"\\texttt{{{x.script.replace('_', us)}}} & \\texttt{{{short(x.args)}}} & "
            f"{x.seconds / 60:.0f} \\\\" for x in r.itertuples()]
    save_table(G, "tab_runtime", tabular("lp{4.1cm}r", [r"Script & Configuration & minutes \\"],
                                         rows))
    N["rtTotalHours"] = f"{r.seconds.sum() / 3600:.1f}"
    N["rtCpus"] = int(r.cpus.iloc[0])
    N["rtMachine"] = str(r.machine.iloc[0]).replace("_", "\\_")


def highlights(N, out):
    """Elsevier highlights: 3-5 bullets of at most 85 characters (plain text)."""
    v = {line.split("}{", 1)[0][len("\\newcommand{\\"):]: line.split("}{", 1)[1][:-1]
         for line in N.lines[1:]}
    plain = {k: val.replace("\\ensuremath{-}", "-").replace("{,}", ",") for k, val in v.items()}
    hl = [
        f"Open ray-traced 6G corpora: {plain['nCities']} cities, two bands, "
        "siting and material replicates",
        "Tile path-loss exponents depend on serving sites, tile size and cut-off",
        f"Exact tile labels reach {plain['netOracle']} dB and {plain['assocOracle']}% "
        "correct serving sites",
        f"Hurdle link transfer: {plain['netLink']} dB, {plain['assocLink']}% serving "
        f"sites, " + ("every held-out city" if plain["nLinkBetterGbdtPl"] == plain["nFolds"]
                      else f"{plain['nLinkBetterGbdtPl']} of {plain['nFolds']} cities"),
        f"City-level certificate promises {plain['certClaimLink']}% coverage at a "
        "guaranteed risk",
    ]
    for h in hl:
        assert len(h) <= 85, (len(h), h)
    out.write_text("\n".join(hl) + "\n")


def city_figure(repo, base, cfg, fig_dir, city):
    """(a) building heights, (b) best-server path loss, (c) tile exponents."""
    import json

    import geopandas as gpd
    from matplotlib.collections import PolyCollection

    from terranet.data.descriptors.scene_tiles import scene_building_parts
    from terranet.data.sionna_gen.geo import LocalFrame
    raw = repo / base.paths.raw
    meta = json.loads((raw / cfg.scenes_dir / city / "scene_meta.json").read_text())
    frame = LocalFrame.from_center(meta["lat0"], meta["lon0"])
    half = float(meta["size_m"]) / 2
    bld = gpd.read_file(raw / cfg.scenes_dir / city / "building.geojson")
    parts, heights = scene_building_parts(bld, frame, float(meta["size_m"]))
    meas = pd.read_parquet(raw / cfg.dataset / city / "measurements.parquet",
                           columns=["rx_lat", "rx_lon", "pathloss_db"])
    best = meas.groupby(["rx_lat", "rx_lon"]).pathloss_db.min().reset_index()
    bx, by = frame.to_local(best.rx_lon.to_numpy(), best.rx_lat.to_numpy())
    tiles = pd.read_parquet(repo / base.paths.processed / cfg.dataset / city / "tiles.parquet")
    tx, ty = frame.to_local(tiles.lon_c.to_numpy(), tiles.lat_c.to_numpy())

    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.45))
    hv = np.clip(heights, 0, 60)
    pc = PolyCollection([np.asarray(q.exterior.coords) for q in parts], array=hv,
                        cmap="Greys", edgecolors="none", clim=(0, 60))
    ax[0].add_collection(pc)
    fig.colorbar(pc, ax=ax[0], fraction=0.046, pad=0.02, label="height (m)")

    def raster(x, y, v, cell):
        n = int(round(2 * half / cell))
        ix = np.clip(((x + half) / cell).astype(int), 0, n - 1)
        iy = np.clip(((y + half) / cell).astype(int), 0, n - 1)
        img = np.full((n, n), np.nan)
        img[iy, ix] = v
        return img
    ext = (-half, half, -half, half)
    cell = float(np.median(np.diff(np.unique(np.round(bx, 3)))))
    im = ax[1].imshow(raster(bx, by, best.pathloss_db.to_numpy(), cell), origin="lower",
                      extent=ext, cmap="Greys", vmin=70, vmax=150, interpolation="nearest")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02, label="best-server path loss (dB)")
    ok = np.isfinite(tiles.gamma).to_numpy()
    tcell = float(cfg.tile_size_m)
    im2 = ax[2].imshow(raster(tx[ok], ty[ok], tiles.gamma.to_numpy()[ok], tcell), origin="lower",
                       extent=ext, cmap="Greys", interpolation="nearest")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02, label=r"tile exponent $\gamma$")
    for i, a in enumerate(ax):
        a.set_xlim(-half, half)
        a.set_ylim(-half, half)
        a.set_aspect("equal")
        a.set_xticks([-half, 0, half])
        a.set_yticks([-half, 0, half])
        lab = [f"\u2212{half / 1000:g}", "0", f"{half / 1000:g}"]
        a.set_xticklabels(lab)
        a.set_yticklabels(lab if i == 0 else [])
        a.set_xlabel("east (km)")
        a.set_title(f"({'abc'[i]})", loc="left")
    ax[0].set_ylabel("north (km)")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(fig_dir / "fig_city.pdf", dpi=300)
    plt.close(fig)


def corpus_settings(repo, base, cfg, N):
    """Ray-tracing settings, read from what was actually run (origin.json)."""
    import json

    gen = OmegaConf.load(repo / "configs" / "data" / "sionna_cities.yaml")
    runs = [json.loads((repo / base.paths.raw / cfg.dataset / c / "origin.json").read_text())
            for c in cfg.cities]
    one = {k: {str(r.get(k)) for r in runs} for k in
           ("sionna_version", "api", "rt_samples", "cell_size_m", "max_pathloss_db")}
    for k, v in one.items():
        assert len(v) == 1, f"cities were generated with different {k}: {v}"
    prop = {json.dumps(r.get("propagation"), sort_keys=True) for r in runs}
    assert len(prop) == 1, prop
    r0 = runs[0]
    N["sionnaVersion"] = str(r0["sionna_version"])
    N["rtSamples"] = sci(r0["rt_samples"])
    N["rtDepth"] = int(r0["propagation"]["max_depth"])
    N["cellSize"] = f"{float(r0['cell_size_m']):g}"
    N["maxPl"] = f"{float(r0['max_pathloss_db']):g}"
    N["sceneSize"] = f"{float(gen.size_m) / 1000:g}"
    N["siteMast"] = f"{float(gen.bs_height_agl):g}"
    N["ueHeight"] = f"{float(gen.ue_height_agl):g}"
    N["overtureRelease"] = str(gen.overture_release)
    secs = sum(float(r.get("wall_clock_s", 0.0)) for r in runs)
    n_sites = sum(int(r.get("n_bs", 0)) for r in runs)
    N["rtHours"] = f"{secs / 3600:.1f}"
    N["rtSecPerSite"] = f"{secs / max(n_sites, 1):.0f}"
    N["rtBackend"] = "CPU (LLVM)" if "llvm" in str(r0.get("mitsuba_variant", "")) else "GPU (CUDA)"
    N["minSep"] = "250"          # terranet.data.sionna_gen.bs_placement default
    from terranet.data.sionna_gen.bs_placement import CAPPED_ROOF_M
    N["cappedLo"], N["cappedHi"] = f"{CAPPED_ROOF_M[0]:g}", f"{CAPPED_ROOF_M[1]:g}"
    mat = OmegaConf.load(repo / "configs" / "data" / "sionna_cities_materials.yaml")
    N["matList"] = ", ".join(str(m) for m in mat.materials)
    N["nMat"] = len(mat.materials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default="configs/data/sionna.yaml")
    ap.add_argument("--band2", default="configs/data/sionna_7p5.yaml")
    a = ap.parse_args()
    repo = Path(a.repo)
    T = repo / "outputs" / "tables"
    G = repo / "paper" / "generated"
    F = repo / "paper" / "figures"
    G.mkdir(parents=True, exist_ok=True)
    for old in G.glob("*.tex"):            # no stale generated file survives a run
        old.unlink()
    cfg = OmegaConf.load(repo / a.config)
    base = OmegaConf.load(repo / "configs" / "base.yaml")
    tiles = pd.concat([pd.read_parquet(repo / base.paths.processed / cfg.dataset / c
                                       / "tiles.parquet").assign(city=c) for c in cfg.cities])
    tiles = tiles[np.isfinite(tiles.gamma)]
    N = Numbers()
    corpus_settings(repo, base, cfg, N)
    N["tileSize"] = f"{cfg.tile_size_m:g}"
    N["refDist"] = f"{cfg.d0_m:g}"
    N["minLinks"] = int(cfg.min_measurements_per_tile)
    N["freqGHz"] = f"{cfg.freq_ghz:g}"
    corpus(T, G, N, F, tiles)
    N["qcMargin"] = f"{float(cfg.qc_fspl_margin_db):g}"
    from terranet.evaluation.network import SINR_FLOOR_DB
    N.raw("sinrFloor", f"{SINR_FLOOR_DB:g}")
    from terranet.experiments.common import LINK_KAPPA, N_INNER_EXPENSIVE, SITES_SMALL
    N["kappaLink"] = f"{LINK_KAPPA:g}"
    N["nInnerExpensive"] = int(N_INNER_EXPENSIVE)
    gen = OmegaConf.load(repo / "configs" / "data" / "sionna_cities.yaml")
    area = (float(gen.size_m) / 1000) ** 2

    def isd(n_sites):          # hexagonal layout: area per site = sqrt(3)/2 ISD^2
        return float(np.sqrt(2 * area / n_sites / np.sqrt(3))) * 1000
    N["isdMain"] = f"{round(isd(int(gen.bs_per_city)), -1):.0f}"
    N["sceneArea"] = f"{area:g}"
    N["areaPerSite"] = f"{area / int(gen.bs_per_city):g}"
    N["areaPerSiteSmall"] = f"{area / SITES_SMALL:g}"
    words = {10: "Ten", 12: "Twelve", 15: "Fifteen", 20: "Twenty", 25: "Twenty-five"}
    N["nSitesWord"] = words.get(int(gen.bs_per_city), str(int(gen.bs_per_city)))
    N["isdSmall"] = f"{round(isd(SITES_SMALL), -1):.0f}"
    cfg2 = OmegaConf.load(repo / a.band2)
    N["freqTwoGHz"] = f"{float(cfg2.freq_ghz):g}"
    OP_NAME["link_x_phys"] = rf"Link, {float(cfg.freq_ghz):g}\,GHz model"
    OP_NAME["link_x_fs"] = rf"Link, {float(cfg.freq_ghz):g}\,GHz model, offset only"
    rand = OmegaConf.load(repo / "configs" / "data" / "sionna_cities_randsite.yaml")
    N["nRandCities"] = len(rand.rt_cities)
    N["randCities"] = ", ".join(nice(c) for c in rand.rt_cities)
    fig_city_name = str(cfg.get("figure_city", cfg.cities[0]))
    city_figure(repo, base, cfg, F, fig_city_name)
    N["figCity"] = nice(fig_city_name)
    loco(T, G, N, F)
    stats(T, G, N)
    degeneracy(T, G, N)
    fewshot(T, G, N, F)
    label_noise(T, G, N)
    network(T, G, N, F)
    variants(T, G, N)
    convergence(T, G, N)
    certified(T, G, N)
    tile_size(T, G, N)
    censoring(T, G, N)
    siting(T, G, N)
    band(T, repo / str(cfg2.outputs_dir) / "tables", G, N, float(cfg2.freq_ghz))
    outliers(T, N)
    external(T, G, N)
    mixed(T, G, N)
    mixed_sizes(T, G, N)
    runtime(T, G, N)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ext", repo / "scripts" / "17_external_scenes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    N["nExtSites"] = mod.SITES
    N["extMinSep"] = f"{mod.MIN_SEP:g}"
    # siunitx cannot parse the text-mode minus: macros used inside \SI must be raw
    vals = {ln.split("}{", 1)[0][len("\\newcommand{\\"):]: ln.split("}{", 1)[1][:-1]
            for ln in N.lines[1:]}
    tex = "".join(p.read_text() for p in (repo / "paper").rglob("*.tex")
                  if "generated" not in p.parts)
    for name in set(re.findall(r"\\SI\{\\([A-Za-z]+)\}", tex)):
        assert "ensuremath" not in vals.get(name, ""), f"\\{name} is negative inside \\SI"
    N.write(G / "numbers.tex")
    highlights(N, repo / "paper" / "highlights.txt")
    print(f"wrote {len(N.lines) - 1} macros, tables to {G}, figures to {F}")


if __name__ == "__main__":
    main()
