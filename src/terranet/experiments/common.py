"""Shared data access and transfer operators for the leave-one-city-out experiments.

Protocol (identical for every operator):
  * the held-out city is never touched; every other city is a training city;
  * hyper-parameters are chosen by an inner leave-one-training-city-out loop
    (`inner_splits`): each training city is held out once, the operator is
    fitted on the others, and the grid value with the smallest mean inner
    loss (`selection_loss`, standardised over both targets) is kept; the
    operator is then refitted on all training cities. For operators with
    early stopping the number of rounds or epochs of the final fit is the
    median best iteration over the inner folds;
  * descriptors are max-abs normalised with constants fitted on the training
    cities of the fit at hand (inner or outer) and applied unchanged to the
    held-out tiles;
  * stochastic operators are run for every seed in SEEDS; the inner selection
    uses seed 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from terranet.data.descriptors.scene_tiles import DESCRIPTOR_COLUMNS

SEEDS = (0, 1, 2, 3, 4)
TARGETS = ["gamma", "pl0"]
LINK_KAPPA = 50.0          # fixed prior weight (links) of the per-site offset (ablation)
SITES_SMALL = 10           # size of the reduced network (first sites of the siting rule)
N_INNER_EXPENSIVE = 3      # inner validation cities for the two expensive operators


def load_cfg(config: str):
    """Dataset config and shared paths; `outputs_dir` in the dataset config
    redirects tables, predictions and caches (used for the second band)."""
    cfg, base = OmegaConf.load(config), OmegaConf.load("configs/base.yaml")
    if cfg.get("outputs_dir"):
        base.paths.outputs = str(cfg.outputs_dir)
    return cfg, base


def load_splits(path: str = "data/splits/loco.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def inner_splits(train_cities, n_inner=None, seed=0):
    """Inner leave-one-training-city-out splits [(inner_train, inner_val)].
    With n_inner, only that many validation cities are drawn (fixed seed)."""
    cities = list(train_cities)
    vals = cities
    if n_inner is not None and n_inner < len(cities):
        rng = np.random.default_rng(seed)
        vals = [cities[i] for i in sorted(rng.choice(len(cities), n_inner, replace=False))]
    return [([c for c in cities if c != v], v) for v in vals]


def load_tiles(base, cfg, city: str, fitted_only: bool = True) -> pd.DataFrame:
    d = pd.read_parquet(Path(base.paths.processed) / cfg.dataset / city / "tiles.parquet")
    if fitted_only:
        d = d[np.isfinite(d.gamma)]
    return d.reset_index().rename(columns={"index": "tile_row"})


def xy(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return d[DESCRIPTOR_COLUMNS].to_numpy(np.float64), d[TARGETS].to_numpy(np.float64)


def stack(base, cfg, cities) -> pd.DataFrame:
    return pd.concat([load_tiles(base, cfg, c) for c in cities], ignore_index=True)


class MaxAbs:
    """Column max-abs scaling fitted on source cities only."""

    def fit(self, X):
        self.mx = np.abs(X).max(0)
        self.mx[self.mx == 0] = 1.0
        return self

    def __call__(self, X):
        return X / self.mx


class CityData:
    """Descriptors and labels of every city, with inner-fold access."""

    def __init__(self, base, cfg, cities):
        self.d = {c: xy(load_tiles(base, cfg, c)) for c in cities}

    def pool(self, cities):
        X = np.vstack([self.d[c][0] for c in cities])
        Y = np.vstack([self.d[c][1] for c in cities])
        return X, Y

    def fold(self, train, val):
        """Normalised (Xtr, Ytr, Xva, Yva) of one (inner or outer) fold."""
        Xtr, Ytr = self.pool(train)
        Xva, Yva = self.d[val]
        norm = MaxAbs().fit(Xtr)
        return norm(Xtr), Ytr, norm(Xva), Yva


# --------------------------------------------------------------------------
# standard-model constants (3GPP TR 38.901 UMa, fc in GHz, d0 = reference)
# --------------------------------------------------------------------------
def tr38901_uma(freq_ghz: float, d0_m: float, los: bool) -> np.ndarray:
    """(gamma, PL0 at d0) of the 3GPP UMa single-slope formula (h_UT = 1.5 m)."""
    if los:   # PL1 = 28 + 22 log10 d + 20 log10 fc  (d < breakpoint)
        g, a = 2.2, 28.0 + 20 * np.log10(freq_ghz)
    else:     # PL'_NLOS = 13.54 + 39.08 log10 d + 20 log10 fc - 0.6 (h_UT - 1.5)
        g, a = 3.908, 13.54 + 20 * np.log10(freq_ghz)
    return np.array([g, a + 10 * g * np.log10(d0_m)])


def cost231_hata(freq_ghz: float, d0_m: float, h_bs_m: float, h_ut_m: float = 1.5,
                 metropolitan: bool = True) -> np.ndarray:
    """(gamma, PL0 at d0) of COST-231 Hata (large-city a(h_m)); used here outside
    its validity range (1.5-2 GHz, 1-20 km, h_bs 30-200 m) as a reference."""
    f = freq_ghz * 1e3
    a_hm = 3.2 * np.log10(11.75 * h_ut_m) ** 2 - 4.97
    A = (46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_bs_m) - a_hm
         + (3.0 if metropolitan else 0.0))
    B = 44.9 - 6.55 * np.log10(h_bs_m)
    return np.array([B / 10, A + B * np.log10(d0_m / 1e3)])


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------
def nw_predict(Xq, Xs, Ys, sigma, batch=256, return_diag=False):
    """Nadaraya-Watson with a Gaussian kernel, memory-safe."""
    sq = (Xs ** 2).sum(1)
    out = np.empty((len(Xq), Ys.shape[1]))
    diag = {"D": [], "neff": [], "dmin": []}
    for s in range(0, len(Xq), batch):
        q = Xq[s:s + batch]
        d2 = np.maximum(sq[None] + (q ** 2).sum(1)[:, None] - 2 * q @ Xs.T, 0.0)
        dmin = d2.min(1, keepdims=True)
        w = np.exp(-(d2 - dmin) / (2 * sigma ** 2))          # shift-invariant
        w /= w.sum(1, keepdims=True)
        out[s:s + batch] = w @ Ys
        if return_diag:
            diag["D"].append(d2.max(1) - dmin[:, 0])
            diag["dmin"].append(dmin[:, 0])
            diag["neff"].append(1.0 / (w ** 2).sum(1))
    if return_diag:
        return out, {k: np.concatenate(v) for k, v in diag.items()}
    return out


SIGMA_GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
KNN_GRID = (1, 5, 20, 50, 200, 1000)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
MLP_ALPHA_GRID = (1e-4, 1e-2, 1.0)


def selection_loss(pred, Y, scale):
    """Model-selection loss: RMS over both targets of the error standardised by
    `scale` (the spread of the targets over the training tiles), so gamma and
    PL0 count equally."""
    return float(np.sqrt((((pred - Y) / np.asarray(scale)) ** 2).mean()))


def select_grid(grid, loss_fn):
    """Grid value with the smallest loss; returns (value, {value: loss})."""
    errs = {g: float(loss_fn(g)) for g in grid}
    return min(errs, key=errs.get), errs


def inner_select(data, train_cities, grid, fit_predict, n_inner=None, seed=0):
    """Mean inner-fold loss of every grid value. `fit_predict(g, Xtr, Ytr, Xva)`
    returns predictions for the inner validation tiles."""
    scale = data.pool(train_cities)[1].std(0)
    errs = {g: [] for g in grid}
    for tr, va in inner_splits(train_cities, n_inner, seed):
        Xtr, Ytr, Xva, Yva = data.fold(tr, va)
        for g in grid:
            errs[g].append(selection_loss(fit_predict(g, Xtr, Ytr, Xva), Yva, scale))
    mean = {g: float(np.mean(v)) for g, v in errs.items()}
    return min(mean, key=mean.get), mean


def knn_predict(Xq, Xs, Ys, k):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=min(k, len(Xs))).fit(Xs, Ys).predict(Xq)


def ridge_predict(Xq, Xs, Ys, alpha):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha).fit(Xs, Ys).predict(Xq)


GBDT_PARAMS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
GBDT_MAX_TREES = 2000


def _lgb_supports_eval_xy() -> bool:
    try:
        import inspect

        import lightgbm as lgb
        return "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters
    except ImportError:
        return False


_LGB_EVAL_XY = _lgb_supports_eval_xy()


def lgb_eval_kw(Xva, yva):
    return ({"eval_X": (Xva,), "eval_y": (yva,)} if _LGB_EVAL_XY
            else {"eval_set": [(Xva, yva)]})


class _Multi:
    """Column-wise wrapper of per-target models."""

    def __init__(self, models, n_trees=None):
        self.models, self.n_trees = models, n_trees

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])


def fit_gbdt_es(Xtr, Ytr, Xva, Yva, seed, sample_weight=None, patience=100):
    """LightGBM per target with early stopping on (Xva, Yva); n_trees = best
    iteration per target (used by the inner selection)."""
    import lightgbm as lgb
    models, best = [], []
    for j in range(Ytr.shape[1]):
        m = lgb.LGBMRegressor(n_estimators=GBDT_MAX_TREES, random_state=seed, **GBDT_PARAMS)
        m.fit(Xtr, Ytr[:, j], sample_weight=sample_weight,
              callbacks=[lgb.early_stopping(patience, verbose=False)],
              **lgb_eval_kw(Xva, Yva[:, j]))
        models.append(m)
        best.append(int(m.best_iteration_))

    class _M(_Multi):
        def predict(self, X):
            return np.column_stack([m.predict(X, num_iteration=n)
                                    for m, n in zip(self.models, self.n_trees, strict=True)])
    return _M(models, best)


def fit_gbdt(Xtr, Ytr, n_trees, seed, sample_weight=None):
    """LightGBM per target with a fixed number of rounds (the final fit)."""
    import lightgbm as lgb
    models = []
    for j in range(Ytr.shape[1]):
        m = lgb.LGBMRegressor(n_estimators=int(n_trees[j]), random_state=seed, **GBDT_PARAMS)
        m.fit(Xtr, Ytr[:, j], sample_weight=sample_weight)
        models.append(m)
    return _Multi(models, list(n_trees))


def select_gbdt_trees(data, train_cities, seed=0, n_inner=None):
    """Median best iteration per target over the inner folds."""
    best = []
    for tr, va in inner_splits(train_cities, n_inner, seed):
        Xtr, Ytr, Xva, Yva = data.fold(tr, va)
        best.append(fit_gbdt_es(Xtr, Ytr, Xva, Yva, seed).n_trees)
    return [max(1, int(np.median([b[j] for b in best]))) for j in range(len(best[0]))]


MLP_MAX_EPOCHS, MLP_PATIENCE = 300, 20


def fit_mlp_es(Xtr, Ytr, Xva, Yva, alpha, seed, scale, max_epochs=MLP_MAX_EPOCHS,
               patience=MLP_PATIENCE):
    """Two hidden layers of 256, targets standardised on the training tiles,
    trained epoch by epoch with early stopping on the validation city.
    Returns (model, best_epoch)."""
    from sklearn.neural_network import MLPRegressor
    from threadpoolctl import threadpool_limits
    mu, sd = Ytr.mean(0), Ytr.std(0)
    m = MLPRegressor(hidden_layer_sizes=(256, 256), alpha=alpha, max_iter=1,
                     random_state=seed, warm_start=False)
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    for ep in range(1, max_epochs + 1):
        with threadpool_limits(1):      # small matrices: BLAS threads only cost time
            m.partial_fit(Xtr, (Ytr - mu) / sd)
        e = selection_loss(m.predict(Xva) * sd + mu, Yva, scale)
        if e < best - 1e-6:
            best, best_ep, bad = e, ep, 0
            best_state = ([w.copy() for w in m.coefs_], [b.copy() for b in m.intercepts_])
        else:
            bad += 1
            if bad >= patience:
                break
    m.coefs_, m.intercepts_ = best_state
    return _MlpModel(m, mu, sd), best_ep


class _MlpModel:
    def __init__(self, m, mu, sd):
        self.m, self.mu, self.sd = m, mu, sd

    def predict(self, X):
        return self.m.predict(X) * self.sd + self.mu


def fit_mlp(Xtr, Ytr, alpha, epochs, seed):
    """Final MLP fit for a fixed number of epochs."""
    from sklearn.neural_network import MLPRegressor
    mu, sd = Ytr.mean(0), Ytr.std(0)
    from threadpoolctl import threadpool_limits
    m = MLPRegressor(hidden_layer_sizes=(256, 256), alpha=alpha, max_iter=1, random_state=seed)
    with threadpool_limits(1):
        for _ in range(int(epochs)):
            m.partial_fit(Xtr, (Ytr - mu) / sd)
    return _MlpModel(m, mu, sd)


def select_mlp(data, train_cities, seed=0, n_inner=None):
    """(alpha, epochs) from the inner folds: alpha with the smallest mean inner
    loss at its best epoch; epochs = median best epoch for that alpha."""
    scale = data.pool(train_cities)[1].std(0)
    loss = {a: [] for a in MLP_ALPHA_GRID}
    ep = {a: [] for a in MLP_ALPHA_GRID}
    for tr, va in inner_splits(train_cities, n_inner, seed):
        Xtr, Ytr, Xva, Yva = data.fold(tr, va)
        for a in MLP_ALPHA_GRID:
            m, e = fit_mlp_es(Xtr, Ytr, Xva, Yva, a, seed, scale)
            loss[a].append(selection_loss(m.predict(Xva), Yva, scale))
            ep[a].append(e)
    mean = {a: float(np.mean(v)) for a, v in loss.items()}
    a = min(mean, key=mean.get)
    return a, max(1, int(np.median(ep[a]))), mean


def importance_weights(Xs, Xt, clip_q=0.99):
    """Density-ratio weights p_target(v) / p_source(v) for source tiles from a
    probabilistic domain classifier (logistic regression on the descriptors;
    only the unlabelled target descriptors are used). Clipped at the clip_q
    quantile and normalised to mean one (Shimodaira 2000)."""
    from sklearn.linear_model import LogisticRegression
    X = np.r_[Xs, Xt]
    d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, d)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p) * len(Xs) / len(Xt)
    w = np.minimum(w, np.quantile(w, clip_q))
    return w / w.mean()


def coral(Xs, Xt, eps=1e-2):
    """CORAL (Sun et al. 2016): re-colour the source descriptors with the
    target covariance, Xs C_s^(-1/2) C_t^(1/2); both covariances are
    regularised by eps times their mean variance times the identity."""
    def msqrt(C, inv):
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-12)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    eye = np.eye(Xs.shape[1])
    Cs, Ct = np.cov(Xs, rowvar=False), np.cov(Xt, rowvar=False)
    Cs = Cs + eps * np.trace(Cs) / len(Cs) * eye
    Ct = Ct + eps * np.trace(Ct) / len(Ct) * eye
    return Xs @ msqrt(Cs, True) @ msqrt(Ct, False)


def fit_gp_ard(Xtr, Ytr, seed, n_hyper=600, maxiter=50):
    """Gaussian process with one length scale per descriptor (ARD) per target.
    Hyper-parameters maximise the marginal likelihood on a random subset of
    n_hyper source tiles (L-BFGS-B, at most maxiter iterations); the posterior
    mean then uses every source tile. No validation data are needed."""
    import os
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    if os.environ.get("TERRANET_FAST"):     # smoke tests only
        n_hyper, maxiter = min(n_hyper, 100), min(maxiter, 5)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xtr), min(n_hyper, len(Xtr)), replace=False)
    mu, sd = Ytr.mean(0), Ytr.std(0)
    Z = (Ytr - mu) / sd
    models, scales = [], []
    for j in range(Ytr.shape[1]):
        k = (ConstantKernel(1.0, (1e-2, 1e2))
             * RBF(np.ones(Xtr.shape[1]), (1e-2, 1e3))
             + WhiteKernel(0.5, (1e-3, 1e1)))
        def opt(obj, x0, bounds):
            from scipy.optimize import minimize
            r = minimize(obj, x0, jac=True, bounds=bounds, method="L-BFGS-B",
                         options={"maxiter": maxiter})
            return r.x, r.fun
        gp = GaussianProcessRegressor(k, optimizer=opt, normalize_y=False, random_state=seed)
        with warnings.catch_warnings():     # length scales at the bounds are
            warnings.simplefilter("ignore", ConvergenceWarning)   # expected (ARD)
            gp.fit(Xtr[sub], Z[sub, j])
        full = GaussianProcessRegressor(gp.kernel_, optimizer=None, normalize_y=False)
        full.fit(Xtr, Z[:, j])
        models.append(full)
        scales.append(gp.kernel_.k1.k2.length_scale)

    class _M:
        length_scales = scales

        def predict(self, X):
            return np.column_stack([m.predict(X) for m in models]) * sd + mu
    return _M()


def facility_location(X, k):
    """Greedy survey design: k rows of X maximising the facility-location
    objective F(T) = sum_i max_{j in T} exp(-||x_i - x_j||^2 / (2 h^2)), with h
    the median pairwise distance. F is monotone submodular, so the greedy set
    attains at least (1 - 1/e) of the optimum (Nemhauser et al. 1978)."""
    d2 = np.maximum((X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * X @ X.T, 0.0)
    h2 = np.median(d2[np.triu_indices(len(X), 1)]) if len(X) > 1 else 1.0
    Sim = np.exp(-d2 / (2 * max(h2, 1e-12)))
    cur = np.zeros(len(X))
    chosen = []
    for _ in range(min(k, len(X))):
        gain = np.maximum(Sim - cur[:, None], 0.0).sum(0)
        gain[chosen] = -1.0
        j = int(np.argmax(gain))
        chosen.append(j)
        cur = np.maximum(cur, Sim[:, j])
    return np.array(chosen, int)
