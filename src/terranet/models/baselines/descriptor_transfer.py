"""Base-paper Gaussian descriptor transfer (Ozyurt 2026).

Deployment-tile (gamma, PL0) = similarity-weighted average of training tiles,
weights = exp(-||v_u - v_j||^2 / 2 sigma^2). Sigma tuned on a validation set.

Memory-safe: squared distances via the (||a||^2 + ||b||^2 - 2 a.b) identity,
computed in row-batches, so peak memory is O(batch * n_train) not
O(batch * n_train * dim).
"""
from __future__ import annotations

import numpy as np


class GaussianDescriptorTransfer:
    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma
        self.V_train: np.ndarray | None = None
        self.theta_train: np.ndarray | None = None
        self._tr_sqnorm: np.ndarray | None = None

    def fit(self, V_train, theta_train):
        self.V_train = np.ascontiguousarray(V_train, dtype=np.float64)
        self.theta_train = np.ascontiguousarray(theta_train, dtype=np.float64)
        self._tr_sqnorm = (self.V_train ** 2).sum(1)      # (n_train,)
        return self

    def _sqdist(self, Vb):
        # ||a||^2 + ||b||^2 - 2 a.b  ->  (batch, n_train), no 3-D tensor
        d2 = self._tr_sqnorm[None, :] + (Vb ** 2).sum(1)[:, None] - 2.0 * (Vb @ self.V_train.T)
        return np.maximum(d2, 0.0)

    def predict(self, V_deploy, batch: int = 256) -> np.ndarray:
        assert self.V_train is not None
        V = np.ascontiguousarray(V_deploy, dtype=np.float64)
        out = np.empty((len(V), self.theta_train.shape[1]))
        inv = 1.0 / (2.0 * self.sigma ** 2)
        for s in range(0, len(V), batch):
            d2 = self._sqdist(V[s:s + batch])
            w = np.exp(-d2 * inv)
            w /= np.maximum(w.sum(1, keepdims=True), 1e-30)
            out[s:s + batch] = w @ self.theta_train
        return out

    def select_sigma(self, V_tr, th_tr, V_val, th_val,
                     grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0),
                     batch: int = 256) -> float:
        """Fit once, reuse the val<->train sqdist across all sigmas."""
        self.fit(V_tr, th_tr)
        Vv = np.ascontiguousarray(V_val, dtype=np.float64)
        th_val = np.asarray(th_val, dtype=np.float64)
        best, best_err = self.sigma, np.inf
        # precompute val->train squared distances once, in batches
        d2_blocks = [self._sqdist(Vv[s:s + batch]) for s in range(0, len(Vv), batch)]
        for sig in grid:
            inv = 1.0 / (2.0 * sig ** 2)
            err = 0.0
            for bi, s in enumerate(range(0, len(Vv), batch)):
                w = np.exp(-d2_blocks[bi] * inv)
                w /= np.maximum(w.sum(1, keepdims=True), 1e-30)
                pred = w @ self.theta_train
                err += ((pred - th_val[s:s + batch]) ** 2).sum()
            err /= th_val.size
            if err < best_err:
                best, best_err = sig, err
        self.sigma = best
        return best
