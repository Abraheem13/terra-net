"""Reimplementation of the base paper's transfer mechanism (Eqs. 7-8).

Deployment-tile parameters = Gaussian-kernel similarity-weighted average of
training-tile parameters over normalized 116-dim descriptors.
This is the primary baseline TERRA-Net must beat under identical splits.
"""
from __future__ import annotations

import numpy as np


class GaussianDescriptorTransfer:
    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma
        self.V_train: np.ndarray | None = None
        self.theta_train: np.ndarray | None = None

    def fit(self, V_train: np.ndarray, theta_train: np.ndarray) -> "GaussianDescriptorTransfer":
        self.V_train = V_train.astype(np.float64)
        self.theta_train = theta_train.astype(np.float64)
        return self

    def predict(self, V_deploy: np.ndarray, batch: int = 4096) -> np.ndarray:
        assert self.V_train is not None
        out = np.empty((len(V_deploy), self.theta_train.shape[1]))
        for s in range(0, len(V_deploy), batch):
            Vb = V_deploy[s:s + batch].astype(np.float64)
            d2 = ((Vb[:, None, :] - self.V_train[None, :, :]) ** 2).sum(-1)
            w = np.exp(-d2 / (2 * self.sigma ** 2))
            w /= np.maximum(w.sum(1, keepdims=True), 1e-30)
            out[s:s + batch] = w @ self.theta_train
        return out

    def select_sigma(self, V_tr, th_tr, V_val, th_val,
                     grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)) -> float:
        best, best_err = self.sigma, np.inf
        for s in grid:
            self.sigma = s
            self.fit(V_tr, th_tr)
            err = float(((self.predict(V_val) - th_val) ** 2).mean())
            if err < best_err:
                best, best_err = s, err
        self.sigma = best
        self.fit(V_tr, th_tr)
        return best
