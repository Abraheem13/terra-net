"""Metrics: parameter-level, measurement-level (dB), and map/coverage-level."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).sum() / np.maximum((target ** 2).sum(), 1e-12))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(((pred - target) ** 2).mean()))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def correlations(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"pearson": float(pearsonr(pred, target)[0]),
            "spearman": float(spearmanr(pred, target)[0])}


def coverage_classification(pred_db: np.ndarray, target_db: np.ndarray,
                            threshold_db: float) -> dict[str, float]:
    """Coverage-map agreement at a received-power threshold (planning-relevant)."""
    p, t = pred_db >= threshold_db, target_db >= threshold_db
    tp, fp = (p & t).sum(), (p & ~t).sum()
    fn = (~p & t).sum()
    iou = tp / max(tp + fp + fn, 1)
    return {"coverage_iou": float(iou), "coverage_acc": float((p == t).mean())}


def full_report(pred: np.ndarray, target: np.ndarray, prefix: str = "") -> dict[str, float]:
    rep = {"nmse": nmse(pred, target), "rmse": rmse(pred, target), "mae": mae(pred, target)}
    rep.update(correlations(pred.ravel(), target.ravel()))
    return {f"{prefix}{k}": v for k, v in rep.items()}
