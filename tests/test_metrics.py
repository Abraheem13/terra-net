import numpy as np

from terranet.evaluation.metrics import full_report, nmse


def test_nmse_zero_for_perfect():
    t = np.array([1.0, 2.0, 3.0])
    assert nmse(t, t) == 0.0


def test_full_report_keys():
    rng = np.random.default_rng(0)
    t = rng.normal(0, 1, 100)
    rep = full_report(t + 0.1, t, "x_")
    assert {"x_nmse", "x_rmse", "x_mae", "x_pearson", "x_spearman"} <= rep.keys()
