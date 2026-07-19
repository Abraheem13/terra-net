import numpy as np

from terranet.evaluation.statistics import holm_bonferroni, paired_bootstrap_ci, paired_test


def test_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 10)
    b = rng.normal(0.5, 0.1, 10)
    res = paired_bootstrap_ci(a, b)
    assert res["significant"] and res["diff_mean"] > 0


def test_paired_test_runs():
    a, b = np.arange(10.0), np.arange(10.0) + 0.5
    out = paired_test(b, a)
    assert out["p_t"] < 0.05


def test_holm():
    res = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.9})
    assert res["a"] and not res["c"]
