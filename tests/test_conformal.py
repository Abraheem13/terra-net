import numpy as np

from terranet.models.uq.conformal import SplitConformal


def test_coverage_close_to_nominal():
    rng = np.random.default_rng(0)
    n = 5000
    mean = rng.normal(0, 1, n)
    sigma = rng.uniform(0.5, 2.0, n)
    target = mean + rng.normal(0, 1, n) * sigma
    cp = SplitConformal(alpha=0.1).calibrate(mean[:2500], target[:2500], sigma[:2500])
    lo, hi = cp.interval(mean[2500:], sigma[2500:])
    cov = SplitConformal.empirical_coverage(lo, hi, target[2500:])
    assert abs(cov - 0.9) < 0.03
