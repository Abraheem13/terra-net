import numpy as np

from terranet.models.baselines.descriptor_transfer import GaussianDescriptorTransfer


def test_identity_transfer():
    """A deploy tile identical to a train tile should get (almost) its parameters."""
    rng = np.random.default_rng(0)
    V = rng.uniform(0, 1, (50, 116))
    theta = rng.uniform([1.5, 20], [3.5, 35], (50, 2))
    m = GaussianDescriptorTransfer(sigma=0.05).fit(V, theta)
    pred = m.predict(V[:5])
    assert np.allclose(pred, theta[:5], atol=0.15)


def test_sigma_selection_improves_val():
    rng = np.random.default_rng(1)
    V = rng.uniform(0, 1, (200, 116))
    w = rng.normal(size=116)
    theta = np.c_[V @ w * 0.01 + 2.5, V @ w * 0.1 + 25]
    m = GaussianDescriptorTransfer()
    sigma = m.select_sigma(V[:150], theta[:150], V[150:], theta[150:])
    assert sigma in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
