"""Cell-edge displacement and serving residuals of the network evaluation."""
import numpy as np

from terranet.evaluation.network import City


def grid_city(n=20, cell=10.0):
    c = object.__new__(City)
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    c.ix, c.iy, c.cell = ix.ravel(), iy.ravel(), cell
    return c


def test_edge_displacement_zero_when_identical():
    c = grid_city()
    serv = (c.ix >= 10).astype(int)
    assert c.boundary_displacement(serv, serv) == 0.0


def test_edge_displacement_of_shifted_boundary():
    c = grid_city()
    true = (c.ix >= 10).astype(int)
    pred = (c.ix >= 13).astype(int)          # boundary moved by 3 cells = 30 m
    assert np.isclose(c.boundary_displacement(true, pred), 30.0)


def test_serving_residual_reconstructs_true_serving_loss():
    c = object.__new__(City)
    c.true_pl = np.array([[100.0, 110.0], [np.inf, 120.0], [130.0, 125.0]])
    c.truth = c.summary(c.true_pl)
    pred = np.array([[105.0, 104.0], [118.0, 121.0], [126.0, 131.0]])
    ps, r, ts = c.serving_residual(pred)
    assert np.allclose(ps + r, ts)
    assert np.allclose(ts, [100.0, 120.0, 125.0])
