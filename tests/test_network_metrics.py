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


def test_city_includes_unreached_cells_of_the_lattice():
    """A solver cell no site reaches is a coverage hole of the truth, not a
    missing pixel."""
    import pandas as pd

    from terranet.data.sionna_gen.geo import LocalFrame
    from terranet.data.tiling import grid_frame, region_from_points
    frame = LocalFrame.from_center(52.37, 4.90)
    g = np.arange(-95.0, 100.0, 10.0)                  # 20 x 20 cells of 10 m
    px, py = np.meshgrid(g, g, indexing="ij")
    px, py = px.ravel(), py.ravel()
    hole = (np.abs(px) < 15) & (np.abs(py) < 15)       # 4 cells no site reaches
    rows = []
    for b, (bx, by) in enumerate([(-300.0, 0.0), (300.0, 0.0)]):
        d = np.hypot(px - bx, py - by)
        lat, lon = frame.to_geo(px[~hole], py[~hole])
        blat, blon = frame.to_geo(bx, by)
        rows.append(pd.DataFrame({"bs_id": f"bs{b}", "rx_lat": lat, "rx_lon": lon, "rx_h": 1.5,
                                  "tx_lat": float(blat), "tx_lon": float(blon), "tx_h": 30.0,
                                  "dist_m": d[~hole], "pathloss_db": 80 + 30 * np.log10(d[~hole])}))
    meas = pd.concat(rows, ignore_index=True)
    grid = grid_frame(region_from_points(meas.rx_lon.to_numpy(), meas.rx_lat.to_numpy()), 100.0)
    tiles = pd.DataFrame({"gamma": np.full(len(grid), 3.0), "pl0": np.full(len(grid), 100.0)})
    c = City(meas, grid, tiles, 100.0, frame)
    assert c.true_pl.shape == (400, 2) and c.n_unreached == 4
    r = c.evaluate(c.true_pl)
    assert r["cov_true_140"] < 1.0 and r["assoc_acc"] == 1.0 and r["reach_acc"] == 1.0
    # a predictor that reaches the holes is penalised in reach and link precision
    full = np.where(np.isfinite(c.true_pl), c.true_pl, 120.0)
    r2 = c.evaluate(full)
    assert r2["reach_acc"] < 1.0 and r2["link_precision"] < 1.0 and r2["pl_rmse_cens_db"] > 0
    # masks restrict the metrics
    m = c.tile == c.tile[0]
    assert c.evaluate(c.true_pl, mask=m)["n_pixels"] == int(m.sum())
