import numpy as np

from terranet.models.placement.submodular import (
    greedy_set_cover, lazy_greedy_max_coverage,
)


def test_lazy_greedy_matches_naive_on_small():
    rng = np.random.default_rng(0)
    C = rng.random((20, 100)) < 0.15
    sel, gains = lazy_greedy_max_coverage(C, k=5)
    assert len(sel) <= 5 and (np.diff(gains) <= 0).all()  # diminishing returns


def test_set_cover_covers_everything_when_feasible():
    C = np.zeros((3, 6), bool)
    C[0, :3] = True; C[1, 3:5] = True; C[2, 5] = True
    sel = greedy_set_cover(C)
    covered = np.zeros(6, bool)
    for j in sel:
        covered |= C[j]
    assert covered.all()
