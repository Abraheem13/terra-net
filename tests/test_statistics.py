from terranet.evaluation.statistics import holm_bonferroni


def test_holm():
    res = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.9})
    assert res["a"] and not res["c"]


def test_holm_stops_at_first_failure():
    res = holm_bonferroni({"a": 0.001, "b": 0.03, "c": 0.02, "d": 0.9})
    # sorted: a (0.001 <= 0.05/4), c (0.02 <= 0.05/3? no) -> c, b, d rejected
    assert res["a"] and not res["c"] and not res["b"] and not res["d"]
