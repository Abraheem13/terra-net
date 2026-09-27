"""End-to-end run of every analysis script on a tiny synthetic corpus."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
import synthetic_corpus  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ["02_build_tiles.py", "03_make_splits.py", "04_fit_baselines.py",
           "06_train_neural.py", "05_kshot.py", "07_label_noise.py",
           "08_degeneracy.py", "09_network_eval.py", "11_link_transfer.py",
           "13_certified_coverage.py", "14_censoring.py", "12_tile_size.py", "10_stats.py"]


def test_pipeline_runs_end_to_end(tmp_path):
    synthetic_corpus.write(tmp_path)
    synthetic_corpus.config(tmp_path)
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, str(REPO / "scripts" / s)], cwd=tmp_path,
                           capture_output=True, text=True,
                           env={**os.environ, "TERRANET_FAST": "1"})
        assert r.returncode == 0, f"{s} failed:\n{r.stderr[-3000:]}"
    T = tmp_path / "outputs" / "tables"
    loco = pd.read_csv(T / "loco_metrics.csv")
    assert set(loco.operator) >= {"median", "kernel", "ridge", "gbdt", "mlp", "knn"}
    # k = 0 of the few-shot table reproduces the zero-shot table exactly
    k = pd.read_csv(T / "kshot.csv")
    k0 = k[(k.k == 0) & (k["mode"] == "bias") & (k.sampling == "random")]
    z = loco.set_index(["fold", "operator", "seed"]).gamma_rmse
    for r in k0.itertuples():
        assert abs(z.loc[(r.fold, r.operator, r.seed)] - r.gamma_rmse) < 1e-12
    net = pd.read_csv(T / "network_metrics.csv")
    assert (net.pl_rmse_db > 0).all() and net.operator.str.contains("k10").any()
    link = pd.read_csv(T / "network_link.csv")
    assert (link.pl_rmse_db > 0).all() and set(link.operator) >= {"link", "link+k10"}
    cert = pd.read_csv(T / "certified.csv")
    assert set(cert.operator) >= {"link", "gbdt", "oracle"}
    assert (cert.false_claim <= cert.claimed + 1e-12).all()
    assert cert.resid_cov.between(0, 1).all()
    assert {"design"} <= set(k.sampling)
    ts = pd.read_csv(T / "tile_size.csv")
    assert set(ts.G) == {50.0, 100.0, 200.0}
    assert (pd.read_csv(T / "censoring.csv").censored_share >= 0).all()
