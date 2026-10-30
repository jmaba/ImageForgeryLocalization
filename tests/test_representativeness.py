"""Metric sanity: planted signal -> AUC ~ 1, pure noise -> AUC ~ 0.5.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import FeatureSet  # noqa: E402

G, D = 24, 16


def make_fs(signal: float, n_edit=60, n_real=60, seed=0):
    rng = np.random.default_rng(seed)
    n = n_edit + n_real
    P = rng.normal(size=(n, G, G, D)).astype(np.float16)
    M = np.zeros((n, G, G), np.float32)
    for i in range(n_edit):
        y, x = rng.integers(0, G - 8, 2)
        M[i, y:y + 8, x:x + 8] = 1
        P[i, y:y + 8, x:x + 8, 0] += signal
    meta = pd.DataFrame(dict(dataset=np.where(np.arange(n) < n_edit // 2, "a", "b"),
                             sample_id=[f"img{i:04d}" for i in range(n)],
                             label=[2] * n_edit + [0] * n_real,
                             format=["png"] * n))
    Gv = P.astype(np.float32).mean((1, 2)) + signal * (meta.label.values == 2)[:, None] * np.eye(D)[0]
    return FeatureSet("synthetic", meta, Gv.astype(np.float32), P, M)


def test_planted_signal_is_found():
    fs = make_fs(signal=3.0)
    r, _, _ = R.patch_probe(fs)
    assert r["img_mean_auc"] > 0.95
    assert R.probe_image(fs)["probe_auc"] > 0.9
    zs, _ = R.intra_image_zero_shot(fs)
    assert zs["zs_proto_img_mean_auc"] > 0.8
    oc, _ = R.one_class_real(fs)
    # one-class is diluted by the 15 uninformative dims: analytic AUC = Phi(9 / sqrt(2*50)) ~= 0.82
    assert 0.75 < oc["oc_img_mean_auc"] < 0.9


def test_noise_is_chance():
    fs = make_fs(signal=0.0)
    r, _, _ = R.patch_probe(fs)
    assert abs(r["img_mean_auc"] - 0.5) < 0.08
    assert abs(R.probe_image(fs)["probe_auc"] - 0.5) < 0.15
    mat = R.patch_transfer(fs, min_imgs=10)
    assert np.all(np.abs(mat.values - 0.5) < 0.1)
