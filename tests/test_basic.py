"""Unit tests for the core shape pipeline (run with: pytest -q)."""
import numpy as np
import pandas as pd

from shapedem.shapes.extract import points_from_mask
from shapedem.shapes.descriptors import shape_descriptors, SIZE_PREFIX, SHAPE_PREFIX
from shapedem.shapes.ssm import umeyama
from shapedem.train.baselines import cv_classification, cv_regression
from shapedem.config import load_config


# Creates a synthetic binary sphere mask -- avoids dependency on real CT data for shape tests.
def _sphere(r=12, shape=(48, 48, 48)):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = np.array(shape) / 2.0
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r


# Verifies the point cloud has exactly n_points rows and is not flagged as truncated; failure means resampling or flag logic is broken.
def test_points_from_mask_shape_and_flag():
    sh = points_from_mask(_sphere(), np.ones(3), np.eye(4), n_points=512,
                          min_voxels=50, boundary_tol=10 ** 9)
    assert sh is not None
    assert sh["points"].shape == (512, 3)
    assert int(sh["truncated"]) == 0
    assert float(sh["volume_mm3"]) > 0


# Edge case: a mask with only 1 voxel (below min_voxels=50) must be rejected to avoid noisy shape features.
def test_min_voxels_returns_none():
    m = np.zeros((20, 20, 20), bool); m[0, 0, 0] = True
    assert points_from_mask(m, np.ones(3), np.eye(4), min_voxels=50) is None


# r=25 in a 40-cubed volume exceeds the half-extent (20), so the sphere is clipped by the volume boundary and must be flagged.
def test_truncation_detected_at_boundary():
    big = _sphere(r=25, shape=(40, 40, 40))     # radius > half-extent -> clipped at the faces
    sh = points_from_mask(big, np.ones(3), np.eye(4), boundary_tol=20)
    assert int(sh["truncated"]) == 1


# A sphere should yield sphericity ~1 and elongation >0.85 (near-isotropic axes); failure means the descriptor math is broken.
def test_sphere_descriptors_sane():
    d = shape_descriptors(points_from_mask(_sphere(), np.ones(3), np.eye(4),
                                           n_points=2048, boundary_tol=10 ** 9))
    assert 0.8 < d["shape_sphericity"] <= 1.05   # sphere ~ 1 (marching-cubes facets)
    assert d["shape_elong_21"] > 0.85            # near-isotropic
    assert any(k.startswith(SIZE_PREFIX) for k in d)
    assert any(k.startswith(SHAPE_PREFIX) for k in d)


# Umeyama alignment must recover known scale, rotation, and translation; QR decomposition produces a valid rotation matrix from random data.
def test_umeyama_recovers_similarity():
    rng = np.random.RandomState(0)
    src = rng.randn(100, 3)
    A = rng.randn(3, 3); R, _ = np.linalg.qr(A)       # QR gives a proper rotation (det=+1 after sign fix)
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    s, t = 2.0, np.array([1.0, -2.0, 3.0])
    dst = (s * (R @ src.T).T) + t
    s2, R2, t2 = umeyama(src, dst)
    assert abs(s2 - s) < 1e-6
    assert np.allclose(R2, R, atol=1e-6)
    assert np.allclose(t2, t, atol=1e-5)


# Two well-separated Gaussians must yield AUC>0.9; failure means the CV classification pipeline is broken, not the data.
def test_cv_classification_separable():
    rng = np.random.RandomState(0); n = 80
    X = pd.DataFrame({"f0": np.r_[rng.normal(0, 1, n // 2), rng.normal(4, 1, n // 2)]})
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    assert cv_classification(X, y, kind="logreg", n_splits=4)["auc"] > 0.9


# A near-linear signal with minimal noise must give R^2>0.8; failure means the regression pipeline is broken, not the data.
def test_cv_regression_runs():
    rng = np.random.RandomState(0); n = 60
    x = rng.uniform(0, 1, n)
    X = pd.DataFrame({"f0": x + rng.normal(0, 0.01, n)})
    r = cv_regression(X, 10 * x, kind="ridge", n_splits=4)
    assert r["r2"] > 0.8


# Ensures ${SHAPEDEM_ROOT} env-var tokens are fully expanded in config paths; un-substituted tokens would break downstream I/O.
def test_config_paths_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAPEDEM_ROOT", str(tmp_path))
    cfg = load_config()
    assert str(tmp_path) in cfg["paths"]["data_dir"]
    assert "${" not in cfg["paths"]["data_dir"]      # tokens fully substituted
