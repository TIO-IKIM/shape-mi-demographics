"""Tests for the intensity stage and truncation-aware feature loading."""
import numpy as np

from shapedem.intensity import hu_stats
from shapedem.features import is_valid, load_valid


# Uniform 100 HU region must yield mean=100 and std=0; failure means the HU stat aggregation is computing wrong values.
def test_hu_stats_known_values():
    ct = np.zeros((10, 10, 10), float)
    ct[2:8, 2:8, 2:8] = 100.0
    mask = ct > 0
    s = hu_stats(ct, mask, prefix="organ__")
    assert abs(s["organ__int_mean"] - 100.0) < 1e-6
    assert s["organ__int_std"] == 0.0
    assert abs(s["organ__int_p50"] - 100.0) < 1e-6
    assert set(s) == {"organ__int_mean", "organ__int_std",
                      "organ__int_p10", "organ__int_p25", "organ__int_p50",
                      "organ__int_p75", "organ__int_p90"}


# A linear gradient (0..9) must have p10 < p50 < p90; failure means percentile computation or ordering is wrong.
def test_hu_stats_gradient():
    ct = np.tile(np.arange(10.0), (10, 10, 1))   # values 0..9 along last axis
    mask = np.ones((10, 10, 10), bool)
    s = hu_stats(ct, mask)
    assert abs(s["int_mean"] - 4.5) < 1e-6
    assert s["int_p10"] < s["int_p50"] < s["int_p90"]


# Tests truncation-aware loading: non-truncated (good) loads, truncated (bad) is rejected, and missing files return False.
def test_is_valid_and_load_valid(tmp_path):
    good = {"points": np.zeros((10, 3), np.float32), "truncated": np.int8(0),
            "volume_mm3": np.float32(1.0)}
    bad = {"points": np.zeros((10, 3), np.float32), "truncated": np.int8(1),
           "volume_mm3": np.float32(1.0)}
    gp, bp = str(tmp_path / "g.npz"), str(tmp_path / "b.npz")
    np.savez_compressed(gp, **good)
    np.savez_compressed(bp, **bad)
    assert is_valid(gp) is True
    assert is_valid(bp) is False
    assert load_valid(gp) is not None
    assert load_valid(bp) is None
    assert is_valid(str(tmp_path / "missing.npz")) is False
