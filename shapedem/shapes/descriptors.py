"""Correspondence-free shape descriptors from a surface point cloud.

These power the robust classical baselines and the size-vs-shape decomposition.
We deliberately separate SIZE features (scale-dependent) from SHAPE features
(computed after normalizing the point cloud to unit centroid size), so an
experiment can switch between size-only / shape-only / full feature regimes.
"""
from __future__ import annotations
import numpy as np


def centroid_size(points: np.ndarray) -> float:
    """Root sum of squared distances to the centroid (Procrustes-style scale measure)."""
    c = points.mean(axis=0)
    return float(np.sqrt(((points - c) ** 2).sum()))


def _normalized(points: np.ndarray) -> np.ndarray:
    """Center and scale to unit centroid size — removes position and scale, keeps shape."""
    c = points.mean(axis=0)
    p = points - c
    cs = np.sqrt((p ** 2).sum())
    return p / cs if cs > 0 else p


def shape_descriptors(shape: dict) -> dict[str, float]:
    """Return a flat dict of named scalar features for one organ instance.

    Keys prefixed 'size_' are scale-dependent; 'shape_' are scale-invariant.
    """
    # float64 for numerical stability in covariance / eigendecomposition
    pts = np.asarray(shape["points"], dtype=np.float64)
    vol = float(shape.get("volume_mm3", np.nan))
    area = float(shape.get("surface_area_mm2", np.nan))

    cs = centroid_size(pts)
    # Bounding-box extents sorted descending so x >= y >= z regardless of orientation
    ext = pts.max(0) - pts.min(0)
    ext_sorted = np.sort(ext)[::-1]

    # Eigenvalues of covariance on the unit-size cloud capture pure shape elongation
    q = _normalized(pts)
    cov = q.T @ q / len(q)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending: largest variance first
    ev = np.clip(ev, 1e-12, None)  # avoid division by zero for degenerate shapes

    feats: dict[str, float] = {}
    # --- SIZE features (scale-dependent, in mm / mm^2 / mm^3) ---
    feats["size_centroid"] = cs
    feats["size_volume_mm3"] = vol
    feats["size_area_mm2"] = area
    feats["size_bbox_x"], feats["size_bbox_y"], feats["size_bbox_z"] = map(float, ext_sorted)
    feats["size_bbox_diag"] = float(np.sqrt((ext ** 2).sum()))

    # --- SHAPE features (scale-invariant, all are ratios or normalized quantities) ---
    # Elongation: how much variance spreads along secondary/tertiary axes vs primary
    feats["shape_elong_21"] = float(ev[1] / ev[0])          # 1.0 = isotropic in these two axes
    feats["shape_elong_31"] = float(ev[2] / ev[0])          # near 0 = very elongated / flat
    feats["shape_flatness"] = float(ev[2] / ev[1])           # low = disc-like (thin in one axis)
    feats["shape_aniso"] = float(1.0 - ev[2] / ev[0])       # 0 = sphere, 1 = maximally anisotropic
    # Bounding-box aspect ratios — simpler geometric complement to eigenvalue ratios
    feats["shape_aspect_yx"] = float(ext_sorted[1] / ext_sorted[0]) if ext_sorted[0] > 0 else np.nan
    feats["shape_aspect_zx"] = float(ext_sorted[2] / ext_sorted[0]) if ext_sorted[0] > 0 else np.nan
    # Sphericity: ratio of sphere surface area to actual surface area at equal volume
    if vol > 0 and area > 0:
        feats["shape_sphericity"] = float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)
    else:
        feats["shape_sphericity"] = np.nan
    # SA/V normalized by V^(2/3) cancels the cubic scaling, leaving a shape-only ratio
    feats["shape_sa_vol_norm"] = float(area / vol ** (2 / 3)) if vol > 0 else np.nan
    return feats


SIZE_PREFIX = "size_"
SHAPE_PREFIX = "shape_"
