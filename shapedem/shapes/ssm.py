"""Correspondence-based statistical shape model (SSM) per organ.

Point clouds have no inherent correspondence, so we (1) pick a template instance,
(2) similarity-align every instance to it with ICP (Umeyama, with scale -> the SSM
captures scale-free shape), resampling each to the template's point ordering, then
(3) PCA over the corresponded coordinates. Per-subject PCA coefficients are an
interpretable alternative to the hand-crafted descriptors.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

from ..pipeline import feature_path
from ..features import load_valid


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity transform (s, R, t) mapping src -> dst for corresponded points."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (s0 ** 2).sum() / len(src)
    s = (D * np.diag(S)).sum() / var_s if (with_scale and var_s > 0) else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_to_template(template: np.ndarray, inst: np.ndarray, iters: int = 6) -> np.ndarray:
    """Return inst resampled to the template's point ordering, similarity-aligned."""
    cur = inst.copy()
    for _ in range(iters):
        idx = cKDTree(cur).query(template)[1]
        s, R, t = umeyama(cur[idx], template)
        cur = (s * (R @ cur.T).T) + t
    idx = cKDTree(cur).query(template)[1]
    return cur[idx]


def build_ssm(corresponded: np.ndarray, n_modes: int = 15):
    """corresponded: [M, N, 3] -> (mean[N,3], modes[k, N*3], coeffs[M,k])."""
    data = corresponded.reshape(len(corresponded), -1)
    mean = data.mean(0)
    Xc = data - mean
    U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_modes, Vt.shape[0])
    modes = Vt[:k]
    coeffs = Xc @ modes.T
    return mean.reshape(-1, 3), modes, coeffs


def organ_ssm(cfg, dataset, subjects, organ, n_modes=15, iters=6):
    """Build an SSM for one organ; return (subject_ids, coeffs, mean, modes)."""
    insts, ids = [], []
    for s in subjects:
        sh = load_valid(feature_path(cfg, dataset, s, organ))
        if sh is not None:
            insts.append(np.asarray(sh["points"], np.float64)); ids.append(s)
    if len(insts) < n_modes + 2:
        return ids, None, None, None
    # template = instance with median centroid size
    sizes = [np.sqrt(((p - p.mean(0)) ** 2).sum()) for p in insts]
    template = insts[int(np.argsort(sizes)[len(sizes) // 2])]
    template = template - template.mean(0)
    corr = np.stack([align_to_template(template, p, iters) for p in insts])
    mean, modes, coeffs = build_ssm(corr, n_modes)
    return ids, coeffs, mean, modes


def ssm_features(cfg, dataset, subjects, organs, n_modes=15):
    """Per-subject concatenated SSM coefficients across organs (NaN where absent)."""
    import pandas as pd
    cols = {}
    for organ in organs:
        ids, coeffs, _, _ = organ_ssm(cfg, dataset, subjects, organ, n_modes)
        if coeffs is None:
            continue
        for j in range(coeffs.shape[1]):
            cols[f"{organ}__ssm{j}"] = pd.Series(coeffs[:, j], index=ids)
    df = pd.DataFrame(cols).reindex(subjects)
    df.index.name = "subject"
    return df
