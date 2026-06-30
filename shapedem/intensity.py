"""Per-organ INTENSITY features from the raw CT, to answer the paper's core
question: how much demographic signal is in shape ALONE vs the full image?

For each organ we read the CT and the organ mask and summarise the Hounsfield
units inside the mask (mean, std, percentiles). These are intensity-only
features (no geometry); compared against the shape descriptors and their union.
"""
from __future__ import annotations
import os
import tempfile
import shutil

import numpy as np
import nibabel as nib
import pandas as pd

from .config import load_config
from .data import totalseg

_PCTS = (10, 25, 50, 75, 90)


def hu_stats(ct: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict:
    """Hounsfield-unit summary statistics inside a mask (pure; unit-tested)."""
    v = ct[mask]
    if v.size == 0:  # empty mask after truncation filtering
        return {}
    out = {f"{prefix}int_mean": float(v.mean()), f"{prefix}int_std": float(v.std())}
    for p, q in zip(_PCTS, np.percentile(v, _PCTS)):
        out[f"{prefix}int_p{p}"] = float(q)
    return out


def intensity_row(cfg, source, subject, organs, min_voxels=500):
    tmp = tempfile.mkdtemp(prefix=f"int_{subject}_", dir=cfg["paths"]["work_dir"])
    row = {"subject": subject}
    try:
        ct_member = cfg["datasets"]["totalseg"]["ct_member_fmt"].format(subject=subject)
        members = [ct_member] + [totalseg.seg_member(cfg, subject, o) for o in organs]
        got = source.extract_members(members, tmp)
        ct_path = got.get(ct_member)
        if ct_path is None:
            return row
        ct = np.asanyarray(nib.load(ct_path).dataobj).astype(np.float32)
        for o in organs:
            mpath = got.get(totalseg.seg_member(cfg, subject, o))
            if mpath is None:
                continue
            m = np.asanyarray(nib.load(mpath).dataobj) > 0
            if int(m.sum()) < min_voxels or m.shape != ct.shape:
                continue
            row.update(hu_stats(ct, m, prefix=f"{o}__"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return row


def _worker(config_path, subject, organs):
    cfg = load_config(config_path)
    src = totalseg.from_config(cfg, "local")
    with src:
        return intensity_row(cfg, src, subject, organs)


def extract_intensity(cfg, config_path, subjects, organs, jobs=48):
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=jobs, prefer="processes", verbose=5)(
        delayed(_worker)(config_path, s, organs) for s in subjects)
    df = pd.DataFrame(rows).set_index("subject")
    out = os.path.join(cfg["paths"]["results_dir"], "intensity_features.csv")
    df.to_csv(out)
    print(f"[intensity] {df.shape} -> {out}")
    return df


def compare_modality(cfg):
    """Shape-only vs intensity-only vs both, for sex and age (5-fold CV)."""
    from .analysis import core
    from .train.baselines import cv_classification, cv_regression
    rdir = cfg["paths"]["results_dir"]
    subjects = core.subjects_with_features(cfg, "totalseg")
    kept, _ = core.select_organs(cfg, "totalseg", subjects, list(cfg["organs"]), 0.15)
    Xs, y = core.build_matrix(cfg, "totalseg", subjects, kept)
    Xi = pd.read_csv(os.path.join(rdir, "intensity_features.csv")).set_index("subject").reindex(Xs.index)
    shape_cols = [c for c in Xs.columns if not c.endswith("__present")]
    both = Xs[shape_cols].join(Xi)
    feats = {"shape": Xs[shape_cols], "intensity": Xi, "shape+intensity": both}
    rows = []
    for name, X in feats.items():
        sex = cv_classification(X, y["y_sex"].to_numpy(), kind="xgb")
        age = cv_regression(X, y["y_age"].to_numpy(), kind="xgb")
        rows.append({"modality": name, "n_features": X.shape[1],
                     "sex_auc": sex["auc"], "sex_balacc": sex["balanced_acc"],
                     "age_mae": age["mae"], "age_r2": age["r2"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rdir, "modality_comparison.csv"), index=False)
    print(df.to_string(index=False))
    print(f"[modality] -> {os.path.join(rdir, 'modality_comparison.csv')}")
    return df
