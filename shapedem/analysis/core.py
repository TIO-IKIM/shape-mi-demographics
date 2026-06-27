"""Paper-A analyses on the classical shape descriptors.

All functions return tidy pandas tables and are pure (no I/O); the `analyze`
CLI command persists them to experiments/.

Analyses:
  - per-organ attribution         which organ carries the most sex/age signal
  - multi-organ fusion            all organs together vs best single organ
  - size_vs_shape                 size-only / shape-only / full feature regimes
  - cross_domain                  train on the primary institute, test on the rest
  - uniqueness_curve              shape as a fingerprint (re-identifiability proxy)
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..data import totalseg
from ..features import assemble_descriptors, select_columns, organ_presence, is_valid
from ..train.baselines import (
    cv_classification, cv_regression, fit_eval_classification, fit_eval_regression,
)


# ------------------------- data loading -------------------------
# Load cached label CSV (y_sex, y_age, y_pathology, institute, split) or build it from remote metadata.
def load_labels(cfg, backend: str = "remote") -> pd.DataFrame:
    p = os.path.join(cfg["paths"]["results_dir"], "totalseg_labels.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
    else:
        df = totalseg.build_label_table(cfg, totalseg.from_config(cfg, backend).read_meta())
    return df.set_index("image_id")


# Discover which subjects have pre-computed features on disk (one directory per subject).
def subjects_with_features(cfg, dataset: str = "totalseg") -> list[str]:
    root = Path(cfg["paths"]["features_dir"]) / dataset
    return sorted(p.name for p in root.glob("s*")) if root.exists() else []


# Filter organs to those present in at least presence_min fraction of subjects (drops rare/truncated organs).
def select_organs(cfg, dataset, subjects, organs, presence_min=0.15):
    pres = organ_presence(cfg, dataset, subjects, organs)
    kept = pres[pres >= presence_min].index.tolist()
    return kept, pres


# Assemble the full feature matrix X (subjects x organ-descriptors) and align labels y by subject.
# X columns follow the pattern "{organ}__{descriptor}"; y includes y_sex, y_age, y_pathology, etc.
def build_matrix(cfg, dataset, subjects, organs):
    X = assemble_descriptors(cfg, dataset, subjects, organs)
    y = load_labels(cfg).reindex(X.index)  # Align labels to the same subject order as features.
    return X, y


# ------------------------- analyses -------------------------
# Measure how much demographic signal each organ carries independently.
# Trains a separate model per organ using only that organ's descriptors, then ranks by sex AUC.
# Only subjects where the organ is actually present (not truncated by FOV) are included.
def per_organ_attribution(X, y, organs, model="xgb") -> pd.DataFrame:
    rows = []
    for organ in organs:
        # Isolate this organ's feature columns, excluding the binary presence flag itself.
        cols = [c for c in X.columns if c.startswith(organ + "__") and not c.endswith("__present")]
        present = X[f"{organ}__present"] == 1  # Filter to subjects where this organ was segmented.
        Xo = X.loc[present, cols]
        ys = y.loc[present, "y_sex"].to_numpy()
        ya = y.loc[present, "y_age"].to_numpy()
        sex = cv_classification(Xo, ys, kind=model)
        age = cv_regression(Xo, ya, kind=model)
        rows.append({"organ": organ, "n": int(present.sum()),
                     "sex_balacc": sex["balanced_acc"], "sex_auc": sex["auc"],
                     "sex_auc_std": sex["auc_std"],
                     "age_mae": age["mae"], "age_r2": age["r2"]})
    return pd.DataFrame(rows).sort_values("sex_auc", ascending=False).reset_index(drop=True)


# All organs together: tests whether combining descriptors across organs improves over single-organ models.
def multi_organ_fusion(X, y, model="xgb") -> dict:
    cols = [c for c in X.columns if not c.endswith("__present")]  # Exclude binary presence flags.
    return {
        "n_features": len(cols),
        "sex": cv_classification(X[cols], y["y_sex"].to_numpy(), kind=model),
        "age": cv_regression(X[cols], y["y_age"].to_numpy(), kind=model),
        "pathology": cv_classification(X[cols], y["y_pathology"].to_numpy(), kind=model),
    }


# Ablation: compare size-only, shape-only, and full (size+shape) feature sets.
# Answers whether organ shape carries signal beyond what size alone explains.
def size_vs_shape(X, y, model="xgb") -> pd.DataFrame:
    rows = []
    for regime in ("size", "shape", "full"):
        cols = [c for c in select_columns(X, regime) if not c.endswith("__present")]
        sex = cv_classification(X[cols], y["y_sex"].to_numpy(), kind=model)
        age = cv_regression(X[cols], y["y_age"].to_numpy(), kind=model)
        rows.append({"regime": regime, "n_features": len(cols),
                     "sex_balacc": sex["balanced_acc"], "sex_auc": sex["auc"],
                     "sex_auc_std": sex["auc_std"], "age_mae": age["mae"],
                     "age_mae_std": age["mae_std"], "age_r2": age["r2"]})
    return pd.DataFrame(rows)


# Cross-domain generalization test: does a model trained on the primary institution
# transfer to scans from other institutions? Compares in-domain CV to out-of-domain accuracy.
# A large drop flags institution-specific confounds (scanner, protocol, demographics).
def cross_domain(X, y, model="xgb", primary="I") -> pd.DataFrame:
    """Train on the primary institute, test on the rest (out-of-domain transfer),
    alongside an in-domain reference (CV within the primary)."""
    inst = y["institute"].astype(str)
    tr = inst == primary  # Train set: primary institution only.
    te = ~tr  # Test set: all other institutions.
    cols = [c for c in X.columns if not c.endswith("__present")]
    Xc = X[cols]
    rows = []
    # In-domain baseline: CV within the primary institution (upper bound on expected performance).
    sex_in = cv_classification(Xc[tr], y.loc[tr, "y_sex"].to_numpy(), kind=model)
    age_in = cv_regression(Xc[tr], y.loc[tr, "y_age"].to_numpy(), kind=model)
    rows.append({"split": f"in_domain(CV, inst={primary})", "n": int(tr.sum()),
                 "sex_balacc": sex_in["balanced_acc"], "sex_auc": sex_in["auc"],
                 "age_mae": age_in["mae"], "age_r2": age_in["r2"]})
    # Out-of-domain: train on primary, evaluate on held-out institutions (no overlap).
    sex_out = fit_eval_classification(Xc[tr], y.loc[tr, "y_sex"].to_numpy(),
                                      Xc[te], y.loc[te, "y_sex"].to_numpy(), kind=model)
    age_out = fit_eval_regression(Xc[tr], y.loc[tr, "y_age"].to_numpy(),
                                  Xc[te], y.loc[te, "y_age"].to_numpy(), kind=model)
    rows.append({"split": "out_of_domain(test=rest)", "n": int(te.sum()),
                 "sex_balacc": sex_out["balanced_acc"], "sex_auc": sex_out["auc"],
                 "age_mae": age_out["mae"], "age_r2": age_out["r2"]})
    return pd.DataFrame(rows)


# Full ablation table: cross feature sets (volume-only, size, shape, all) with model families
# (linear, RF, XGB) plus a naive baseline (random/mean predictor) as a sanity floor.
def model_comparison(X, y) -> pd.DataFrame:
    """Ablation/alternatives table: feature sets x model families, sex AUC + age MAE."""
    full = [c for c in X.columns if not c.endswith("__present")]
    size = [c for c in select_columns(X, "size") if not c.endswith("__present")]
    shape = [c for c in select_columns(X, "shape") if not c.endswith("__present")]
    vol = [c for c in X.columns if c.endswith("__size_volume_mm3")]

    def run(name, cols, ck, rk):
        sex = cv_classification(X[cols], y["y_sex"].to_numpy(), kind=ck)
        age = cv_regression(X[cols], y["y_age"].to_numpy(), kind=rk)
        return {"method": name, "n_features": len(cols),
                "sex_auc": sex["auc"], "sex_auc_std": sex["auc_std"],
                "sex_balacc": sex["balanced_acc"],
                "age_mae": age["mae"], "age_mae_std": age["mae_std"], "age_r2": age["r2"]}

    # Naive baseline: AUC=0.5 (random), MAE=mean-absolute-deviation (predicting cohort mean).
    age_v = y["y_age"].dropna().to_numpy()
    rows = [{"method": "Naive (majority / mean age)", "n_features": 0,
             "sex_auc": 0.5, "sex_balacc": 0.5,
             "age_mae": float(np.mean(np.abs(age_v - age_v.mean()))), "age_r2": 0.0}]
    rows.append(run("Organ volumes only (XGB)", vol, "xgb", "xgb"))
    rows.append(run("Size descriptors (XGB)", size, "xgb", "xgb"))
    rows.append(run("Shape descriptors (XGB)", shape, "xgb", "xgb"))
    rows.append(run("All descriptors (linear)", full, "logreg", "ridge"))
    rows.append(run("All descriptors (RandomForest)", full, "rf", "rf"))
    rows.append(run("All descriptors (XGBoost)", full, "xgb", "xgb"))
    return pd.DataFrame(rows)


# Produce the paper's headline numbers with 95% bootstrap CIs.
# Uses OOF predictions (every subject predicted by a model that never trained on it)
# so the CIs reflect honest, full-cohort performance rather than per-fold noise.
def headline_with_ci(X, y, model="xgb") -> dict:
    """Out-of-fold predictions + bootstrap 95% CIs for the headline numbers."""
    from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
    from ..train.baselines import oof_classification, oof_regression, bootstrap_ci
    cols = [c for c in X.columns if not c.endswith("__present")]
    yt, prob = oof_classification(X[cols], y["y_sex"].to_numpy(), kind=model)
    auc, alo, ahi = bootstrap_ci(yt, prob, roc_auc_score)
    yr, pred = oof_regression(X[cols], y["y_age"].to_numpy(), kind=model)
    mae, mlo, mhi = bootstrap_ci(yr, pred, mean_absolute_error)
    r2 = float(r2_score(yr, pred))
    return {"sex_auc": auc, "sex_auc_lo": alo, "sex_auc_hi": ahi,
            "age_mae": mae, "age_mae_lo": mlo, "age_mae_hi": mhi, "age_r2": r2,
            "n_sex": int(len(yt)), "n_age": int(len(yr))}


# Leave-one-institution-out (LOIO): the strictest generalization test.
# Each institution is held out in turn; the model trains on ALL remaining institutions.
# This reveals whether shape-based predictions generalize across scanners/protocols/demographics.
# min_n filters out tiny sites where CIs would be uninformatively wide.
def leave_one_institution_out(X, y, model="xgb", min_n=40) -> pd.DataFrame:
    """Proper LOIO: hold out each sufficiently-large institute, train on the rest.
    Reports per-institute prevalence and a bootstrap 95% CI on the held-out sex AUC
    (small held-out sites give wide, high-variance intervals)."""
    from sklearn.metrics import roc_auc_score
    from ..train.baselines import _clf, fit_eval_regression, bootstrap_ci
    inst = y["institute"].astype(str)
    cols = [c for c in X.columns if not c.endswith("__present")]
    ys_all = y["y_sex"].to_numpy(); ya_all = y["y_age"].to_numpy()
    rows = []
    for name, cnt in inst.value_counts().items():
        if cnt < min_n:  # Skip institutions too small for reliable evaluation.
            continue
        te = (inst == name).to_numpy(); tr = ~te  # Held-out = this institution; train = the rest.
        ytr, yte = ys_all[tr], ys_all[te]
        mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
        if len(np.unique(ytr[mtr])) < 2 or mte.sum() == 0:  # Need both classes in train + non-empty test.
            continue
        mdl = _clf(model); mdl.fit(X[cols][tr][mtr], ytr[mtr].astype(int))  # No test data in training.
        proba = mdl.predict_proba(X[cols][te][mte])[:, 1]
        yt = yte[mte].astype(int)
        auc = roc_auc_score(yt, proba) if len(np.unique(yt)) > 1 else float("nan")
        _, lo, hi = bootstrap_ci(yt, proba, roc_auc_score)  # 95% CI on held-out AUC.
        age = fit_eval_regression(X[cols][tr], ya_all[tr], X[cols][te], ya_all[te], kind=model)
        rows.append({"held_out": name, "n_test": int(cnt),
                     "frac_female": float(np.nanmean(yte)),  # Prevalence check: sex ratio varies by site.
                     "sex_auc": float(auc), "sex_auc_lo": lo, "sex_auc_hi": hi,
                     "age_mae": age["mae"], "age_r2": age["r2"]})
    return pd.DataFrame(rows).sort_values("n_test", ascending=False).reset_index(drop=True)


# Sensitivity analysis: vary the quantization granularity (n_bins) to show that uniqueness
# is not just an artifact of high dimensionality but genuinely depends on bin resolution.
def uniqueness_sensitivity(X, organs) -> pd.DataFrame:
    """Single-organ uniqueness as a function of quantisation (defuses the
    'fingerprint is just dimensionality' critique by showing bin-count dependence)."""
    rows = []
    for nb in (2, 3, 5, 10, 20):
        c = uniqueness_curve(X, organs, n_bins=nb)
        k1 = c[c.k_organs == 1]["frac_unique"]
        rows.append({"n_bins": nb, "frac_unique_1organ": float(k1.iloc[0]) if len(k1) else float("nan")})
    return pd.DataFrame(rows)


# Check whether excluding truncated organs biases the cohort demographics.
# If valid-organ subjects differ in age/sex from the full cohort, missingness is not at random.
def truncation_bias(cfg, dataset, subjects, organs, labels) -> pd.DataFrame:
    """Is FOV-truncation filtering associated with demographics (i.e. not MAR)?
    Compare the valid-organ subpopulation's age/sex to the full cohort."""
    from ..pipeline import feature_path
    rows = []
    for o in organs:
        valid = [s for s in subjects if is_valid(feature_path(cfg, dataset, s, o))]
        sub = labels.reindex(valid)
        rows.append({"organ": o, "n_valid": len(valid),
                     "age_valid": float(sub["y_age"].mean()),
                     "frac_female_valid": float(sub["y_sex"].mean())})
    df = pd.DataFrame(rows)
    df["age_cohort"] = float(labels["y_age"].mean())
    df["frac_female_cohort"] = float(labels["y_sex"].mean())
    return df


# Report per-institute demographics to check for confounds (e.g. one site is mostly male/young).
def institute_demographics(y) -> pd.DataFrame:
    """Per-institute composition (confound check for cross-domain)."""
    g = y.groupby(y["institute"].astype(str))
    return pd.DataFrame({"n": g.size(), "frac_female": g["y_sex"].mean(),
                         "mean_age": g["y_age"].mean()}).sort_values("n", ascending=False)


# Re-identifiability / privacy analysis: can a subject be uniquely identified from their organ shapes?
# Quantizes shape descriptors into coarse bins (mimicking what an attacker might learn from a scan),
# concatenates bins across k organs into a "fingerprint", and measures what fraction of subjects
# have a unique fingerprint. Rising uniqueness with k implies shape is a linkability risk.
def uniqueness_curve(X, organs, n_bins=10, seed=0) -> pd.DataFrame:
    """Re-identifiability proxy: quantize per-organ SHAPE descriptors into bins,
    form a signature from the first k organs, and report the fraction of subjects
    whose signature is unique. Rising uniqueness with k => shape acts as a
    fingerprint (a privacy/linkability risk), without needing repeat scans."""
    rng = np.random.RandomState(seed)
    shape_cols_by_organ = {
        o: [c for c in X.columns if c.startswith(o + "__shape_")] for o in organs
    }
    rows = []
    for k in range(1, len(organs) + 1):
        use_organs = organs[:k]
        # Only include subjects where ALL k organs are present (complete signature).
        present = np.ones(len(X), dtype=bool)
        for o in use_organs:
            present &= (X[f"{o}__present"] == 1).to_numpy()
        Xk = X.loc[present]
        if len(Xk) < 5:  # Too few subjects for a meaningful uniqueness estimate.
            rows.append({"k_organs": k, "n_subjects": int(present.sum()),
                         "frac_unique": float("nan")})
            continue
        sig_cols = sum((shape_cols_by_organ[o] for o in use_organs), [])
        vals = Xk[sig_cols].to_numpy()
        # Quantile-bin each descriptor into n_bins levels to form a discrete signature.
        codes = np.zeros_like(vals, dtype=np.int32)
        for j in range(vals.shape[1]):
            col = vals[:, j]
            finite = np.isfinite(col)
            if finite.sum() < 2:
                continue
            edges = np.quantile(col[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
            codes[:, j] = np.digitize(col, edges)
        # Each subject's signature is the tuple of all bin codes across all k organs.
        sigs = [tuple(row) for row in codes]
        from collections import Counter
        cnt = Counter(sigs)
        frac_unique = float(np.mean([cnt[s] == 1 for s in sigs]))  # Fraction with no duplicate.
        rows.append({"k_organs": k, "n_subjects": int(len(Xk)),
                     "frac_unique": frac_unique})
    return pd.DataFrame(rows)
