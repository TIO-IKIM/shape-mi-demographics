"""Classical (correspondence-free) baselines + cross-validated evaluation.

These are the must-beat models for the deep multi-task net, and they are what
the size-vs-shape and per-organ attribution analyses run on (fast, on CPU).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mean_absolute_error, r2_score


# --- Classifier factory: returns a fresh (unfitted) pipeline per call ---
# Pipeline wraps imputer+scaler+model so they fit ONLY on training data each fold (no leakage).
def _clf(kind: str):
    if kind == "logreg":
        # Linear baseline; class_weight="balanced" handles sex-ratio imbalance; high max_iter for convergence.
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        # 400 trees for stable votes; class_weight="balanced" up-weights the minority class.
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=8)),
        ])
    if kind == "xgb":
        from xgboost import XGBClassifier
        # Moderate depth + subsampling to regularize; hist method is fast on tabular data.
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", n_jobs=8,
        )
    raise ValueError(kind)


# --- Regressor factory: mirrors _clf but for continuous targets (age) ---
# Same Pipeline-ensures-no-leakage pattern as _clf.
def _reg(kind: str):
    if kind == "ridge":
        # Linear baseline; alpha=10 adds moderate L2 regularization to handle collinear shape descriptors.
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=10.0)),
        ])
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        # 400 trees for stability; no class_weight needed for regression.
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestRegressor(n_estimators=400, n_jobs=8)),
        ])
    if kind == "xgb":
        from xgboost import XGBRegressor
        # Same regularization strategy as the classifier variant.
        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=8,
        )
    raise ValueError(kind)


# "Out-of-fold" (OOF): every sample gets a prediction from a model that never saw it during training.
# This gives one honest prediction per sample across the full dataset, enabling bootstrap CIs on
# the complete cohort rather than per-fold subsets.
def oof_classification(X, y, kind="xgb", n_splits=5, seed=0):
    """Out-of-fold predicted probabilities (for bootstrap CIs)."""
    y = np.asarray(y); m = ~np.isnan(y)  # Drop subjects with missing labels (partially labeled cohort).
    X, y = X.loc[m], y[m].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = np.zeros(len(y))  # Will be filled fold-by-fold; each index written exactly once.
    for tr, te in skf.split(X, y):
        mdl = _clf(kind); mdl.fit(X.iloc[tr], y[tr])  # Fresh pipeline per fold -- no data leakage.
        prob[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    return y, prob


# Same OOF pattern for regression (age). Uses plain KFold since target is continuous.
def oof_regression(X, y, kind="xgb", n_splits=5, seed=0):
    y = np.asarray(y, float); m = ~np.isnan(y)  # NaN = missing age label.
    X, y = X.loc[m], y[m]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        mdl = _reg(kind); mdl.fit(X.iloc[tr], y[tr]); pred[te] = mdl.predict(X.iloc[te])
    return y, pred


# Compute a 95% bootstrap confidence interval for any sklearn-compatible metric.
# Resamples with replacement; try/except silently skips degenerate draws (e.g. single-class samples for AUC).
def bootstrap_ci(y_true, y_score, metric, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed); n = len(y_true); vals = []; skipped = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)  # Sample-with-replacement indices.
        try:
            vals.append(metric(y_true[idx], y_score[idx]))
        except (ValueError, IndexError):
            skipped += 1
    if skipped:
        import warnings
        warnings.warn(f"bootstrap_ci: {skipped}/{n_boot} degenerate resamples skipped")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


# Stratified K-fold CV for classification (sex, pathology).
# Returns mean +/- std of balanced accuracy and AUC across folds.
def cv_classification(X: pd.DataFrame, y: np.ndarray, kind: str = "xgb",
                      n_splits: int = 5, seed: int = 0) -> dict:
    y = np.asarray(y)
    m = ~np.isnan(y)  # Mask out subjects with missing labels.
    X, y = X.loc[m], y[m].astype(int)
    # Clamp n_splits to the minority-class count so every fold has both classes.
    n_splits = min(n_splits, np.bincount(y).min()) if len(np.unique(y)) > 1 else 2
    n_splits = max(2, n_splits)  # Need at least 2 folds for meaningful CV.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        model = _clf(kind)  # Fresh pipeline each fold -- scaler/imputer fit only on training fold.
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(balanced_accuracy_score(y[te], pred))
        if len(np.unique(y[te])) > 1:  # AUC undefined with single-class test fold.
            aucs.append(roc_auc_score(y[te], proba))
    return {"n": int(m.sum()), "balanced_acc": float(np.mean(accs)),
            "balanced_acc_std": float(np.std(accs)),
            "auc": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std": float(np.std(aucs)) if aucs else float("nan")}


# Single train/test split evaluation (no CV). Used for cross-domain and LOIO analyses
# where the split is defined externally (e.g. by institution), not by random folding.
def fit_eval_classification(Xtr, ytr, Xte, yte, kind: str = "xgb") -> dict:
    """Train on one split, evaluate on another (used for cross-domain transfer)."""
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)  # Handle partially-labeled subjects.
    Xtr, ytr = Xtr.loc[mtr], ytr[mtr].astype(int)
    Xte, yte = Xte.loc[mte], yte[mte].astype(int)
    if len(np.unique(ytr)) < 2 or len(Xte) == 0:  # Bail if training set is single-class or test is empty.
        return {"n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                "balanced_acc": float("nan"), "auc": float("nan")}
    model = _clf(kind)  # Pipeline ensures scaler fits on train only -- no test leakage.
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
    return {"n_train": int(len(ytr)), "n_test": int(len(yte)),
            "balanced_acc": float(balanced_accuracy_score(yte, pred)), "auc": float(auc)}


# Regression counterpart of fit_eval_classification -- same single-split, no-CV pattern.
def fit_eval_regression(Xtr, ytr, Xte, yte, kind: str = "xgb") -> dict:
    ytr, yte = np.asarray(ytr, float), np.asarray(yte, float)
    mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
    Xtr, ytr = Xtr.loc[mtr], ytr[mtr]
    Xte, yte = Xte.loc[mte], yte[mte]
    if len(Xtr) == 0 or len(Xte) == 0:
        return {"n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                "mae": float("nan"), "r2": float("nan")}
    model = _reg(kind)  # Pipeline: imputer+scaler fit on train only.
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"n_train": int(len(ytr)), "n_test": int(len(yte)),
            "mae": float(mean_absolute_error(yte, pred)), "r2": float(r2_score(yte, pred))}


# Plain K-fold CV for regression (age). StratifiedKFold is not applicable to continuous targets.
# Same leak-free pattern: fresh pipeline per fold ensures scaler/imputer fit only on train.
def cv_regression(X: pd.DataFrame, y: np.ndarray, kind: str = "xgb",
                  n_splits: int = 5, seed: int = 0) -> dict:
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)  # Drop subjects without an age label.
    X, y = X.loc[m], y[m]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, r2s = [], []
    for tr, te in kf.split(X):
        model = _reg(kind)  # Fresh pipeline each fold.
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        maes.append(mean_absolute_error(y[te], pred))
        r2s.append(r2_score(y[te], pred))
    return {"n": int(m.sum()), "mae": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "r2": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}
