"""Is shape good enough to *auto-label* demographic metadata for unlabelled
segmentation archives? Three views, all the constructive flip-side of the
privacy finding:

  * selective prediction  -- accuracy when we only auto-label the most confident
                             fraction (coverage), the natural labeling-tool mode;
  * calibration (ECE)     -- are the confidences trustworthy enough to threshold;
  * data efficiency       -- how many labelled subjects are needed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def selective_classification(y, p, coverages=(1.0, 0.9, 0.75, 0.5, 0.25)):
    """Accuracy over the most-confident `coverage` fraction; plus the max coverage
    achievable at >=95% / >=99% accuracy."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    order = np.argsort(-np.abs(p - 0.5))            # most confident first
    n = len(y)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        idx = order[:k]
        rows.append({"coverage": c, "n": k,
                     "accuracy": float(((p[idx] >= 0.5).astype(int) == y[idx]).mean())})

    def cov_at(thr):
        for k in range(n, 0, -1):
            idx = order[:k]
            if ((p[idx] >= 0.5).astype(int) == y[idx]).mean() >= thr:
                return k / n
        return 0.0
    return pd.DataFrame(rows), {"cov_at_95": cov_at(0.95), "cov_at_99": cov_at(0.99)}


def expected_calibration_error(y, p, n_bins=10):
    y = np.asarray(y).astype(float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf, acc, w = float(p[m].mean()), float(y[m].mean()), float(m.mean())
        ece += w * abs(acc - conf)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                     "confidence": conf, "accuracy": acc})
    return float(ece), pd.DataFrame(rows)


def learning_curve(X, y, task="clf", fractions=(0.1, 0.25, 0.5, 1.0),
                   seeds=(0, 1, 2), test_size=0.3):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    from ..train.baselines import _clf, _reg
    y = np.asarray(y, float)
    m = ~np.isnan(y)
    X, y = X.loc[m], y[m]
    rows = []
    for frac in fractions:
        vals, ntr = [], 0
        for s in seeds:
            strat = y.astype(int) if task == "clf" else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                                  random_state=s, stratify=strat)
            ntr = max(10, int(frac * len(Xtr)))
            sub = np.random.RandomState(s).choice(len(Xtr), ntr, replace=False)
            Xs, ys = Xtr.iloc[sub], ytr[sub]
            if task == "clf":
                mdl = _clf("xgb"); mdl.fit(Xs, ys.astype(int))
                vals.append(roc_auc_score(yte.astype(int), mdl.predict_proba(Xte)[:, 1]))
            else:
                mdl = _reg("xgb"); mdl.fit(Xs, ys)
                vals.append(mean_absolute_error(yte, mdl.predict(Xte)))
        rows.append({"fraction": frac, "n_train": ntr,
                     "metric_mean": float(np.mean(vals)), "metric_std": float(np.std(vals))})
    return pd.DataFrame(rows)
