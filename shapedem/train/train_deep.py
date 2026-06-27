"""Train/evaluate the multi-organ multi-task PointNet on TotalSegmentator.

Uses the dataset's official train/val/test split with **validation-based
early stopping and best-checkpoint selection** (no epoch cherry-picking), and
averages over seeds for a stable estimate. Multi-task masked loss (BCE for sex &
pathology, SmoothL1 for standardized age). Writes experiments/deep_results.json.
"""
from __future__ import annotations
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mean_absolute_error, r2_score

from ..config import load_config
from ..analysis.core import load_labels, subjects_with_features, select_organs
from ..datasets.torch_dataset import MultiOrganPC, TASKS
from ..models.multitask import MultiOrganMultiTask


# Partition subjects using the dataset's official "split" column -- no random splitting here,
# so results are comparable across runs and match what the dataset authors intended.
def _split_subjects(labels, subjects):
    s = labels.reindex(subjects)["split"].astype(str)
    return {k: [subj for subj in subjects if s.get(subj) == k] for k in ("train", "val", "test")}


# Thin wrapper: builds dataset + DataLoader in one call. age_mean/age_std come from TRAIN set only.
def _loader(cfg, subjects, organs, labels, n_points, age_mean, age_std, bs, shuffle):
    ds = MultiOrganPC(cfg, "totalseg", subjects, organs, labels, n_points, age_mean, age_std)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=8, drop_last=False)


# Collect predictions across the full loader and compute per-task metrics.
# Respects the task mask (ymask) to handle missing labels in the multi-task setup.
@torch.no_grad()
def _evaluate(model, loader, device, age_mean, age_std):
    model.eval()
    preds = {t: [] for t in TASKS}; ys = {t: [] for t in TASKS}; ms = {t: [] for t in TASKS}
    for b in loader:
        out = model(b["points"].to(device), b["size"].to(device), b["organ_mask"].to(device))
        for i, t in enumerate(TASKS):
            preds[t].append(out[t].cpu().numpy())
            ys[t].append(b["y"][:, i].numpy()); ms[t].append(b["ymask"][:, i].numpy())
    res = {}
    for t in TASKS:
        p = np.concatenate(preds[t]); y = np.concatenate(ys[t]); m = np.concatenate(ms[t]) > 0.5
        p, y = p[m], y[m]  # Keep only subjects with a valid label for this task.
        if len(y) == 0:
            res[t] = {"n": 0}; continue
        if t == "age":
            # Undo z-score normalization to report MAE in original years.
            p = p * age_std + age_mean; y = y * age_std + age_mean
            res[t] = {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p))}
        else:
            # Binary tasks (sex, pathology): sigmoid -> threshold at 0.5.
            prob = 1 / (1 + np.exp(-p)); pred = (prob >= 0.5).astype(int)
            auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
            res[t] = {"n": int(len(y)), "balanced_acc": float(balanced_accuracy_score(y, pred)), "auc": float(auc)}
    return res


# Single-seed training loop with validation-based early stopping and best-checkpoint selection.
# No test data is seen during training or checkpoint selection -- test eval happens only at the end.
def _run_once(cfg, organs, by, labels, n_points, age_mean, age_std, device,
              max_epochs, bs, emb, lr, use_size, patience, seed, encoder):
    torch.manual_seed(seed); np.random.seed(seed)
    dl_tr = _loader(cfg, by["train"], organs, labels, n_points, age_mean, age_std, bs, True)
    dl_va = _loader(cfg, by["val"], organs, labels, n_points, age_mean, age_std, bs, False)
    # Fall back to val for test if no official test split exists.
    dl_te = _loader(cfg, by["test"] or by["val"], organs, labels, n_points, age_mean, age_std, bs, False)
    model = MultiOrganMultiTask(len(organs), emb=emb, use_size=use_size, encoder=encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Per-sample losses: reduction="none" so we can mask out missing labels per task.
    bce = nn.BCEWithLogitsLoss(reduction="none"); l1 = nn.SmoothL1Loss(reduction="none")
    best_score, best_state, bad = -1e9, None, 0  # Early-stopping state.
    for ep in range(max_epochs):
        model.train()
        for b in dl_tr:
            pts, sz, om = b["points"].to(device), b["size"].to(device), b["organ_mask"].to(device)
            y, ym = b["y"].to(device), b["ymask"].to(device)
            out = model(pts, sz, om)
            loss = 0.0
            for i, t in enumerate(TASKS):
                li = (l1 if t == "age" else bce)(out[t], y[:, i])
                # Masked mean: only subjects with a valid label for this task contribute.
                loss = loss + (li * ym[:, i]).sum() / (ym[:, i].sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
        # --- Checkpoint selection on validation set only (never test) ---
        va = _evaluate(model, dl_va, device, age_mean, age_std)
        # Composite score: val sex AUC + val age R^2 (both higher-is-better).
        score = (va["sex"].get("auc", 0) or 0) + max(-1, va["age"].get("r2", -1) or -1)
        if score > best_score:
            best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:  # No improvement for `patience` epochs -- stop early.
                break
    if best_state is not None:
        model.load_state_dict(best_state)  # Revert to best validation checkpoint before test eval.
    return _evaluate(model, dl_te, device, age_mean, age_std)


# Main entry point: trains the deep multi-task model, averaging results over multiple seeds
# for stable estimates. Age z-normalization is computed from TRAIN split only (no test leakage).
def train_deep(config=None, epochs=200, bs=32, emb=128, lr=1e-3, use_size=True,
               presence_min=0.15, patience=40, seeds=(0, 1, 2),
               encoder="pointnet", n_points_override=None, out_name="deep_results.json"):
    cfg = load_config(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = load_labels(cfg)
    subjects = subjects_with_features(cfg, "totalseg")
    organs, _ = select_organs(cfg, "totalseg", subjects, list(cfg["organs"]), presence_min)
    by = _split_subjects(labels, subjects)
    if not by["val"]:
        # No official val split -- carve 10% off training (deterministic seed=0).
        rng = np.random.RandomState(0); tr = by["train"][:]; rng.shuffle(tr)
        cut = max(1, int(0.1 * len(tr))); by["val"], by["train"] = tr[:cut], tr[cut:]
    n_points = int(n_points_override) if n_points_override else cfg["extract"]["n_points"]
    # Age normalization from TRAIN only -- prevents test statistics from leaking into training.
    age_tr = labels.reindex(by["train"])["y_age"]
    age_mean, age_std = float(age_tr.mean()), float(age_tr.std() or 1.0)  # Fallback std=1 if constant.
    print(f"[deep] encoder={encoder} device={device} organs={len(organs)} n_points={n_points} "
          f"train/val/test={len(by['train'])}/{len(by['val'])}/{len(by['test'])} seeds={list(seeds)}")

    # Train independently with each seed; averaging smooths initialization variance.
    per_seed = []
    for s in seeds:
        r = _run_once(cfg, organs, by, labels, n_points, age_mean, age_std, device,
                      epochs, bs, emb, lr, use_size, patience, s, encoder)
        per_seed.append(r)
        print(f"[deep] seed {s}: sex_auc={r['sex'].get('auc'):.3f} "
              f"age_mae={r['age'].get('mae'):.2f} age_r2={r['age'].get('r2'):.3f}")

    def agg(task, key):
        # Filter NaNs via the x == x trick (NaN != NaN) before averaging across seeds.
        vals = [r[task][key] for r in per_seed if key in r[task] and r[task][key] == r[task][key]]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    metrics = {}
    for t in TASKS:
        metrics[t] = {"n": per_seed[0][t].get("n", 0)}
        for key in ("auc", "balanced_acc", "mae", "r2"):
            if any(key in r[t] for r in per_seed):
                m, sd = agg(t, key); metrics[t][key] = m; metrics[t][key + "_std"] = sd
    res = {"split_eval": "test" if by["test"] else "val", "n_organs": len(organs),
           "encoder": encoder, "use_size": use_size, "seeds": list(seeds),
           "selection": "val sexAUC+ageR2, early stop", "metrics": metrics, "per_seed": per_seed}
    out = os.path.join(cfg["paths"]["results_dir"], out_name)
    json.dump(res, open(out, "w"), indent=2)
    print("[deep] aggregate:", json.dumps(metrics, indent=2)); print(f"[deep] -> {out}")
    return res
