"""Torch dataset of multi-organ point clouds for the deep multi-task model.

Each sample = one subject. Per organ we provide a size-normalized point cloud
(centered, unit centroid size -> SHAPE only) plus a separate log-centroid-size
scalar (so the model can use size explicitly if desired). Missing/truncated
organs are zero-filled and flagged by an organ mask. Labels carry per-task masks
so the multi-task loss ignores absent labels.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from ..pipeline import feature_path
from ..features import load_valid

TASKS = ("sex", "age", "pathology")


class MultiOrganPC(Dataset):
    def __init__(self, cfg, dataset, subjects, organs, labels, n_points,
                 age_mean=0.0, age_std=1.0):
        self.cfg, self.dataset, self.subjects = cfg, dataset, list(subjects)
        self.organs, self.n_points = list(organs), int(n_points)
        self.labels = labels
        self.age_mean, self.age_std = float(age_mean), float(age_std)

    def __len__(self):
        return len(self.subjects)

    def _organ_tensor(self, subj, organ):
        sh = load_valid(feature_path(self.cfg, self.dataset, subj, organ))
        if sh is None:
            return np.zeros((self.n_points, 3), np.float32), 0.0, 0.0
        pts = np.asarray(sh["points"], np.float32)
        if len(pts) != self.n_points:  # resample/pad to fixed N; unseeded for augmentation -- results averaged over 3 seeds
            idx = np.random.choice(len(pts), self.n_points, replace=len(pts) < self.n_points)
            pts = pts[idx]
        c = pts.mean(0)
        p = pts - c
        cs = float(np.sqrt((p ** 2).sum()))            # centroid size -> size feature
        rms = float(np.sqrt((p ** 2).mean() * 3))      # ~organ radius; coords become O(1)
        p = p / rms if rms > 0 else p
        return p.astype(np.float32), float(np.log(cs + 1e-6)), 1.0

    def __getitem__(self, i):
        subj = self.subjects[i]
        P, S, M = [], [], []
        for o in self.organs:
            p, s, m = self._organ_tensor(subj, o)
            P.append(p); S.append(s); M.append(m)
        row = self.labels.loc[subj]
        y, ymask = {}, {}
        for t, raw in (("sex", row.get("y_sex")), ("age", row.get("y_age")),
                       ("pathology", row.get("y_pathology"))):
            v = float(raw) if raw is not None and not (isinstance(raw, float) and np.isnan(raw)) else np.nan
            if np.isnan(v):
                y[t], ymask[t] = 0.0, 0.0
            else:
                y[t] = (v - self.age_mean) / self.age_std if t == "age" else v
                ymask[t] = 1.0
        return {
            "points": torch.from_numpy(np.stack(P)),            # [O, N, 3]
            "size": torch.tensor(S, dtype=torch.float32),        # [O]
            "organ_mask": torch.tensor(M, dtype=torch.float32),  # [O]
            "y": torch.tensor([y[t] for t in TASKS], dtype=torch.float32),       # [3]
            "ymask": torch.tensor([ymask[t] for t in TASKS], dtype=torch.float32),
        }
