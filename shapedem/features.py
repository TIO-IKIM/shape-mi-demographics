"""Assemble a per-subject descriptor matrix from saved point-cloud npz files."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .pipeline import feature_path
from .shapes.extract import load_shape
from .shapes.descriptors import shape_descriptors, SIZE_PREFIX, SHAPE_PREFIX


def is_valid(fp: str) -> bool:
    """True iff an extracted, non-truncated shape exists (cheap: reads only the flag)."""
    if not os.path.exists(fp):
        return False
    # Only read the scalar flag, not the full point cloud — fast enough for batch checks
    with np.load(fp, allow_pickle=False) as d:
        return ("truncated" not in d.files) or (int(d["truncated"]) == 0)


def load_valid(fp: str):
    """Full shape dict if present & non-truncated, else None."""
    if not os.path.exists(fp):
        return None
    sh = load_shape(fp)
    if int(sh.get("truncated", 0)) == 1:
        return None
    return sh


def assemble_descriptors(cfg, dataset: str, subjects: list[str], organs: list[str]) -> pd.DataFrame:
    """Build a wide-format DataFrame: one row per subject, columns = organ__descriptor."""
    rows = []
    for subj in subjects:
        feat: dict[str, float] = {"subject": subj}
        for organ in organs:
            sh = load_valid(feature_path(cfg, dataset, subj, organ))
            if sh is not None:
                # Flatten all descriptors into organ__descriptor columns for ML input
                for k, v in shape_descriptors(sh).items():
                    feat[f"{organ}__{k}"] = v
                feat[f"{organ}__present"] = 1.0
            else:
                # Missing organs get NaN descriptors (pandas default) + explicit flag
                feat[f"{organ}__present"] = 0.0
        rows.append(feat)
    return pd.DataFrame(rows).set_index("subject")


def select_columns(df: pd.DataFrame, regime: str) -> list[str]:
    """Filter columns by regime {size, shape, full} to test size-vs-shape hypotheses."""
    cols = []
    for c in df.columns:
        # Strip the organ__ prefix to inspect the descriptor name itself
        base = c.split("__", 1)[1] if "__" in c else c
        if base == "present":
            # Always keep presence flags — needed for imputation / masking downstream
            cols.append(c)
        elif regime == "full":
            cols.append(c)
        elif regime == "size" and base.startswith(SIZE_PREFIX):
            cols.append(c)
        elif regime == "shape" and base.startswith(SHAPE_PREFIX):
            cols.append(c)
    return cols


def organ_presence(cfg, dataset: str, subjects: list[str], organs: list[str]) -> pd.Series:
    """Fraction of subjects for which each organ produced a VALID (non-truncated) shape."""
    counts = {o: 0 for o in organs}
    for subj in subjects:
        for o in organs:
            if is_valid(feature_path(cfg, dataset, subj, o)):
                counts[o] += 1
    n = max(len(subjects), 1)
    return pd.Series({o: counts[o] / n for o in organs}).sort_values(ascending=False)
