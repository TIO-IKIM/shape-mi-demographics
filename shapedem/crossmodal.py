"""Cross-modality for Paper A: does the shape->demographics signal hold on MRI,
and does it transfer between CT and MRI? Uses TotalSegmentator-MRI (298 MRI,
same layout/metadata as the CT release) and the identical shape pipeline, so CT
and MRI descriptors live in the same feature space and can be compared directly.
"""
from __future__ import annotations
import os
import re
import csv
import io
import tempfile

import numpy as np
import pandas as pd

from .config import load_config
from .data.totalseg import Source
from .shapes.extract import mask_to_pointcloud, save_shape
from .pipeline import feature_path

MR_KEY = "totalseg_mr"
SEX_MAP = {"m": 0, "f": 1}


def _mr_source(cfg, backend="local"):
    ts = cfg["datasets"][MR_KEY]
    return Source(backend, url=ts["zip_url"], local_zip=ts["local_zip"], meta_member=ts["meta_member"])


def download_mr(cfg):
    import subprocess
    ts = cfg["datasets"][MR_KEY]
    zp = ts["local_zip"]; os.makedirs(os.path.dirname(zp), exist_ok=True)
    try:
        import zipfile
        if len(zipfile.ZipFile(zp).namelist()) > 1000:
            print("[mr] zip present"); return zp
    except Exception:
        pass
    print("[mr] downloading TS-MRI ...")
    subprocess.run(["curl", "-L", "-C", "-", "--retry", "8", "--retry-all-errors",
                    "-o", zp, ts["zip_url"]], check=True)
    return zp


def extract_mr(cfg, organs):
    src = _mr_source(cfg, "local")
    meta = src.read_meta()
    subjects = src.list_subjects()
    seg_fmt = cfg["datasets"][MR_KEY]["seg_member_fmt"]
    with src:
        for i, subj in enumerate(subjects):
            members = {o: seg_fmt.format(subject=subj, organ=o) for o in organs}
            todo = {o: m for o, m in members.items()
                    if not os.path.exists(feature_path(cfg, MR_KEY, subj, o))}
            if not todo:
                continue
            tmp = tempfile.mkdtemp(prefix=f"mr_{subj}_", dir=cfg["paths"]["work_dir"])
            try:
                got = src.extract_members(list(todo.values()), tmp)
                for o, m in todo.items():
                    p = got.get(m)
                    if p is None:
                        continue
                    try:
                        sh = mask_to_pointcloud(p, n_points=cfg["extract"]["n_points"],
                                                min_voxels=cfg["extract"]["min_voxels"],
                                                boundary_tol=cfg["extract"]["boundary_tol"])
                    except Exception:
                        sh = None
                    if sh is not None:
                        save_shape(feature_path(cfg, MR_KEY, subj, o), sh)
            finally:
                import shutil; shutil.rmtree(tmp, ignore_errors=True)
            if (i + 1) % 50 == 0:
                print(f"[mr] {i+1}/{len(subjects)}")
    # labels
    rows = []
    for _, r in meta.iterrows():
        g = (str(r.get("gender", "")).strip().lower())
        rows.append({"image_id": r["image_id"].strip(),
                     "y_sex": SEX_MAP.get(g, float("nan")),
                     "y_age": pd.to_numeric(r.get("age"), errors="coerce"),
                     "institute": str(r.get("institute", "")).strip()})
    lab = pd.DataFrame(rows)
    lab.to_csv(os.path.join(cfg["paths"]["results_dir"], "totalseg_mr_labels.csv"), index=False)
    print(f"[mr] labels: {len(lab)}; sex {int((lab.y_sex==1).sum())}F/{int((lab.y_sex==0).sum())}M; "
          f"age available {int(lab.y_age.notna().sum())}")
    return lab


def _matrix(cfg, dataset, labels_csv, organs):
    from .features import assemble_descriptors
    from .analysis.core import subjects_with_features
    lab = pd.read_csv(os.path.join(cfg["paths"]["results_dir"], labels_csv)).set_index("image_id")
    subs = subjects_with_features(cfg, dataset)
    X = assemble_descriptors(cfg, dataset, subs, organs)
    y = lab.reindex(X.index)
    return X, y


def analyze_and_transfer(cfg, organs=None):
    import json
    from .features import organ_presence
    from .train.baselines import cv_classification, cv_regression, fit_eval_classification, fit_eval_regression
    rdir = cfg["paths"]["results_dir"]
    organs = organs or list(cfg["organs"])
    Xc, yc = _matrix(cfg, "totalseg", "totalseg_labels.csv", organs)
    Xm, ym = _matrix(cfg, MR_KEY, "totalseg_mr_labels.csv", organs)
    cols = [c for c in Xc.columns if not c.endswith("__present") and c in Xm.columns]
    pres = organ_presence(cfg, MR_KEY, ym.index.tolist(), organs)
    out = {"n_ct": int(len(Xc)), "n_mr": int(len(Xm)), "mr_presence": pres.to_dict()}
    # within-MRI 5-fold
    out["mr_sex_cv"] = cv_classification(Xm[cols], ym["y_sex"].to_numpy(), kind="xgb")
    out["mr_age_cv"] = cv_regression(Xm[cols], ym["y_age"].to_numpy(), kind="xgb")
    # cross-modality transfer
    out["ct_to_mr_sex"] = fit_eval_classification(Xc[cols], yc["y_sex"].to_numpy(),
                                                  Xm[cols], ym["y_sex"].to_numpy(), kind="xgb")
    out["mr_to_ct_sex"] = fit_eval_classification(Xm[cols], ym["y_sex"].to_numpy(),
                                                  Xc[cols], yc["y_sex"].to_numpy(), kind="xgb")
    out["ct_to_mr_age"] = fit_eval_regression(Xc[cols], yc["y_age"].to_numpy(),
                                              Xm[cols], ym["y_age"].to_numpy(), kind="xgb")
    json.dump(out, open(os.path.join(rdir, "crossmodal_results.json"), "w"), indent=2)
    print("[crossmodal]", json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or k.endswith("sex") or k.endswith("age") or "cv" in k}, indent=2))
    return out


def write_tables(cfg):
    import json
    rdir = cfg["paths"]["results_dir"]
    tdir = os.path.join(rdir, "tables")
    os.makedirs(tdir, exist_ok=True)
    cm = json.load(open(os.path.join(rdir, "crossmodal_results.json")))
    with open(os.path.join(tdir, "crossmodal.tex"), "w") as f:
        f.write("\\begin{tabular}{lrc}\n\\toprule\nSetting & $n$ & Sex AUC \\\\\n\\midrule\n")
        f.write(f"Within MRI (5-fold CV) & {cm['mr_sex_cv']['n']} & "
                f"{cm['mr_sex_cv']['auc']:.3f}\\,$\\pm$\\,{cm['mr_sex_cv']['auc_std']:.3f} \\\\\n")
        f.write(f"CT $\\rightarrow$ MRI (transfer) & {cm['ct_to_mr_sex']['n_test']} & {cm['ct_to_mr_sex']['auc']:.3f} \\\\\n")
        f.write(f"MRI $\\rightarrow$ CT (transfer) & {cm['mr_to_ct_sex']['n_test']} & {cm['mr_to_ct_sex']['auc']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    macs = "".join([
        f"\\newcommand{{\\MRn}}{{{cm['n_mr']}}}\n",
        f"\\newcommand{{\\MRnSex}}{{{cm['mr_sex_cv']['n']}}}\n",
        f"\\newcommand{{\\MRsexAUC}}{{{cm['mr_sex_cv']['auc']:.3f}}}\n",
        f"\\newcommand{{\\MRageMAE}}{{{cm['mr_age_cv']['mae']:.1f}}}\n",
        f"\\newcommand{{\\CTtoMRsex}}{{{cm['ct_to_mr_sex']['auc']:.3f}}}\n",
        f"\\newcommand{{\\MRtoCTsex}}{{{cm['mr_to_ct_sex']['auc']:.3f}}}\n",
    ])
    with open(os.path.join(rdir, "crossmodal_macros.tex"), "w") as f:
        f.write("% AUTO-GENERATED\n" + macs)
    print(f"[crossmodal] tables/macros -> {rdir}")
