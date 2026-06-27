"""Command-line entrypoints for the pipeline stages.

Usage (env activated, repo on PYTHONPATH):
    python -m shapedem.cli meta      [--backend remote|local]
    python -m shapedem.cli extract   [--backend ...] [--subjects all|smoke|<n>] [--jobs N]
    python -m shapedem.cli baseline  [--subjects all|<n>] [--regime full|size|shape]
    python -m shapedem.cli smoke
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import totalseg


def _meta(cfg, backend: str) -> pd.DataFrame:
    src = totalseg.from_config(cfg, backend)
    meta = src.read_meta()
    return totalseg.build_label_table(cfg, meta)


def _balanced_subjects(labels: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    df = labels.dropna(subset=["y_sex"])
    per = max(1, n // 2)
    parts = []
    for v in (0, 1):
        sub = df[df["y_sex"] == v]
        parts.append(sub.sample(min(per, len(sub)), random_state=seed))
    out = pd.concat(parts)
    return out["image_id"].tolist()


def cmd_meta(args):
    cfg = load_config(args.config)
    labels = _meta(cfg, args.backend)
    out = os.path.join(cfg["paths"]["results_dir"], "totalseg_labels.csv")
    labels.to_csv(out, index=False)
    print(f"[meta] {len(labels)} subjects -> {out}")
    print("[meta] sex (y_sex):", labels["y_sex"].value_counts(dropna=False).to_dict())
    print("[meta] age: n=%d mean=%.1f [%.0f-%.0f]" % (
        labels["y_age"].notna().sum(), labels["y_age"].mean(),
        labels["y_age"].min(), labels["y_age"].max()))
    print("[meta] pathology (y_pathology):", labels["y_pathology"].value_counts(dropna=False).to_dict())


def _resolve_subjects(cfg, labels, spec: str, seed=0):
    if spec == "all":
        return labels["image_id"].tolist()
    if spec == "smoke":
        return _balanced_subjects(labels, cfg["smoke"]["n_subjects"], seed)
    return _balanced_subjects(labels, int(spec), seed)


def cmd_extract(args):
    from joblib import Parallel, delayed
    from .pipeline import extract_one
    cfg = load_config(args.config)
    labels = _meta(cfg, "remote")  # meta is tiny; always read remotely
    if args.subjects == "all":
        subjects = labels["image_id"].tolist()
    else:
        subjects = _resolve_subjects(cfg, labels, args.subjects)
    organs = list(cfg["organs"])
    print(f"[extract] {len(subjects)} subjects x {len(organs)} organs, jobs={args.jobs}")
    results = Parallel(n_jobs=args.jobs, prefer="processes", verbose=5)(
        delayed(extract_one)(args.config, "totalseg", s, organs, args.overwrite)
        for s in subjects
    )
    counts: dict[str, int] = {}
    for st in results:
        for v in st.values():
            key = v.split(":")[0] if v.startswith("error") else v
            counts[key] = counts.get(key, 0) + 1
    print(f"[extract] done: {counts}")


def cmd_baseline(args):
    from .features import assemble_descriptors, select_columns, organ_presence
    from .train.baselines import cv_classification, cv_regression
    cfg = load_config(args.config)
    labels = _meta(cfg, "remote").set_index("image_id")
    organs = list(cfg["organs"])
    # subjects = those with at least one extracted organ
    feat_root = Path(cfg["paths"]["features_dir"]) / "totalseg"
    subjects = sorted(p.name for p in feat_root.glob("s*")) if feat_root.exists() else []
    if args.subjects != "all":
        subjects = subjects[: int(args.subjects)]
    print(f"[baseline] {len(subjects)} subjects with features")
    pres = organ_presence(cfg, "totalseg", subjects, organs)
    print("[baseline] organ presence:\n", pres.to_string())
    X = assemble_descriptors(cfg, "totalseg", subjects, organs)
    cols = select_columns(X, args.regime)
    Xr = X[cols]
    y = labels.reindex(subjects)
    res = {
        "regime": args.regime, "n_subjects": len(subjects), "n_features": len(cols),
        "sex": cv_classification(Xr, y["y_sex"].to_numpy(), kind=args.model),
        "age": cv_regression(Xr, y["y_age"].to_numpy(), kind=args.model),
    }
    print(json.dumps(res, indent=2))
    out = os.path.join(cfg["paths"]["results_dir"], f"baseline_{args.regime}_{args.model}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"[baseline] -> {out}")


def cmd_analyze(args):
    from .analysis import core
    cfg = load_config(args.config)
    rdir = cfg["paths"]["results_dir"]
    subjects = core.subjects_with_features(cfg, "totalseg")
    organs = list(cfg["organs"])
    kept, pres = core.select_organs(cfg, "totalseg", subjects, organs, presence_min=args.presence_min)
    pres.to_csv(os.path.join(rdir, "organ_presence.csv"), header=["valid_fraction"])
    print(f"[analyze] {len(subjects)} subjects; {len(kept)}/{len(organs)} organs kept "
          f"(presence>={args.presence_min}):\n{pres.to_string()}")
    X, y = core.build_matrix(cfg, "totalseg", subjects, kept)

    attr = core.per_organ_attribution(X, y, kept, model=args.model)
    attr.to_csv(os.path.join(rdir, "attribution.csv"), index=False)
    print("[analyze] per-organ attribution:\n", attr.to_string(index=False))

    svs = core.size_vs_shape(X, y, model=args.model)
    svs.to_csv(os.path.join(rdir, "size_vs_shape.csv"), index=False)
    print("[analyze] size vs shape:\n", svs.to_string(index=False))

    cd = core.cross_domain(X, y, model=args.model, primary=cfg["domain"]["primary_institute"])
    cd.to_csv(os.path.join(rdir, "cross_domain.csv"), index=False)
    print("[analyze] cross-domain:\n", cd.to_string(index=False))

    uq = core.uniqueness_curve(X, kept)
    uq.to_csv(os.path.join(rdir, "uniqueness_curve.csv"), index=False)
    print("[analyze] uniqueness curve:\n", uq.to_string(index=False))

    fusion = core.multi_organ_fusion(X, y, model=args.model)
    json.dump(fusion, open(os.path.join(rdir, "fusion.json"), "w"), indent=2)
    print("[analyze] multi-organ fusion:\n", json.dumps(fusion, indent=2))
    print(f"[analyze] tables -> {rdir}")


def cmd_figures(args):
    from .analysis import figures
    cfg = load_config(args.config)
    figures.make_all(cfg)


def cmd_deep(args):
    from .train.train_deep import train_deep
    pts = args.points or (512 if args.encoder == "dgcnn" else None)
    out = "deep_dgcnn_results.json" if args.encoder == "dgcnn" else "deep_results.json"
    train_deep(config=args.config, epochs=args.epochs, bs=args.batch, emb=args.emb,
               lr=args.lr, use_size=not args.no_size, presence_min=args.presence_min,
               encoder=args.encoder, n_points_override=pts, out_name=out)


def cmd_intensity(args):
    from . import intensity
    from .analysis import core
    cfg = load_config(args.config)
    subjects = core.subjects_with_features(cfg, "totalseg")
    kept, _ = core.select_organs(cfg, "totalseg", subjects, list(cfg["organs"]), 0.15)
    intensity.extract_intensity(cfg, args.config, subjects, kept, jobs=args.jobs)


def cmd_modality(args):
    from . import intensity
    intensity.compare_modality(load_config(args.config))


def cmd_stats(args):
    from .analysis import core
    cfg = load_config(args.config); rdir = cfg["paths"]["results_dir"]
    subjects = core.subjects_with_features(cfg, "totalseg")
    kept, _ = core.select_organs(cfg, "totalseg", subjects, list(cfg["organs"]), 0.15)
    X, y = core.build_matrix(cfg, "totalseg", subjects, kept)
    hl = core.headline_with_ci(X, y)
    json.dump(hl, open(os.path.join(rdir, "headline_stats.json"), "w"), indent=2)
    print("[stats] headline:", json.dumps(hl, indent=2))
    loio = core.leave_one_institution_out(X, y)
    loio.to_csv(os.path.join(rdir, "loio.csv"), index=False)
    print("[stats] LOIO:\n", loio.to_string(index=False))
    demo = core.institute_demographics(y)
    demo.to_csv(os.path.join(rdir, "institute_demographics.csv"))
    print("[stats] institute demographics:\n", demo.to_string())
    uq = core.uniqueness_sensitivity(X, kept)
    uq.to_csv(os.path.join(rdir, "uniqueness_sensitivity.csv"), index=False)
    print("[stats] uniqueness vs bins:\n", uq.to_string(index=False))
    tb = core.truncation_bias(cfg, "totalseg", subjects, kept, y)
    tb.to_csv(os.path.join(rdir, "truncation_bias.csv"), index=False)
    print("[stats] truncation bias (valid subset vs cohort age %.1f, female %.2f):"
          % (y["y_age"].mean(), y["y_sex"].mean()))
    print(tb[["organ", "n_valid", "age_valid", "frac_female_valid"]].to_string(index=False))


def cmd_labeling(args):
    """Auto-labeling viability: selective prediction, calibration, data efficiency."""
    from .analysis import core, labeling
    from .train.baselines import oof_classification
    cfg = load_config(args.config); rdir = cfg["paths"]["results_dir"]
    subs = core.subjects_with_features(cfg, "totalseg")
    kept, _ = core.select_organs(cfg, "totalseg", subs, list(cfg["organs"]), 0.15)
    X, y = core.build_matrix(cfg, "totalseg", subs, kept)
    cols = [c for c in X.columns if not c.endswith("__present")]
    yt, p = oof_classification(X[cols], y["y_sex"].to_numpy(), kind="xgb")
    sel, covs = labeling.selective_classification(yt, p)
    ece, reli = labeling.expected_calibration_error(yt, p)
    lc_sex = labeling.learning_curve(X[cols], y["y_sex"].to_numpy(), task="clf")
    lc_age = labeling.learning_curve(X[cols], y["y_age"].to_numpy(), task="reg")
    sel.to_csv(os.path.join(rdir, "selective_sex.csv"), index=False)
    reli.to_csv(os.path.join(rdir, "calibration_sex.csv"), index=False)
    lc_sex.to_csv(os.path.join(rdir, "learning_curve_sex.csv"), index=False)
    lc_age.to_csv(os.path.join(rdir, "learning_curve_age.csv"), index=False)
    res = {"n": int(len(yt)), "sex_ece": ece, **covs,
           "sex_acc_full": float(sel.iloc[0]["accuracy"]),
           "sex_acc_at_50cov": float(sel[sel.coverage == 0.5]["accuracy"].iloc[0]),
           "sex_acc_at_25cov": float(sel[sel.coverage == 0.25]["accuracy"].iloc[0])}
    json.dump(res, open(os.path.join(rdir, "labeling_results.json"), "w"), indent=2)
    print("[labeling]", json.dumps(res, indent=2))
    print("[labeling] selective:\n", sel.to_string(index=False))
    print("[labeling] learning curve (sex AUC):\n", lc_sex.to_string(index=False))


def cmd_crossmodal(args):
    from . import crossmodal
    cfg = load_config(args.config)
    organs = list(cfg["organs"])
    if args.step in ("extract", "all"):
        crossmodal.download_mr(cfg); crossmodal.extract_mr(cfg, organs)
    if args.step in ("analyze", "all"):
        crossmodal.analyze_and_transfer(cfg, organs)
    if args.step in ("tables", "all"):
        crossmodal.write_tables(cfg)



def cmd_compare(args):
    """Ablation/alternatives comparison: feature sets x models + SSM + deep nets."""
    import pandas as pd
    from .analysis import core
    from .shapes.ssm import ssm_features
    from .train.baselines import cv_classification, cv_regression
    cfg = load_config(args.config)
    rdir = cfg["paths"]["results_dir"]
    subjects = core.subjects_with_features(cfg, "totalseg")
    kept, _ = core.select_organs(cfg, "totalseg", subjects, list(cfg["organs"]), args.presence_min)
    X, y = core.build_matrix(cfg, "totalseg", subjects, kept)

    comp = core.model_comparison(X, y); comp["eval"] = "5-fold CV"
    rows = comp.to_dict("records")

    print("[compare] building SSM features (ICP correspondence)...")
    Xssm = ssm_features(cfg, "totalseg", subjects, kept, n_modes=args.ssm_modes).reindex(X.index)
    sex = cv_classification(Xssm, y["y_sex"].to_numpy(), kind="xgb")
    age = cv_regression(Xssm, y["y_age"].to_numpy(), kind="xgb")
    rows.append({"method": "SSM-PCA coeffs (XGB)", "n_features": Xssm.shape[1],
                 "sex_auc": sex["auc"], "sex_balacc": sex["balanced_acc"],
                 "age_mae": age["mae"], "age_r2": age["r2"], "eval": "5-fold CV"})

    for fn, label in [("deep_results.json", "Multi-organ PointNet"),
                      ("deep_dgcnn_results.json", "Multi-organ DGCNN")]:
        p = os.path.join(rdir, fn)
        if os.path.exists(p):
            m = json.load(open(p))["metrics"]
            rows.append({"method": label, "n_features": float("nan"),
                         "sex_auc": m["sex"].get("auc"), "sex_balacc": m["sex"].get("balanced_acc"),
                         "age_mae": m["age"].get("mae"), "age_r2": m["age"].get("r2"),
                         "eval": "held-out test"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rdir, "comparison.csv"), index=False)
    print(df.to_string(index=False))
    print(f"[compare] -> {os.path.join(rdir, 'comparison.csv')}")


def cmd_tables(args):
    from .analysis import tables
    cfg = load_config(args.config)
    rdir = cfg["paths"]["results_dir"]
    tables.make_tables(cfg, rdir)


def cmd_smoke(args):
    """End-to-end correctness gate on a tiny remote subset."""
    from .pipeline import extract_subject
    from .features import assemble_descriptors, select_columns, organ_presence
    from .train.baselines import cv_classification, cv_regression
    cfg = load_config(args.config)
    labels = _meta(cfg, "remote")
    subjects = _balanced_subjects(labels, cfg["smoke"]["n_subjects"])
    organs = cfg["smoke"]["organs"]
    print(f"[smoke] {len(subjects)} subjects, organs={organs}")
    src = totalseg.from_config(cfg, "remote")
    with src:
        for i, subj in enumerate(subjects):
            st = extract_subject(cfg, src, "totalseg", subj, organs)
            print(f"[smoke] {i+1}/{len(subjects)} {subj}: {st}")
    pres = organ_presence(cfg, "totalseg", subjects, organs)
    print("[smoke] presence:\n", pres.to_string())
    X = assemble_descriptors(cfg, "totalseg", subjects, organs)
    lab = labels.set_index("image_id").reindex(subjects)
    full = X[select_columns(X, "full")]
    res = {
        "n_subjects": len(subjects), "n_features": full.shape[1],
        "presence": pres.to_dict(),
        "sex_full": cv_classification(full, lab["y_sex"].to_numpy(), kind="logreg", n_splits=3),
        "age_full": cv_regression(full, lab["y_age"].to_numpy(), kind="ridge", n_splits=3),
    }
    print(json.dumps(res, indent=2))
    out = os.path.join(cfg["paths"]["results_dir"], "smoke_results.json")
    json.dump(res, open(out, "w"), indent=2)
    # sanity assertions
    assert full.shape[1] > 0, "no features produced"
    assert pres.max() > 0, "no organ extracted for any subject"
    print(f"[smoke] OK -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser("shapedem")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("meta"); m.add_argument("--backend", default="remote"); m.set_defaults(fn=cmd_meta)
    e = sub.add_parser("extract")
    e.add_argument("--backend", default="local"); e.add_argument("--subjects", default="all")
    e.add_argument("--jobs", type=int, default=1); e.add_argument("--overwrite", action="store_true")
    e.set_defaults(fn=cmd_extract)
    b = sub.add_parser("baseline")
    b.add_argument("--subjects", default="all"); b.add_argument("--regime", default="full",
                   choices=["full", "size", "shape"]); b.add_argument("--model", default="xgb")
    b.set_defaults(fn=cmd_baseline)
    a = sub.add_parser("analyze")
    a.add_argument("--model", default="xgb"); a.add_argument("--presence_min", type=float, default=0.15)
    a.set_defaults(fn=cmd_analyze)
    f = sub.add_parser("figures"); f.set_defaults(fn=cmd_figures)
    t = sub.add_parser("tables"); t.set_defaults(fn=cmd_tables)
    d = sub.add_parser("deep")
    d.add_argument("--epochs", type=int, default=60); d.add_argument("--batch", type=int, default=32)
    d.add_argument("--emb", type=int, default=128); d.add_argument("--lr", type=float, default=1e-3)
    d.add_argument("--no_size", action="store_true"); d.add_argument("--presence_min", type=float, default=0.15)
    d.add_argument("--encoder", default="pointnet", choices=["pointnet", "dgcnn"])
    d.add_argument("--points", type=int, default=0)
    d.set_defaults(fn=cmd_deep)
    it = sub.add_parser("intensity"); it.add_argument("--jobs", type=int, default=48); it.set_defaults(fn=cmd_intensity)
    md = sub.add_parser("modality"); md.set_defaults(fn=cmd_modality)
    st = sub.add_parser("stats"); st.set_defaults(fn=cmd_stats)
    lb = sub.add_parser("labeling"); lb.set_defaults(fn=cmd_labeling)
    cm = sub.add_parser("crossmodal")
    cm.add_argument("--step", default="all", choices=["extract", "analyze", "tables", "all"])
    cm.set_defaults(fn=cmd_crossmodal)
    c = sub.add_parser("compare")
    c.add_argument("--presence_min", type=float, default=0.15); c.add_argument("--ssm_modes", type=int, default=15)
    c.set_defaults(fn=cmd_compare)
    s = sub.add_parser("smoke"); s.set_defaults(fn=cmd_smoke)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
