"""Turn result files in experiments/ into LaTeX tables + a macro file.

Guarantees every number in the manuscript is sourced from a committed result
file: the paper \\input{}s these tables and uses the \\newcommand macros for all
inline figures, so nothing is hand-typed.
"""
from __future__ import annotations
import json
import os
import pandas as pd


def _disp(organ: str) -> str:
    return organ.replace("_", " ")


def _tex(s) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def make_tables(cfg, out_dir: str):
    rdir = cfg["paths"]["results_dir"]
    tdir = os.path.join(out_dir, "tables")
    os.makedirs(tdir, exist_ok=True)

    attr = pd.read_csv(os.path.join(rdir, "attribution.csv"))
    svs = pd.read_csv(os.path.join(rdir, "size_vs_shape.csv")).set_index("regime")
    uq = pd.read_csv(os.path.join(rdir, "uniqueness_curve.csv"))
    fusion = json.load(open(os.path.join(rdir, "fusion.json")))
    labels = pd.read_csv(os.path.join(rdir, "totalseg_labels.csv"))

    # ---------- attribution table ----------
    with open(os.path.join(tdir, "attribution.tex"), "w") as f:
        f.write("\\begin{tabular}{lrccc}\n\\toprule\n")
        f.write("Organ & $n$ & Sex AUC & Sex bal.acc & Age $R^2$ \\\\\n\\midrule\n")
        a_sd = "sex_auc_std" in attr.columns
        for _, r in attr.iterrows():
            sd = f"\\,$\\pm$\\,{r['sex_auc_std']:.3f}" if a_sd else ""
            f.write(f"{_disp(r['organ'])} & {int(r['n'])} & {r['sex_auc']:.3f}{sd} & "
                    f"{r['sex_balacc']:.3f} & {r['age_r2']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- size vs shape table (with 5-fold std) ----------
    with open(os.path.join(tdir, "size_vs_shape.tex"), "w") as f:
        f.write("\\begin{tabular}{lrcccc}\n\\toprule\n")
        f.write("Feature set & dim & Sex AUC & Sex bal.acc & Age MAE & Age $R^2$ \\\\\n\\midrule\n")
        names = {"size": "Size only", "shape": "Shape only (scale-free)", "full": "Size $+$ shape"}
        has_sd = "sex_auc_std" in svs.columns
        for reg in ("size", "shape", "full"):
            if reg in svs.index:
                r = svs.loc[reg]
                sd = f"\\,$\\pm$\\,{r['sex_auc_std']:.3f}" if has_sd else ""
                ad = f"\\,$\\pm$\\,{r['age_mae_std']:.1f}" if has_sd else ""
                f.write(f"{names[reg]} & {int(r['n_features'])} & {r['sex_auc']:.3f}{sd} & "
                        f"{r['sex_balacc']:.3f} & {r['age_mae']:.1f}{ad} & {r['age_r2']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- cross-domain table: proper leave-one-institution-out ----------
    loio_path = os.path.join(rdir, "loio.csv")
    loio = pd.read_csv(loio_path) if os.path.exists(loio_path) else None
    with open(os.path.join(tdir, "cross_domain.tex"), "w") as f:
        if loio is None:
            f.write("% run `cli stats` to generate the LOIO table\n")
        else:
            has_ci = "sex_auc_lo" in loio.columns
            f.write("\\begin{tabular}{lrccc}\n\\toprule\n")
            f.write("Held-out institute & $n$ & \\% female & Sex AUC (95\\% CI) & Age MAE \\\\\n\\midrule\n")
            for _, r in loio.iterrows():
                ci = (f" ({r['sex_auc_lo']:.2f}--{r['sex_auc_hi']:.2f})" if has_ci else "")
                pf = f"{100*r['frac_female']:.0f}" if "frac_female" in loio.columns else "--"
                f.write(f"{_tex(r['held_out'])} & {int(r['n_test'])} & {pf} & "
                        f"{r['sex_auc']:.3f}{ci} & {r['age_mae']:.1f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- shape vs intensity modality table ----------
    mod_path = os.path.join(rdir, "modality_comparison.csv")
    mod = pd.read_csv(mod_path) if os.path.exists(mod_path) else None
    with open(os.path.join(tdir, "modality.tex"), "w") as f:
        if mod is None:
            f.write("% run `cli modality` to generate this table\n")
        else:
            f.write("\\begin{tabular}{lrcc}\n\\toprule\n")
            f.write("Feature modality & dim & Sex AUC & Age MAE \\\\\n\\midrule\n")
            disp = {"shape": "Geometric descriptors (all)", "intensity": "Intensity (HU stats)",
                    "shape+intensity": "Shape $+$ intensity"}
            for _, r in mod.iterrows():
                f.write(f"{disp.get(r['modality'], _tex(r['modality']))} & {int(r['n_features'])} & "
                        f"{r['sex_auc']:.3f} & {r['age_mae']:.1f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------- comparison table (ablations / alternatives) ----------
    comp_path = os.path.join(rdir, "comparison.csv")
    comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    with open(os.path.join(tdir, "comparison.tex"), "w") as f:
        if comp is None:
            f.write("% run `cli compare` to generate this table\n")
        else:
            f.write("\\begin{tabular}{llcc}\n\\toprule\n")
            f.write("Method & Eval & Sex AUC & Age MAE \\\\\n\\midrule\n")
            hsd = "sex_auc_std" in comp.columns
            for _, r in comp.iterrows():
                sd = (f"\\,$\\pm$\\,{r['sex_auc_std']:.3f}"
                      if hsd and r["sex_auc_std"] == r["sex_auc_std"] else "")
                f.write(f"{_tex(r['method'])} & {_tex(r.get('eval',''))} & "
                        f"{r['sex_auc']:.3f}{sd} & {r['age_mae']:.1f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    def _crow(sub):
        if comp is None:
            return None
        m = comp[comp["method"].str.contains(sub, case=False, regex=False)]
        return m.iloc[0] if len(m) else None

    # ---------- macros ----------
    best_sex = attr.iloc[0]
    second_sex = attr.iloc[1] if len(attr) > 1 else best_sex
    best_age = attr.sort_values("age_r2", ascending=False).iloc[0]
    worst_sex = attr.sort_values("sex_auc").iloc[0]
    uq1 = uq[uq.k_organs == 1]["frac_unique"].iloc[0]
    uq2 = uq[uq.k_organs == 2]["frac_unique"].iloc[0]
    nfem = int((labels["y_sex"] == 1).sum()); nmal = int((labels["y_sex"] == 0).sum())
    macros = "".join([
        _macro("NSubjectsTotal", f"{len(labels)}"),
        _macro("NFemale", f"{nfem}"), _macro("NMale", f"{nmal}"),
        _macro("AgeMean", f"{labels['y_age'].mean():.0f}"),
        _macro("AgeMin", f"{labels['y_age'].min():.0f}"),
        _macro("AgeMax", f"{labels['y_age'].max():.0f}"),
        _macro("NAnalyzed", f"{fusion['sex']['n']}"),
        _macro("NOrgansKept", f"{len(attr)}"),
        _macro("NFeaturesFull", f"{int(svs.loc['full','n_features'])}"),
        _macro("NFeaturesSize", f"{int(svs.loc['size','n_features']) // len(attr)}"),
        _macro("NFeaturesShape", f"{int(svs.loc['shape','n_features']) // len(attr)}"),
        _macro("NFeaturesPerOrgan", f"{int(svs.loc['full','n_features']) // len(attr)}"),
        _macro("SexAUCfull", f"{svs.loc['full','sex_auc']:.3f}"),
        _macro("SexAUCsize", f"{svs.loc['size','sex_auc']:.3f}"),
        _macro("SexAUCshape", f"{svs.loc['shape','sex_auc']:.3f}"),
        _macro("AgeMAEfull", f"{svs.loc['full','age_mae']:.1f}"),
        _macro("AgeRtwoFull", f"{svs.loc['full','age_r2']:.2f}"),
        _macro("AgeMAEshape", f"{svs.loc['shape','age_mae']:.1f}"),
        _macro("BestSexOrgan", _disp(best_sex['organ'])),
        _macro("BestSexOrganAUC", f"{best_sex['sex_auc']:.3f}"),
        _macro("SecondSexOrgan", _disp(second_sex['organ'])),
        _macro("SecondSexOrganAUC", f"{second_sex['sex_auc']:.3f}"),
        _macro("ThirdSexOrgan", _disp(attr.iloc[2]['organ']) if len(attr) > 2 else "n/a"),
        _macro("ThirdSexOrganAUC", f"{attr.iloc[2]['sex_auc']:.3f}" if len(attr) > 2 else "n/a"),
        _macro("BestAgeOrgan", _disp(best_age['organ'])),
        _macro("BestAgeOrganRtwo", f"{best_age['age_r2']:.2f}"),
        _macro("WorstSexOrgan", _disp(worst_sex['organ'])),
        _macro("WorstSexOrganAUC", f"{worst_sex['sex_auc']:.3f}"),
        _macro("UniqOneOrgan", f"{100*uq1:.0f}"),
        _macro("UniqTwoOrgan", f"{100*uq2:.0f}"),
        _macro("PathologyAUC", f"{fusion['pathology']['auc']:.3f}"),
    ])

    # --- LOIO / modality / CI / uniqueness-sensitivity macros (all guarded -> n/a) ---
    def _opt(fn):
        try:
            return fn()
        except Exception:
            return "n/a"
    macros += _macro("PathologyDeepAUC", _opt(lambda: f"{json.load(open(os.path.join(rdir, 'deep_results.json')))['metrics']['pathology']['auc']:.3f}"))
    macros += _macro("LOIOsiteIn", _opt(lambda: f"{int(loio[loio.held_out=='I']['n_test'].iloc[0])}"))
    macros += _macro("LOIOsexHeldI", _opt(lambda: f"{loio[loio.held_out=='I']['sex_auc'].iloc[0]:.3f}"))
    macros += _macro("LOIOsexMin", _opt(lambda: f"{loio['sex_auc'].min():.3f}"))
    macros += _macro("LOIOsexMax", _opt(lambda: f"{loio['sex_auc'].max():.3f}"))
    macros += _macro("LOIOnInst", _opt(lambda: f"{len(loio)}"))
    macros += _macro("ShapeSexAUC", _opt(lambda: f"{mod.set_index('modality').loc['shape','sex_auc']:.3f}"))
    macros += _macro("IntensitySexAUC", _opt(lambda: f"{mod.set_index('modality').loc['intensity','sex_auc']:.3f}"))
    macros += _macro("BothSexAUC", _opt(lambda: f"{mod.set_index('modality').loc['shape+intensity','sex_auc']:.3f}"))
    macros += _macro("ShapeAgeMAE", _opt(lambda: f"{mod.set_index('modality').loc['shape','age_mae']:.1f}"))
    macros += _macro("BothAgeMAE", _opt(lambda: f"{mod.set_index('modality').loc['shape+intensity','age_mae']:.1f}"))
    _hl = os.path.join(rdir, "headline_stats.json")
    if os.path.exists(_hl):
        h = json.load(open(_hl))
        # bootstrap point estimate (paired consistently with its CI)
        macros += _macro("SexAUCboot", f"{h['sex_auc']:.3f}")
        macros += _macro("AgeMAEboot", f"{h['age_mae']:.1f}")
        macros += _macro("SexAUClo", f"{h['sex_auc_lo']:.3f}") + _macro("SexAUChi", f"{h['sex_auc_hi']:.3f}")
        macros += _macro("AgeMAElo", f"{h['age_mae_lo']:.1f}") + _macro("AgeMAEhi", f"{h['age_mae_hi']:.1f}")
    else:
        for nm in ("SexAUCboot", "AgeMAEboot", "SexAUClo", "SexAUChi", "AgeMAElo", "AgeMAEhi"):
            macros += _macro(nm, "n/a")
    # auto-labeling macros (selective prediction + calibration)
    _lab = os.path.join(rdir, "labeling_results.json")
    if os.path.exists(_lab):
        L = json.load(open(_lab))
        macros += _macro("LabelAccFull", f"{100*L['sex_acc_full']:.0f}")
        macros += _macro("LabelAccHalf", f"{100*L['sex_acc_at_50cov']:.0f}")
        macros += _macro("LabelAccQuartile", f"{100*L.get('sex_acc_at_25cov', 0):.1f}")
        macros += _macro("LabelCovNinetyFive", f"{100*L['cov_at_95']:.0f}")
        macros += _macro("LabelCovNinetyNine", f"{100*L['cov_at_99']:.0f}")
        macros += _macro("SexECE", f"{L['sex_ece']:.3f}")
    else:
        for nm in ("LabelAccFull", "LabelAccHalf", "LabelCovNinetyFive", "LabelCovNinetyNine", "SexECE"):
            macros += _macro(nm, "n/a")
    _sel = os.path.join(rdir, "selective_sex.csv")
    if os.path.exists(_sel):
        sdf = pd.read_csv(_sel)
        with open(os.path.join(tdir, "selective.tex"), "w") as f:
            f.write("\\begin{tabular}{lrc}\n\\toprule\nCoverage & $n$ & Sex accuracy \\\\\n\\midrule\n")
            for _, r in sdf.iterrows():
                f.write(f"{100*r['coverage']:.0f}\\% & {int(r['n'])} & {100*r['accuracy']:.1f}\\% \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    _lcs = os.path.join(rdir, "learning_curve_sex.csv")
    if os.path.exists(_lcs):
        lcs = pd.read_csv(_lcs)
        macros += _macro("LCsexNsmall", f"{int(lcs.iloc[0]['n_train'])}")
        macros += _macro("LCsexAUCsmall", f"{lcs.iloc[0]['metric_mean']:.3f}")
    _uqs = os.path.join(rdir, "uniqueness_sensitivity.csv")
    if os.path.exists(_uqs):
        s = pd.read_csv(_uqs).set_index("n_bins")["frac_unique_1organ"]
        macros += _macro("UniqBinsThree", _opt(lambda: f"{100*s.loc[3]:.0f}"))
        macros += _macro("UniqBinsTwenty", _opt(lambda: f"{100*s.loc[20]:.0f}"))
    else:
        macros += _macro("UniqBinsThree", "n/a") + _macro("UniqBinsTwenty", "n/a")
    _uqc = os.path.join(rdir, "uniqueness_curve.csv")
    if os.path.exists(_uqc):
        uc = pd.read_csv(_uqc).dropna()
        macros += _macro("UniqNfirst", f"{int(uc.iloc[0]['n_subjects'])}")
        macros += _macro("UniqNall", f"{int(uc.iloc[-1]['n_subjects'])}")
    macros += _macro("NPerOrganMin", f"{int(attr['n'].min())}")
    macros += _macro("NPerOrganMax", f"{int(attr['n'].max())}")

    # comparison-derived macros (n/a until `cli compare` is run)
    def _cm(name, row, key, fmt):
        ok = row is not None and key in row and row[key] == row[key]
        return _macro(name, fmt.format(row[key]) if ok else "n/a")
    ssm_r, dgcnn_r, vol_r = _crow("SSM-PCA"), _crow("DGCNN"), _crow("volumes only")
    macros += "".join([
        _cm("SSMSexAUC", ssm_r, "sex_auc", "{:.3f}"), _cm("SSMAgeMAE", ssm_r, "age_mae", "{:.1f}"),
        _cm("DGCNNSexAUC", dgcnn_r, "sex_auc", "{:.3f}"), _cm("DGCNNAgeMAE", dgcnn_r, "age_mae", "{:.1f}"),
        _cm("VolumeSexAUC", vol_r, "sex_auc", "{:.3f}"),
    ])
    # deep-model macros (default to n/a so the paper compiles before `cli deep`)
    deep_path = os.path.join(rdir, "deep_results.json")
    def _f(d, k, fmt):
        v = d.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) and v == v else "n/a"
    if os.path.exists(deep_path):
        dr = json.load(open(deep_path)); m = dr["metrics"]
        macros += "".join([
            _macro("DeepSexAUC", _f(m.get("sex", {}), "auc", "{:.3f}")),
            _macro("DeepSexAUCstd", _f(m.get("sex", {}), "auc_std", "{:.3f}")),
            _macro("DeepSexBalAcc", _f(m.get("sex", {}), "balanced_acc", "{:.3f}")),
            _macro("DeepAgeMAE", _f(m.get("age", {}), "mae", "{:.1f}")),
            _macro("DeepAgeRtwo", _f(m.get("age", {}), "r2", "{:.2f}")),
            _macro("DeepNseeds", f"{len(dr.get('seeds', []))}"),
            _macro("DeepEvalSplit", str(dr.get("split_eval", "test"))),
            _macro("DeepNtest", _f(m.get("sex", {}), "n", "{:d}")),
        ])
    else:
        for nm in ("DeepSexAUC", "DeepSexAUCstd", "DeepSexBalAcc", "DeepAgeMAE",
                   "DeepAgeRtwo", "DeepNseeds", "DeepEvalSplit", "DeepNtest"):
            macros += _macro(nm, "n/a")

    with open(os.path.join(out_dir, "results_macros.tex"), "w") as f:
        f.write("% AUTO-GENERATED from experiments/ — do not edit by hand.\n")
        f.write(macros)
    print(f"[tables] wrote tables + results_macros.tex -> {out_dir}")
