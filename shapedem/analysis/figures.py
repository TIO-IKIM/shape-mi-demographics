"""Generate figures from the result CSVs in experiments/.

Pure consumer of committed result files -> figures saved under
experiments/figures/.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_attribution(rdir, outdir):
    df = pd.read_csv(os.path.join(rdir, "attribution.csv"))
    df = df.sort_values("sex_auc", ascending=True)
    df["organ"] = df["organ"].str.replace("_", " ")
    fig, ax = plt.subplots(1, 2, figsize=(11, max(3, 0.35 * len(df))))
    ax[0].barh(df["organ"], df["sex_auc"], color="#3b7dd8")
    ax[0].axvline(0.5, ls="--", c="gray", lw=1); ax[0].set_xlabel("Sex AUC"); ax[0].set_xlim(0.4, 1.0)
    ax[0].set_title("Sex (per-organ, shape)")
    df2 = df.sort_values("age_mae", ascending=False)
    ax[1].barh(df2["organ"], df2["age_mae"], color="#d8743b")
    ax[1].set_xlabel("Age MAE (years)"); ax[1].set_title("Age (per-organ, shape)")
    fig.suptitle("Per-organ demographic signal from shape alone")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.45)
    _save(fig, outdir, "attribution")


def fig_size_vs_shape(rdir, outdir):
    df = pd.read_csv(os.path.join(rdir, "size_vs_shape.csv")).set_index("regime")
    order = [r for r in ("size", "shape", "full") if r in df.index]
    df = df.loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].bar(df.index, df["sex_auc"], color="#3b7dd8"); ax[0].set_ylim(0.5, 1.0)
    ax[0].set_ylabel("Sex AUC"); ax[0].set_title("Sex")
    ax[1].bar(df.index, df["age_mae"], color="#d8743b")
    ax[1].set_ylabel("Age MAE (years)"); ax[1].set_title("Age")
    fig.suptitle("Size vs shape: signal surviving scale removal")
    fig.tight_layout()
    _save(fig, outdir, "size_vs_shape")


def fig_uniqueness(rdir, outdir):
    df = pd.read_csv(os.path.join(rdir, "uniqueness_curve.csv")).dropna()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(df["k_organs"], 100 * df["frac_unique"], "-o", color="#444")
    ax.set_xlabel("# organs in shape signature"); ax.set_ylabel("% subjects uniquely identified")
    ax.set_ylim(0, 105); ax.set_title("Anatomical shape as a fingerprint")
    _save(fig, outdir, "uniqueness")


def fig_ssm_modes(cfg, outdir, organ="hip_right"):
    """Visualize the first SSM principal mode (mean +/- sigma) of one organ."""
    import numpy as np
    from ..shapes.ssm import organ_ssm
    from .core import subjects_with_features
    subs = subjects_with_features(cfg, "totalseg")
    ids, coeffs, mean, modes = organ_ssm(cfg, "totalseg", subs, organ, n_modes=10)
    if coeffs is None:
        print(f"[figures] skip ssm_modes ({organ}): too few instances"); return
    sigma = float(coeffs[:, 0].std())
    mode0 = modes[0].reshape(mean.shape[0], 3)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    for ax, a, title in zip(axes, [-2.5, 0.0, 2.5], [r"$-2.5\sigma$", "mean", r"$+2.5\sigma$"]):
        pts = mean + a * sigma * mode0
        ax.scatter(pts[:, 0], pts[:, 2], s=2, c="#3b7dd8")
        ax.set_aspect("equal"); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{organ.replace('_', ' ')}: first statistical-shape-model mode")
    _save(fig, outdir, "ssm_modes")


def fig_labeling(rdir, outdir):
    """Auto-labeling evidence: selective accuracy-vs-coverage + data-efficiency."""
    sel = pd.read_csv(os.path.join(rdir, "selective_sex.csv"))
    lc = pd.read_csv(os.path.join(rdir, "learning_curve_sex.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax[0].plot(100 * sel["coverage"], 100 * sel["accuracy"], "-o", color="#2a9d8f")
    ax[0].axhline(95, ls="--", c="gray", lw=1)
    ax[0].set_xlabel("coverage (% auto-labelled)"); ax[0].set_ylabel("sex accuracy (%)")
    ax[0].set_title("Selective auto-labelling"); ax[0].invert_xaxis()
    ax[1].errorbar(lc["n_train"], lc["metric_mean"], yerr=lc["metric_std"], marker="o", color="#e76f51")
    ax[1].set_xlabel("# labelled training subjects"); ax[1].set_ylabel("sex AUC")
    ax[1].set_title("Data efficiency")
    fig.tight_layout()
    _save(fig, outdir, "labeling")


def make_all(cfg):
    rdir = cfg["paths"]["results_dir"]
    outdir = os.path.join(rdir, "figures")
    made = []
    for fn in (fig_attribution, fig_size_vs_shape, fig_uniqueness, fig_labeling):
        try:
            fn(rdir, outdir); made.append(fn.__name__)
        except FileNotFoundError as e:
            print(f"[figures] skip {fn.__name__}: {e}")
    try:
        fig_ssm_modes(cfg, outdir); made.append("fig_ssm_modes")
    except Exception as e:  # noqa: BLE001
        print(f"[figures] skip fig_ssm_modes: {e}")
    print(f"[figures] wrote {made} -> {outdir}")
    return outdir
