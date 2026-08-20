"""Render the Materials-and-Methods pipeline overview figure (paper Fig. 1).

Composes six panels at true LNCS print size (12.2 cm wide) from real data:

  (a) coronal CT slice of one TotalSegmentator subject + the 16 study-organ
      masks (facing view, patient right on image left)
  (b) truncation control: a kept CT heart vs a discarded MRI heart, with
      their real boundary-voxel counts read from the shape cache
  (c) marching-cubes mesh -> 2048 surface points (right hip)
  (d) descriptor families (7 size, 8 scale-free per organ)
  (e) three compared representations -> models
  (f) the identical pipeline applied to an MRI subject

No result numbers appear anywhere; the only quantities shown are pipeline
constants (2048 points, 20-voxel tolerance, 7+8 descriptors, 16 organs,
240 features) and the two boundary-voxel counts, all read from data or
matching the released configuration.

Inputs:
  - CT volume + masks for subject s0004: fetched automatically into
    _workspace/figure_data/ via ranged HTTP (remotezip) if missing.
  - Cached point clouds: _workspace/features/totalseg/s0004/*.npz and
    _workspace/features/totalseg_mr/s0001/*.npz. Produce them with the
    extract phase (`shapedem extract`, both datasets) if absent.
  - Organ list: experiments/attribution.csv (never typed by hand).

Outputs:
  experiments/figures/pipeline.{pdf,png}          composed figure
  experiments/figures/pipeline_panels/*.{pdf,png} standalone panels

Run (from the repo root, inside the project environment):
  python scripts/make_pipeline_figure.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "_workspace"
DATA = WORKSPACE / "figure_data" / "totalseg_s0004"
FEAT_CT = WORKSPACE / "features" / "totalseg" / "s0004"
FEAT_MR = WORKSPACE / "features" / "totalseg_mr" / "s0001"
FIGDIR = REPO / "experiments" / "figures"
PANELDIR = FIGDIR / "pipeline_panels"

# same archive the extraction pipeline uses (configs/default.yaml)
ZIP_URL = "https://zenodo.org/api/records/8367088/files/Totalsegmentator_dataset_v2.zip/content"

ORGANS = list(pd.read_csv(REPO / "experiments" / "attribution.csv")["organ"])

# identity palette: bilateral organs share a hue, bones share a family;
# overlays add white contour gaps so identity never rests on color alone
ORGAN_COLORS = {
    "heart": "#8E1F4D", "aorta": "#E25A5A", "esophagus": "#C9A616",
    "inferior_vena_cava": "#2A4FB5", "vertebrae_L3": "#7C89D9",
    "sacrum": "#7C89D9", "kidney_left": "#14A085", "kidney_right": "#14A085",
    "liver": "#A65B2A", "gallbladder": "#3F9E4A", "stomach": "#E77FB0",
    "pancreas": "#E8863C", "spleen": "#7B3FA0", "urinary_bladder": "#2FA8C4",
    "hip_left": "#4A46C4", "hip_right": "#4A46C4",
}

INK = "#000C20"
NEUTRAL = "#5F6680"
TEAL = "#0FA07F"
CORAL = "#E25A5A"
INDIGO = "#7C89D9"
HIP = ORGAN_COLORS["hip_right"]

CM = 1 / 2.54
FW = 12.2  # LNCS \linewidth in cm; fonts below are true print sizes
FS_BODY = 5.6
FS_LABEL = 6.6


# ------------------------------------------------------------------ data

def ensure_data():
    """Ranged-fetch the CT + 16 masks of s0004 (only the missing members)."""
    DATA.mkdir(parents=True, exist_ok=True)
    members = ["s0004/ct.nii.gz"] + [f"s0004/segmentations/{o}.nii.gz"
                                     for o in ORGANS]
    missing = [m for m in members if not (DATA / Path(m).name).exists()]
    if not missing:
        return
    import time

    from remotezip import RemoteZip

    print(f"  fetching {len(missing)} member(s) from Zenodo (ranged HTTP)")
    for attempt in range(1, 6):
        try:
            with RemoteZip(ZIP_URL, initial_buffer_size=64 * 1024 * 1024) as zf:
                for m in list(missing):
                    (DATA / Path(m).name).write_bytes(zf.read(m))
                    missing.remove(m)
                    print(f"    {Path(m).name}")
            return
        except Exception as e:  # noqa: BLE001 - network flakiness
            print(f"    attempt {attempt} failed: {e}")
            if attempt == 5:
                raise
            time.sleep(10 * attempt)


def require_cache():
    for f in (FEAT_CT / "hip_right.npz", FEAT_CT / "hip_left.npz",
              FEAT_CT / "heart.npz", FEAT_MR / "heart.npz"):
        if not f.exists():
            raise SystemExit(
                f"missing {f}\nRun the extract phase first (`shapedem "
                "extract` for the CT and MRI datasets) so the cached "
                "point clouds exist.")


def load_ras(name):
    """Load a NIfTI in canonical RAS+ orientation -> (array, zooms)."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(DATA / name))
    return np.asanyarray(img.dataobj), img.header.get_zooms()


# ------------------------------------------------------------- rendering

def ct_window(vol, level=40, width=400):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((vol - lo) / (hi - lo), 0, 1)


def coronal(vol, j):
    """Coronal display matrix: rows=S (origin='lower'), columns flipped so
    the patient's right appears on the image's left (facing view)."""
    return vol[:, j, :].T[:, ::-1]


def pick_slice(masks, min_vox=150):
    """Coronal index showing the most study organs (greedy, first pick)."""
    nj = next(iter(masks.values())).shape[1]
    vis = {j: sum(m[:, j, :].sum() >= min_vox for m in masks.values())
           for j in range(10, nj - 10, 3)}
    return max(vis, key=vis.get)


def draw_slice(ax, ct, masks, zooms, j, overlay=True):
    M = coronal(ct, j)
    ax.imshow(ct_window(M), cmap="gray", origin="lower",
              aspect=zooms[2] / zooms[0], interpolation="bilinear")
    if overlay:
        for organ, mask in masks.items():
            S = coronal(mask, j)
            if S.sum() < 40:
                continue
            ax.contourf(S, levels=[0.5, 1.5],
                        colors=[ORGAN_COLORS[organ]], alpha=0.55)
            ax.contour(S, levels=[0.5], colors="white", linewidths=1.6)
    ax.set_axis_off()
    ax.text(0.04, 0.02, "R", transform=ax.transAxes, color="white",
            fontsize=6.0, ha="left", va="bottom")
    ax.text(0.96, 0.02, "L", transform=ax.transAxes, color="white",
            fontsize=6.0, ha="right", va="bottom")


def shaded_rgba(points, normals, hexcolor, light=(-0.4, 0.85, 0.45)):
    """Per-point RGBA: color modulated by a fixed-light lambert term."""
    from matplotlib.colors import to_rgb

    L = np.asarray(light, dtype=float)
    L /= np.linalg.norm(L)
    lam = np.clip(normals @ L, 0, None)
    shade = (0.55 + 0.45 * lam)[:, None]
    rgb = np.tile(np.asarray(to_rgb(hexcolor)), (len(points), 1)) * shade
    out = np.ones((len(points), 4))
    out[:, :3] = rgb
    return out


def cloud_axes(ax, pts, zoom=1.28):
    """Orthographic, anatomically true aspect, camera at the front (+A)."""
    ax.set_proj_type("ortho")
    ax.view_init(elev=5, azim=90)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    ax.set_box_aspect(tuple(maxs - mins), zoom=zoom)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_axis_off()


def ras_triad(ax, pts):
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    span = float((maxs - mins).max())
    o = mins - 0.17 * (maxs - mins)
    L = 0.21 * span
    for vec, lab in (((L, 0, 0), "R"), ((0, L, 0), "A"), ((0, 0, L), "S")):
        ax.quiver(*o, *vec, color=NEUTRAL, linewidth=0.8,
                  arrow_length_ratio=0.18)
        ax.text(*(o + 1.28 * np.asarray(vec)), lab, color=NEUTRAL,
                fontsize=5.0, ha="center", va="center")


def p_mesh(ax, verts, faces):
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    color=HIP, edgecolor="none", shade=True)
    cloud_axes(ax, verts)
    ras_triad(ax, verts)


def p_points(ax, pts, nrm):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2,
               c=shaded_rgba(pts, nrm, HIP),
               depthshade=False, rasterized=True, linewidths=0)
    cloud_axes(ax, pts)
    ras_triad(ax, pts)


def small_cloud(ax, npz, color, zoom=1.5, size=0.9):
    ax.scatter(npz["points"][:, 0], npz["points"][:, 1], npz["points"][:, 2],
               s=size, c=shaded_rgba(npz["points"], npz["normals"], color),
               depthshade=False, rasterized=True, linewidths=0)
    cloud_axes(ax, npz["points"], zoom=zoom)


def p_trunc_pair(fig, sub, ct_heart, mr_heart):
    """Truncation control: real kept vs discarded organ, real voxel counts."""
    gs = sub.subgridspec(2, 2, height_ratios=[3.6, 1.0], hspace=0.02,
                         wspace=0.25)
    kept_b = int(ct_heart["boundary_voxels"])
    disc_b = int(mr_heart["boundary_voxels"])
    assert not bool(ct_heart["truncated"]) and bool(mr_heart["truncated"])
    ax_kept = fig.add_subplot(gs[0, 0], projection="3d")
    small_cloud(ax_kept, ct_heart, ORGAN_COLORS["heart"])
    ax_disc = fig.add_subplot(gs[0, 1], projection="3d")
    small_cloud(ax_disc, mr_heart, ORGAN_COLORS["heart"])
    caps = ((f"heart, CT\n{kept_b} boundary vox\n✓ kept", TEAL),
            (f"heart, MRI\n{disc_b} boundary vox\n✗ discarded", CORAL))
    for col, (txt, c) in enumerate(caps):
        ax = fig.add_subplot(gs[1, col])
        ax.set_axis_off()
        ax.text(0.5, 1.0, txt, ha="center", va="top", fontsize=5.2, color=c,
                transform=ax.transAxes, linespacing=1.35)
    return ax_kept, ax_disc


def _box(ax, x, y, w, h, title, lines, edge, fs=FS_BODY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                                boxstyle="round,pad=0.010,rounding_size=0.02",
                                facecolor="white", edgecolor=edge,
                                linewidth=0.9, clip_on=False))
    ty = y + h - 0.035
    ax.text(x + 0.035, ty, title, transform=ax.transAxes,
            fontsize=fs + 0.6, fontweight="bold", va="top", color=INK)
    if lines:
        ax.text(x + 0.035, ty - 0.105, "\n".join(lines),
                transform=ax.transAxes, fontsize=fs, va="top", color=INK,
                linespacing=1.4)


def p_descriptors(ax):
    """Descriptor families — names mirror shapedem/shapes/descriptors.py."""
    ax.set_axis_off()
    _box(ax, 0.0, 0.58, 1.0, 0.40, "size — 7 per organ",
         ["centroid size · volume · area",
          "bbox extents (3) · diagonal"], TEAL)
    _box(ax, 0.0, 0.12, 1.0, 0.40, "scale-free — 8 per organ",
         ["eig. ratios (4) · aspects (2)",
          "sphericity · normalized SA/V"], INDIGO)
    ax.text(0.5, 0.0, "16 organs × 15 = 240 features",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_BODY, color=NEUTRAL)


def p_models(ax):
    """Three compared representations -> targets (paper Sec. 3.4)."""
    ax.set_axis_off()
    rows = [
        ("descriptors (240)", "→ XGBoost · RF · linear", TEAL),
        ("SSM: ICP + PCA", "→ same classifiers", INDIGO),
        ("points (2048×3)", "→ PointNet + attn. pool", CORAL),
    ]
    for k, (head, tail, c) in enumerate(rows):
        _box(ax, 0.0, 0.75 - 0.315 * k, 1.0, 0.25, head, [tail], c)
    ax.text(0.5, 0.0, "targets: sex · age", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=FS_BODY + 0.4,
            fontweight="bold", color=INK)


def p_mri(ax, mr_clouds):
    for organ, (pts, nrm) in mr_clouds.items():
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.7,
                   c=shaded_rgba(pts, nrm, ORGAN_COLORS.get(organ, NEUTRAL)),
                   depthshade=False, rasterized=True, linewidths=0)
    cloud_axes(ax, np.concatenate([p for p, _ in mr_clouds.values()]),
               zoom=1.3)


# ------------------------------------------------------------- composition

def label_over(fig, axes, txt, y):
    axes = axes if isinstance(axes, (list, tuple)) else [axes]
    boxes = [a.get_position() for a in axes]
    x = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    half_w = 0.5 * len(txt) * 0.0105
    x = min(max(x, half_w + 0.005), 0.995 - half_w)
    fig.text(x, min(y, 0.995), txt, ha="center", va="top",
             fontsize=FS_LABEL, fontweight="bold", color=INK)


def arrow_between(fig, ax_l, ax_r):
    bl, br = ax_l.get_position(), ax_r.get_position()
    yy = (bl.y0 + bl.y1) / 2
    fig.patches.append(FancyArrowPatch((bl.x1 + 0.004, yy),
                                       (br.x0 - 0.004, yy),
                                       transform=fig.transFigure,
                                       mutation_scale=8, color=NEUTRAL,
                                       arrowstyle="-|>", linewidth=1.0))


def save(fig, name, out=FIGDIR, transparent=False):
    fig.savefig(out / f"{name}.pdf", transparent=transparent)
    fig.savefig(out / f"{name}.png", transparent=transparent, dpi=600)
    plt.close(fig)
    print(f"  {name}")


def compose(ctx):
    fig = plt.figure(figsize=(FW * CM, 7.8 * CM))
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(2, 1, left=0.004, right=0.996, bottom=0.045,
                          top=0.925, hspace=0.44, height_ratios=[1.22, 1.0])
    top = gs[0].subgridspec(1, 4, wspace=0.15,
                            width_ratios=[2.0, 3.1, 1.5, 1.5])
    a0 = fig.add_subplot(top[0, 0])
    draw_slice(a0, *ctx["slice_args"])
    ax_kept, ax_disc = p_trunc_pair(fig, top[0, 1],
                                    ctx["ct_heart"], ctx["mr_heart"])
    a2 = fig.add_subplot(top[0, 2], projection="3d")
    p_mesh(a2, *ctx["mesh"])
    a3 = fig.add_subplot(top[0, 3], projection="3d")
    p_points(a3, *ctx["pts"])
    bot = gs[1].subgridspec(1, 3, wspace=0.24, width_ratios=[3.3, 3.1, 1.9])
    b0 = fig.add_subplot(bot[0, 0])
    p_descriptors(b0)
    b1 = fig.add_subplot(bot[0, 1])
    p_models(b1)
    b2 = fig.add_subplot(bot[0, 2], projection="3d")
    p_mri(b2, ctx["mr_clouds"])

    y_top = 0.99
    y_bot = max(b.get_position().y1 for b in (b0, b1, b2)) + 0.045
    label_over(fig, a0, "(a) CT + 16 masks", y_top)
    label_over(fig, [ax_kept, ax_disc],
               "(b) truncation control (>20 vox → out)", y_top)
    label_over(fig, [a2, a3], "(c) mesh → 2048 points", y_top)
    label_over(fig, b0, "(d) per-organ descriptors", y_bot)
    label_over(fig, b1, "(e) representations → models", y_bot)
    label_over(fig, b2, "(f) same pipeline, MRI", y_bot)
    arrow_between(fig, a0, ax_kept)
    arrow_between(fig, ax_disc, a2)
    arrow_between(fig, a2, a3)
    save(fig, "pipeline")


def export_panels(ctx):
    jobs = [
        ("slice_ct",
         lambda ax: draw_slice(ax, *ctx["slice_args"], overlay=False),
         False, False, 5.0, 8.0),
        ("slice_labels", lambda ax: draw_slice(ax, *ctx["slice_args"]),
         False, False, 5.0, 8.0),
        ("hip_mesh", lambda ax: p_mesh(ax, *ctx["mesh"]), True, True,
         6.0, 6.0),
        ("hip_points", lambda ax: p_points(ax, *ctx["pts"]), True, True,
         6.0, 6.0),
        ("trunc_kept_ct_heart",
         lambda ax: small_cloud(ax, ctx["ct_heart"], ORGAN_COLORS["heart"]),
         True, True, 6.0, 6.0),
        ("trunc_discarded_mr_heart",
         lambda ax: small_cloud(ax, ctx["mr_heart"], ORGAN_COLORS["heart"]),
         True, True, 6.0, 6.0),
        ("descriptors", p_descriptors, False, True, 7.0, 4.5),
        ("models", p_models, False, True, 7.0, 4.5),
        ("mri_body", lambda ax: p_mri(ax, ctx["mr_clouds"]), True, True,
         5.0, 8.0),
    ]
    for name, draw, is3d, transparent, w, h in jobs:
        fig = plt.figure(figsize=(w * CM, h * CM))
        fig.set_layout_engine("none")
        ax = fig.add_subplot(111, projection="3d" if is3d else None)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95)
        draw(ax)
        save(fig, name, out=PANELDIR, transparent=transparent)


def main():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS_BODY,
        "text.color": INK, "figure.facecolor": "white",
        "savefig.dpi": 600,
    })
    FIGDIR.mkdir(parents=True, exist_ok=True)
    PANELDIR.mkdir(parents=True, exist_ok=True)
    print("pipeline figure ->", FIGDIR)
    require_cache()
    ensure_data()

    from skimage import measure

    ct, zooms = load_ras("ct.nii.gz")
    masks = {o: load_ras(f"{o}.nii.gz")[0] > 0 for o in ORGANS}
    j = pick_slice(masks)

    hip = np.asarray(masks["hip_right"], dtype=np.uint8)
    verts, faces, _, _ = measure.marching_cubes(hip, 0.5,
                                                spacing=tuple(zooms))
    d = np.load(FEAT_CT / "hip_right.npz")

    # orientation receipt: in RAS+ the right hip must lie at larger x than
    # the left; the facing view then shows it on the image's left
    xl = float(np.load(FEAT_CT / "hip_left.npz")["points"][:, 0].mean())
    xr = float(d["points"][:, 0].mean())
    assert xr > xl, "orientation check failed: +x is not patient-right"
    print(f"  receipt: hip_right x={xr:.0f} > hip_left x={xl:.0f} (RAS ok)")

    mr_clouds = {}
    for organ in ORGANS:
        f = FEAT_MR / f"{organ}.npz"
        if f.exists():
            z = np.load(f)
            mr_clouds[organ] = (z["points"], z["normals"])

    ctx = {
        "slice_args": (ct, masks, zooms, j),
        "mesh": (verts, faces),
        "pts": (d["points"], d["normals"]),
        "ct_heart": np.load(FEAT_CT / "heart.npz"),
        "mr_heart": np.load(FEAT_MR / "heart.npz"),
        "mr_clouds": mr_clouds,
    }
    compose(ctx)
    export_panels(ctx)
    print("done.")


if __name__ == "__main__":
    main()
