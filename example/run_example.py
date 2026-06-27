"""Offline working example for shape-demographics (no downloads, ~1 minute).

Synthesises organ masks whose *shape* depends on a hidden "sex" (anisotropy) and
"age" (size), runs the REAL pipeline (marching cubes -> point cloud ->
correspondence-free descriptors -> cross-validated classifier/regressor), and
verifies that sex and age are recovered. This exercises the same code path used
within the paper, end to end, with no data download.

Run:  
1. bash setup.sh # install dependencies (once per machine)
2. source .venv/bin/activate # activate virtualenv (once per shell)
3. python example/run_example.py
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd

from shapedem.config import repo_root
from shapedem.shapes.extract import points_from_mask
from shapedem.shapes.descriptors import shape_descriptors
from shapedem.train.baselines import cv_classification, cv_regression


# DATA SOURCE: This script does NOT load any external data.
# Everything is generated synthetically in-memory below.


def ellipsoid_mask(radii_vox, shape=(64, 64, 64)):
    """Creates a 3D binary mask (True/False array) of an ellipsoid.

    This is SYNTHETIC data generation — it builds a 64x64x64 voxel grid
    and marks every voxel that falls inside an ellipsoid defined by the
    given radii. Think of it as a fake organ segmentation mask.
    """
    # Create 3D coordinate grids (one value per axis, broadcast together)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    # Center of the volume
    c = np.array(shape) / 2.0
    rz, ry, rx = radii_vox
    # Standard ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
    # Returns True for voxels inside the ellipsoid, False outside
    return ((zz - c[0]) / rz) ** 2 + ((yy - c[1]) / ry) ** 2 + ((xx - c[2]) / rx) ** 2 <= 1.0


def main(n=160, seed=0):
    # n=160 synthetic subjects are created (no real patient data)
    rng = np.random.RandomState(seed)

    # Simulates anisotropic voxel spacing like you'd get from a CT scan
    # (1.5mm between slices, 1mm in-plane)
    spacing = np.array([1.5, 1.0, 1.0])
    # 4x4 affine matrix maps voxel indices -> world coordinates (mm)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    rows = []
    for i in range(n):
        # SYNTHETIC LABELS: sex alternates 0/1, age is random 20–80
        # These are the ground-truth values the pipeline will try to recover
        sex = i % 2                                # 0 = "male", 1 = "female"
        age = rng.uniform(20, 80)                  # random age in years
        row = {"subject": f"e{i:04d}", "y_sex": float(sex), "y_age": age}

        # Each subject gets TWO fake organs with slightly different base sizes
        for organ, base in (("organA", 14.0), ("organB", 11.0)):

            # KEY TRICK: the shape of the ellipsoid depends on sex and age,
            # so if the pipeline works, it should be able to recover those
            # variables from the shape features alone.

            # Sex effect: "female" organs are elongated along one axis (+35%)
            elong = 1.0 + (0.35 if sex == 1 else 0.0) + rng.normal(0, 0.05)
            # Age effect: organs grow/shrink linearly with age
            scale = 1.0 + (age - 50) / 200.0

            # Build 3 radii (in mm) with some random noise
            radii_mm = np.array([base * scale * (1 + rng.normal(0, 0.05)),
                                 base * scale,
                                 base * scale * elong])

            # Generate the 3D ellipsoid binary mask (fake organ segmentation)
            mask = ellipsoid_mask(radii_mm / spacing)

            # REAL PIPELINE STEP 1: extract a point cloud from the mask surface
            # (marching cubes to find the surface, then sample n_points on it)
            sh = points_from_mask(mask, spacing, affine, n_points=1024,
                                  min_voxels=50, boundary_tol=10 ** 9)

            # REAL PIPELINE STEP 2: compute shape descriptors from the point cloud
            # (correspondence-free geometric features like volume, surface area, etc.)
            for k, v in shape_descriptors(sh).items():
                row[f"{organ}__{k}"] = v
            row[f"{organ}__present"] = 1.0
        rows.append(row)

    # Assemble all subjects into a DataFrame
    # Each row = one subject, columns = shape features from both organs
    df = pd.DataFrame(rows).set_index("subject")

    # Select only the shape-descriptor columns (contain "__" but not "__present")
    feat_cols = [c for c in df.columns if "__" in c and not c.endswith("__present")]
    X = df[feat_cols]

    # REAL PIPELINE STEP 3: cross-validated classification (can we predict sex?)
    sex = cv_classification(X, df["y_sex"].to_numpy(), kind="logreg")
    # REAL PIPELINE STEP 4: cross-validated regression (can we predict age?)
    age = cv_regression(X, df["y_age"].to_numpy(), kind="ridge")

    # Print performance metrics
    print(f"Synthetic working example: {n} subjects, 2 organs, {len(feat_cols)} shape features")
    print(f"  sex : AUC {sex['auc']:.3f}  balanced-acc {sex['balanced_acc']:.3f}")
    print(f"  age : MAE {age['mae']:.2f} yr  R2 {age['r2']:.3f}")

    # Save results to disk
    out = os.path.join(repo_root(), "_workspace", "example_output")
    os.makedirs(out, exist_ok=True)
    json.dump({"sex": sex, "age": age}, open(os.path.join(out, "example_results.json"), "w"), indent=2)

    # Create a simple bar chart of the results
    import matplotlib
    matplotlib.use("Agg")                          # non-interactive backend (no GUI needed)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["sex AUC", "age $R^2$"], [sex["auc"], age["r2"]], color=["#3b7dd8", "#d8743b"])
    ax.set_ylim(0, 1.0); ax.set_title("Synthetic working example")
    fig.savefig(os.path.join(out, "example.png"), bbox_inches="tight", dpi=150); plt.close(fig)

    # Sanity checks: if the pipeline is working, it MUST recover
    # the embedded sex and age signals from the synthetic shapes
    assert sex["auc"] > 0.7, "sex signal not recovered — the pipeline is broken"
    assert age["r2"] > 0.3, "age signal not recovered — the pipeline is broken"
    print(f"OK — pipeline works. Outputs in {out}/")


if __name__ == "__main__":
    main()
