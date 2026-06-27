"""Extraction driver: dataset subjects -> per-organ surface point clouds (npz).

Streaming & resumable: for each subject we open the zip once, pull only the
needed organ masks to a temp dir, convert to point clouds, save tiny npz, and
delete the temp masks. Re-running skips organs whose npz already exists.
"""
from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path

from .data import totalseg
from .shapes.extract import mask_to_pointcloud, save_shape


def feature_path(cfg, dataset: str, subject: str, organ: str) -> str:
    # Deterministic layout: features_dir/dataset/subject/organ.npz — enables resumability
    return os.path.join(cfg["paths"]["features_dir"], dataset, subject, f"{organ}.npz")


def extract_subject(cfg, source, dataset: str, subject: str, organs: list[str],
                    overwrite: bool = False) -> dict[str, str]:
    """Returns {organ: status} where status in {ok, absent, exists, error}."""
    ex = cfg["extract"]
    # Partition organs into already-done vs. to-do for resumability
    todo = []
    status: dict[str, str] = {}
    for organ in organs:
        fp = feature_path(cfg, dataset, subject, organ)
        if os.path.exists(fp) and not overwrite:
            status[organ] = "exists"
        else:
            todo.append(organ)
    if not todo:
        return status

    # Temp dir under work_dir (not /tmp) so it lives on the same filesystem as output
    tmp = tempfile.mkdtemp(prefix=f"{subject}_", dir=cfg["paths"]["work_dir"])
    try:
        # Map organ names to zip archive member paths, then batch-extract in one pass
        members = {organ: totalseg.seg_member(cfg, subject, organ) for organ in todo}
        got = source.extract_members(list(members.values()), tmp)
        for organ in todo:
            member = members[organ]
            path = got.get(member)
            if path is None:
                status[organ] = "absent"
                continue
            try:
                shape = mask_to_pointcloud(
                    path, n_points=ex["n_points"], min_voxels=ex["min_voxels"],
                    coord_space=ex["coord_space"], seed=ex["seed"],
                    boundary_tol=ex.get("boundary_tol", 20),
                )
            except Exception as e:  # noqa: BLE001
                status[organ] = f"error:{type(e).__name__}"
                continue
            if shape is None:
                status[organ] = "absent"
            else:
                # Save even truncated shapes (for transparent reporting); the
                # truncated flag inside the npz excludes them at analysis time.
                save_shape(feature_path(cfg, dataset, subject, organ), shape)
                status[organ] = "truncated" if int(shape["truncated"]) else "ok"
    finally:
        # Always clean up extracted masks — they're large and no longer needed
        shutil.rmtree(tmp, ignore_errors=True)
    return status


def extract_one(config_path, dataset: str, subject: str, organs: list[str],
                overwrite: bool = False) -> dict[str, str]:
    """Process-pool worker: reload config, open a fresh local zip, extract one subject."""
    # Import inside function — each pool worker needs its own config + zip handle
    from .config import load_config
    cfg = load_config(config_path)
    src = totalseg.from_config(cfg, "local")
    with src:
        return extract_subject(cfg, src, dataset, subject, organs, overwrite=overwrite)
