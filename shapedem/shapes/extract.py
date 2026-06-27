"""Convert a binary organ mask (NIfTI) into a surface point cloud + basic geometry.

Pipeline: load mask -> marching cubes (voxel coords) -> map to world (RAS mm)
via the affine (so shapes are anatomically oriented & scaled consistently across
subjects) -> area-weighted surface sampling to a fixed number of points.

Returns a dict with points (N,3), normals (N,3), and scalar geometry
(volume_mm3, surface_area_mm2, n_voxels). Everything is float32 and tiny.
"""
from __future__ import annotations
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh


def load_mask(path: str):
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    # Binarize: any positive label counts (handles multi-valued segmentations)
    mask = arr > 0
    # Voxel dimensions in mm — needed for volume calculation and scaled-coord mode
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    return mask, zooms, img.affine


def boundary_voxels(mask: np.ndarray) -> int:
    """Count mask voxels lying on any of the 6 faces of the volume.

    A bounded organ fully inside the field of view touches no face; a non-zero
    count means the organ is clipped by the scan FOV (truncated) and its shape
    is not anatomically complete.
    """
    total = 0
    # Check each of the 6 boundary slabs (first & last slice along each axis)
    for ax in range(mask.ndim):
        for idx in (0, mask.shape[ax] - 1):
            sl = [slice(None)] * mask.ndim
            sl[ax] = idx
            total += int(mask[tuple(sl)].sum())
    return total


def mask_to_pointcloud(path: str, n_points: int = 2048, min_voxels: int = 500,
                       coord_space: str = "world", seed: int = 0,
                       boundary_tol: int = 20):
    mask, zooms, affine = load_mask(path)
    return points_from_mask(mask, zooms, affine, n_points=n_points, min_voxels=min_voxels,
                            coord_space=coord_space, seed=seed, boundary_tol=boundary_tol)


def labelmask_to_pointcloud(path: str, labels, n_points: int = 2048, min_voxels: int = 500,
                            coord_space: str = "world", seed: int = 0, boundary_tol: int = 20):
    """Like mask_to_pointcloud but selects a subset of integer labels from a
    multi-label segmentation (e.g. uterus = {1,2})."""
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    # Round then cast: handles float-stored label maps (e.g. interpolation artifacts)
    mask = np.isin(np.rint(arr).astype(np.int64), list(labels))
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    return points_from_mask(mask, zooms, img.affine, n_points=n_points, min_voxels=min_voxels,
                            coord_space=coord_space, seed=seed, boundary_tol=boundary_tol)


def points_from_mask(mask, zooms, affine, n_points=2048, min_voxels=500,
                     coord_space="world", seed=0, boundary_tol=20):
    n_vox = int(mask.sum())
    if n_vox < min_voxels:
        return None  # too few voxels — likely out-of-FOV or failed segmentation
    n_boundary = boundary_voxels(mask)
    # Flag organs clipped by the scan FOV; they are saved but excluded from analysis
    truncated = n_boundary > boundary_tol

    # Marching cubes at 0.5 isosurface extracts the binary mask boundary as a triangle mesh
    verts_vox, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    if coord_space == "world":
        # World (RAS mm) via the NIfTI affine — ensures cross-subject anatomical alignment
        verts = nib.affines.apply_affine(affine, verts_vox)
    else:
        # Scaled voxel coords — preserves orientation but applies voxel spacing
        verts = verts_vox * zooms[None, :]

    # process=False skips trimesh's automatic merge/repair to preserve exact geometry
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    rng = np.random.RandomState(seed)
    # Area-weighted sampling: larger triangles get proportionally more points
    pts, fidx = trimesh.sample.sample_surface(mesh, n_points, seed=rng.randint(2**31 - 1))
    normals = mesh.face_normals[fidx]

    # Volume from voxel count (exact) rather than mesh (watertight assumption needed)
    vox_vol = float(np.prod(zooms))
    return {
        "points": pts.astype(np.float32),
        "normals": normals.astype(np.float32),
        "volume_mm3": np.float32(n_vox * vox_vol),
        "surface_area_mm2": np.float32(mesh.area),
        "n_voxels": np.int64(n_vox),
        "spacing": zooms.astype(np.float32),
        "boundary_voxels": np.int64(n_boundary),
        "truncated": np.int8(1 if truncated else 0),
    }


def save_shape(path_npz: str, shape: dict):
    import os
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)
    np.savez_compressed(path_npz, **shape)


def load_shape(path_npz: str) -> dict:
    with np.load(path_npz, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}
