"""Preprocessing helpers: down-sampling, normal estimation, bounds."""

from __future__ import annotations

import numpy as np

from packages.core.types import BBox, Vec3


def compute_bounds(points: np.ndarray) -> BBox:
    """Return the axis-aligned bounding box of an (N, 3) point array."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return BBox(
        min=Vec3(x=float(mins[0]), y=float(mins[1]), z=float(mins[2])),
        max=Vec3(x=float(maxs[0]), y=float(maxs[1]), z=float(maxs[2])),
    )


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """Voxel-grid down-sampling.

    Each occupied voxel keeps the centroid of its points.
    Returns a new (M, 3) array where M ≤ N.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    # Quantise to voxel grid
    mins = points.min(axis=0)
    keys = ((points - mins) / voxel_size).astype(np.int64)

    # Use a dict to accumulate per-voxel sums and counts
    voxel_map: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
    for i in range(len(keys)):
        k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        if k in voxel_map:
            s, c = voxel_map[k]
            s += points[i]
            voxel_map[k] = (s, c + 1)
        else:
            voxel_map[k] = (points[i].copy(), 1)

    centroids = np.empty((len(voxel_map), 3), dtype=np.float64)
    for idx, (s, c) in enumerate(voxel_map.values()):
        centroids[idx] = s / c

    return centroids


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate surface normals using PCA on *k*-nearest neighbours.

    Returns an (N, 3) array of unit normals.  For large clouds this is the
    bottleneck – a future version can use a KD-tree from scipy.
    """
    from scipy.spatial import cKDTree  # lazy import

    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))
    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbours = points[idx[i]]
        cov = np.cov(neighbours, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue → normal direction
    # Normalise
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms
