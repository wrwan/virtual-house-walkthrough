"""RANSAC single-plane fitting."""

from __future__ import annotations

import numpy as np


def fit_plane_ransac(
    points: np.ndarray,
    *,
    max_iterations: int = 1000,
    distance_threshold: float = 0.02,
    min_inliers: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Fit a single plane to *points* using RANSAC.

    Returns ``(normal, offset, inlier_mask)`` or *None* if no plane with
    enough inliers is found.
    """
    rng = rng or np.random.default_rng()
    n = len(points)
    if n < 3:
        return None

    best_inliers: np.ndarray | None = None
    best_count = 0
    best_normal = np.zeros(3)
    best_d = 0.0

    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        d = np.dot(normal, p0)

        distances = np.abs(points @ normal - d)
        inlier_mask = distances < distance_threshold
        count = int(inlier_mask.sum())

        if count > best_count:
            best_count = count
            best_inliers = inlier_mask
            best_normal = normal
            best_d = d

    if best_count < min_inliers or best_inliers is None:
        return None

    return best_normal, best_d, best_inliers
