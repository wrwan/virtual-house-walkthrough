"""Plane classification, floor collection, clustering, and inlier bounds."""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from packages.core.types import BBox, DetectedPlane, PlaneKind, Vec3

logger = logging.getLogger(__name__)


def classify_plane(
    normal: np.ndarray,
    offset: float,
    vertical_threshold: float = 0.8,
) -> PlaneKind:
    """Classify a plane by its normal direction.

    * If the normal is mostly vertical (|n_z| > threshold) → horizontal
      (floor or ceiling – disambiguated later by height).
    * Otherwise → wall.
    """
    nz = abs(normal[2])
    if nz > vertical_threshold:
        return PlaneKind.FLOOR  # classified; filtered later by height
    return PlaneKind.WALL


def collect_floor_planes(
    planes: list[DetectedPlane],
    height_tolerance: float = 0.5,
) -> list[DetectedPlane]:
    """Return all horizontal planes that belong to the floor level.

    Keeps every horizontal plane whose height (bounds centre Z) is
    within *height_tolerance* of the lowest detected horizontal plane.
    This allows a non-rectangular floor to be represented by multiple
    spatially separate segments while discarding planes at clearly
    different heights (e.g. ceiling, mezzanine).
    """
    horizontal = [p for p in planes if p.kind == PlaneKind.FLOOR]
    if not horizontal:
        return []
    horizontal.sort(
        key=lambda p: ((p.bounds.min.z + p.bounds.max.z) / 2) if p.bounds else 0.0,
    )
    floor_z = (
        ((horizontal[0].bounds.min.z + horizontal[0].bounds.max.z) / 2)
        if horizontal[0].bounds
        else 0.0
    )
    return [
        p
        for p in horizontal
        if p.bounds
        and abs((p.bounds.min.z + p.bounds.max.z) / 2 - floor_z) <= height_tolerance
    ]


def inlier_bounds(points: np.ndarray, mask: np.ndarray) -> BBox:
    """Compute a robust AABB using percentile trimming.

    The 2nd / 98th percentile avoids a single outlier point stretching
    the bounding box far beyond the real surface.
    """
    inlier_pts = points[mask]
    if len(inlier_pts) < 20:
        mins = inlier_pts.min(axis=0)
        maxs = inlier_pts.max(axis=0)
    else:
        mins = np.percentile(inlier_pts, 2, axis=0)
        maxs = np.percentile(inlier_pts, 98, axis=0)
    return BBox(
        min=Vec3(x=float(mins[0]), y=float(mins[1]), z=float(mins[2])),
        max=Vec3(x=float(maxs[0]), y=float(maxs[1]), z=float(maxs[2])),
    )


def cluster_inliers(
    points: np.ndarray,
    cluster_eps: float = 0.3,
    min_cluster_size: int = 30,
) -> list[np.ndarray]:
    """Split a set of inlier points into spatially connected clusters.

    Uses a simple DBSCAN-like approach via a KD-tree: starting from an
    unvisited point, flood-fill all neighbours within *cluster_eps*.
    Returns a list of boolean masks, one per cluster.
    """
    n = len(points)
    if n == 0:
        return []

    tree = cKDTree(points)
    visited = np.zeros(n, dtype=bool)
    clusters: list[np.ndarray] = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        members = []

        while queue:
            idx = queue.pop()
            members.append(idx)
            neighbours = tree.query_ball_point(points[idx], cluster_eps)
            for nb in neighbours:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        if len(members) >= min_cluster_size:
            mask = np.zeros(n, dtype=bool)
            mask[members] = True
            clusters.append(mask)

    logger.info(
        f"  🔗 Clustered {n:,} inliers into {len(clusters)} segment(s) "
        f"(eps={cluster_eps}, min_size={min_cluster_size})"
    )
    return clusters
