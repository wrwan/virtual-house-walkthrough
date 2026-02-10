"""High-level plane detection, wall trimming, and manual wall refinement."""

from __future__ import annotations

import logging

import numpy as np

from packages.core.types import BBox, DetectedPlane, PlaneKind, Vec3
from packages.pipeline.classify import (
    classify_plane,
    cluster_inliers,
    collect_floor_planes,
    inlier_bounds,
)
from packages.pipeline.fuse import fuse_coplanar_planes
from packages.pipeline.ransac import fit_plane_ransac

logger = logging.getLogger(__name__)


# ── iterative detection ──────────────────────────────────────────────

def detect_planes(
    points: np.ndarray,
    *,
    max_planes: int = 50,
    max_iterations: int = 1000,
    distance_threshold: float = 0.02,
    min_inlier_ratio: float = 0.02,
    cluster_eps: float = 0.3,
    min_cluster_size: int = 30,
    seed: int | None = None,
) -> list[DetectedPlane]:
    """Iteratively detect up to *max_planes* planes from the cloud.

    After each plane is detected its inliers are removed and the next
    iteration runs on the remaining points.

    Coplanar but spatially disconnected surfaces are split into separate
    segments via spatial clustering, so a wall in one room won't merge
    with a coplanar wall across a hallway.
    """
    rng = np.random.default_rng(seed)
    remaining = points.copy()
    original = points
    planes: list[DetectedPlane] = []
    min_inliers = max(int(len(points) * min_inlier_ratio), 3)

    active_mask = np.ones(len(points), dtype=bool)

    for iteration in range(max_planes):
        result = fit_plane_ransac(
            remaining,
            max_iterations=max_iterations,
            distance_threshold=distance_threshold,
            min_inliers=min_inliers,
            rng=rng,
        )
        if result is None:
            break

        normal, offset, inlier_mask = result
        kind = classify_plane(normal, offset)

        active_indices = np.where(active_mask)[0]
        original_inlier_mask = np.zeros(len(original), dtype=bool)
        original_inlier_mask[active_indices[inlier_mask]] = True

        inlier_points = original[original_inlier_mask]
        clusters = cluster_inliers(
            inlier_points,
            cluster_eps=cluster_eps,
            min_cluster_size=min_cluster_size,
        )

        inlier_original_indices = np.where(original_inlier_mask)[0]

        for cluster_mask in clusters:
            cluster_original_mask = np.zeros(len(original), dtype=bool)
            cluster_original_mask[inlier_original_indices[cluster_mask]] = True
            bounds = inlier_bounds(original, cluster_original_mask)

            planes.append(
                DetectedPlane(
                    kind=kind,
                    normal=Vec3(x=float(normal[0]), y=float(normal[1]), z=float(normal[2])),
                    offset=float(offset),
                    inlier_count=int(cluster_mask.sum()),
                    bounds=bounds,
                )
            )

        logger.info(
            f"  Plane {iteration}: {kind.value} — {int(inlier_mask.sum()):,} inliers → "
            f"{len(clusters)} segment(s)"
        )

        remaining = remaining[~inlier_mask]
        active_mask[active_indices[inlier_mask]] = False

    # Keep only walls + floor-level horizontal planes
    walls = [p for p in planes if p.kind == PlaneKind.WALL]
    floors = collect_floor_planes(planes)

    # Fuse nearly-coplanar overlapping wall segments into single walls
    walls = fuse_coplanar_planes(walls)

    planes = walls + floors

    return planes


# ── wall intersection trimming ───────────────────────────────────────

def trim_wall_at_intersection(
    wall: DetectedPlane,
    clipper: DetectedPlane,
) -> DetectedPlane:
    """Clip *wall*'s bounding box so it doesn't protrude past *clipper*."""
    if wall.bounds is None or clipper.bounds is None:
        return wall

    cn = np.array([clipper.normal.x, clipper.normal.y, clipper.normal.z])
    cd = clipper.offset
    cn_norm = np.linalg.norm(cn)
    if cn_norm < 1e-12:
        return wall
    cn = cn / cn_norm
    cd = cd / cn_norm

    wb = wall.bounds
    wmin = np.array([wb.min.x, wb.min.y, wb.min.z])
    wmax = np.array([wb.max.x, wb.max.y, wb.max.z])
    wcenter = (wmin + wmax) / 2.0

    center_side = np.dot(cn, wcenter) - cd

    abs_cn = np.abs(cn)
    clip_axis = int(np.argmax(abs_cn))

    if abs(cn[clip_axis]) < 1e-9:
        return wall

    other_sum = sum(cn[j] * wcenter[j] for j in range(3) if j != clip_axis)
    intersection_val = (cd - other_sum) / cn[clip_axis]

    new_min = list(wmin)
    new_max = list(wmax)

    if center_side >= 0:
        new_min[clip_axis] = max(new_min[clip_axis], intersection_val)
    else:
        new_max[clip_axis] = min(new_max[clip_axis], intersection_val)

    for k in range(3):
        if new_min[k] > new_max[k]:
            new_min[k], new_max[k] = new_max[k], new_min[k]

    new_bounds = BBox(
        min=Vec3(x=float(new_min[0]), y=float(new_min[1]), z=float(new_min[2])),
        max=Vec3(x=float(new_max[0]), y=float(new_max[1]), z=float(new_max[2])),
    )

    return DetectedPlane(
        kind=wall.kind,
        normal=wall.normal,
        offset=wall.offset,
        inlier_count=wall.inlier_count,
        bounds=new_bounds,
    )


# ── manual wall refinement ───────────────────────────────────────────

def refine_wall_from_corners(
    points: np.ndarray,
    corners: list[list[float]],
    *,
    search_radius: float = 0.3,
    distance_threshold: float = 0.05,
    ransac_iterations: int = 2000,
) -> DetectedPlane:
    """Fit a wall plane from user-picked corners against the actual point cloud."""
    corners_arr = np.array(corners, dtype=np.float64)

    if len(corners_arr) < 3:
        raise ValueError("Need at least 3 corner points to define a plane")

    # 1. derive initial plane from corners
    centroid = corners_arr.mean(axis=0)
    centered = corners_arr - centroid
    _, _, vt = np.linalg.svd(centered)
    init_normal = vt[-1]
    init_normal /= np.linalg.norm(init_normal)
    init_offset = float(np.dot(init_normal, centroid))

    # 2. gather candidate points
    rgn_min = corners_arr.min(axis=0) - search_radius
    rgn_max = corners_arr.max(axis=0) + search_radius

    in_box = np.all((points >= rgn_min) & (points <= rgn_max), axis=1)
    candidates = points[in_box]

    if len(candidates) < 10:
        logger.warning("Only %d candidate points near corners — using raw corners", len(candidates))
        bounds = BBox(
            min=Vec3(x=float(corners_arr[:, 0].min()), y=float(corners_arr[:, 1].min()), z=float(corners_arr[:, 2].min())),
            max=Vec3(x=float(corners_arr[:, 0].max()), y=float(corners_arr[:, 1].max()), z=float(corners_arr[:, 2].max())),
        )
        return DetectedPlane(
            kind=PlaneKind.WALL,
            normal=Vec3(x=float(init_normal[0]), y=float(init_normal[1]), z=float(init_normal[2])),
            offset=init_offset,
            inlier_count=len(corners_arr),
            bounds=bounds,
        )

    logger.info(
        "Manual wall: %d candidate points in region (%.1f × %.1f × %.1f m)",
        len(candidates),
        *(rgn_max - rgn_min),
    )

    # 3. pre-filter: keep only points close to the initial plane
    dists_to_plane = np.abs(candidates @ init_normal - init_offset)
    near_plane = dists_to_plane < search_radius
    candidates = candidates[near_plane]

    if len(candidates) < 10:
        logger.warning("Too few points near plane — using raw corners")
        bounds = BBox(
            min=Vec3(x=float(corners_arr[:, 0].min()), y=float(corners_arr[:, 1].min()), z=float(corners_arr[:, 2].min())),
            max=Vec3(x=float(corners_arr[:, 0].max()), y=float(corners_arr[:, 1].max()), z=float(corners_arr[:, 2].max())),
        )
        return DetectedPlane(
            kind=PlaneKind.WALL,
            normal=Vec3(x=float(init_normal[0]), y=float(init_normal[1]), z=float(init_normal[2])),
            offset=init_offset,
            inlier_count=len(corners_arr),
            bounds=bounds,
        )

    # 4. RANSAC on candidates
    result = fit_plane_ransac(
        candidates,
        max_iterations=ransac_iterations,
        distance_threshold=distance_threshold,
        min_inliers=max(10, len(candidates) // 10),
    )

    if result is not None:
        normal, offset, inlier_mask = result
        inlier_pts = candidates[inlier_mask]
        logger.info("Manual wall refined: %d inliers from %d candidates", int(inlier_mask.sum()), len(candidates))
    else:
        logger.info("RANSAC found no better plane — using corner-derived plane")
        normal = init_normal
        offset = init_offset
        inlier_pts = candidates[np.abs(candidates @ init_normal - init_offset) < distance_threshold]
        if len(inlier_pts) < 3:
            inlier_pts = corners_arr

    # 5. compute bounds from inliers
    if len(inlier_pts) < 20:
        mins = inlier_pts.min(axis=0)
        maxs = inlier_pts.max(axis=0)
    else:
        mins = np.percentile(inlier_pts, 2, axis=0)
        maxs = np.percentile(inlier_pts, 98, axis=0)

    bounds = BBox(
        min=Vec3(x=float(mins[0]), y=float(mins[1]), z=float(mins[2])),
        max=Vec3(x=float(maxs[0]), y=float(maxs[1]), z=float(maxs[2])),
    )

    return DetectedPlane(
        kind=PlaneKind.WALL,
        normal=Vec3(x=float(normal[0]), y=float(normal[1]), z=float(normal[2])),
        offset=float(offset),
        inlier_count=len(inlier_pts),
        bounds=bounds,
    )
