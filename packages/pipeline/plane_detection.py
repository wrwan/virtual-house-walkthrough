"""RANSAC-based plane detection and classification.

The detector iteratively fits planes to the point cloud and classifies each
detected plane as floor, ceiling, or wall based on its normal orientation.
"""

from __future__ import annotations

import numpy as np

from packages.core.types import BBox, DetectedPlane, PlaneKind, Vec3

# ── RANSAC single-plane fit ──────────────────────────────────────────

def _fit_plane_ransac(
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


# ── classification ───────────────────────────────────────────────────

def _classify_plane(
    normal: np.ndarray,
    offset: float,
    vertical_threshold: float = 0.8,
) -> PlaneKind:
    """Classify a plane by its normal direction.

    * If the normal is mostly vertical (|n_z| > threshold) → horizontal
      (floor or ceiling – disambiguated later by height).
    * Otherwise → wall.

    For horizontal planes, ``offset`` is the signed distance along Z.
    The caller should run :func:`_relabel_horizontal_planes` after all
    planes have been detected to split horizontal planes into floor vs
    ceiling using relative height.
    """
    nz = abs(normal[2])
    # DISABLED: Only detecting walls for now
    # if nz > vertical_threshold:
    #     return PlaneKind.FLOOR  # placeholder – relabelled later
    if nz > vertical_threshold:
        return PlaneKind.FLOOR  # still classified but filtered out later
    return PlaneKind.WALL


def _relabel_horizontal_planes(planes: list[DetectedPlane]) -> None:
    """Relabel horizontal planes: the lowest becomes *floor*, the highest *ceiling*.

    When there are more than two horizontal planes, the one closest to the
    global minimum Z is the floor and the one closest to the maximum Z is the
    ceiling; the rest stay as FLOOR (secondary floor slabs, etc.).
    """
    horizontal = [p for p in planes if p.kind == PlaneKind.FLOOR]
    if len(horizontal) < 2:
        return
    # Use bounds min-z as proxy for height of each horizontal plane
    horizontal.sort(key=lambda p: p.bounds.min.z if p.bounds else 0.0)
    # Lowest → floor (already labelled), highest → ceiling
    horizontal[-1].kind = PlaneKind.CEILING


def _inlier_bounds(points: np.ndarray, mask: np.ndarray) -> BBox:
    inlier_pts = points[mask]
    mins = inlier_pts.min(axis=0)
    maxs = inlier_pts.max(axis=0)
    return BBox(
        min=Vec3(x=float(mins[0]), y=float(mins[1]), z=float(mins[2])),
        max=Vec3(x=float(maxs[0]), y=float(maxs[1]), z=float(maxs[2])),
    )


# ── public API ───────────────────────────────────────────────────────

def detect_planes(
    points: np.ndarray,
    *,
    max_planes: int = 10,
    max_iterations: int = 1000,
    distance_threshold: float = 0.02,
    min_inlier_ratio: float = 0.02,
    seed: int | None = None,
) -> list[DetectedPlane]:
    """Iteratively detect up to *max_planes* planes from the cloud.

    After each plane is detected its inliers are removed and the next
    iteration runs on the remaining points.
    """
    rng = np.random.default_rng(seed)
    remaining = points.copy()
    original = points  # keep for bounds calculation
    planes: list[DetectedPlane] = []
    min_inliers = max(int(len(points) * min_inlier_ratio), 3)

    # We need to keep track of which original indices are still in play
    active_mask = np.ones(len(points), dtype=bool)

    for _ in range(max_planes):
        result = _fit_plane_ransac(
            remaining,
            max_iterations=max_iterations,
            distance_threshold=distance_threshold,
            min_inliers=min_inliers,
            rng=rng,
        )
        if result is None:
            break

        normal, offset, inlier_mask = result
        kind = _classify_plane(normal, offset)

        # Compute bounds against the original point set
        # Map inlier_mask back to original indices
        active_indices = np.where(active_mask)[0]
        original_inlier_mask = np.zeros(len(original), dtype=bool)
        original_inlier_mask[active_indices[inlier_mask]] = True
        bounds = _inlier_bounds(original, original_inlier_mask)

        planes.append(
            DetectedPlane(
                kind=kind,
                normal=Vec3(x=float(normal[0]), y=float(normal[1]), z=float(normal[2])),
                offset=float(offset),
                inlier_count=int(inlier_mask.sum()),
                bounds=bounds,
            )
        )

        # Remove inliers from the working set
        remaining = remaining[~inlier_mask]
        active_mask[active_indices[inlier_mask]] = False

    # DISABLED: Floor/ceiling relabelling — only walls for now
    # _relabel_horizontal_planes(planes)

    # Filter to walls only
    planes = [p for p in planes if p.kind == PlaneKind.WALL]
    return planes
