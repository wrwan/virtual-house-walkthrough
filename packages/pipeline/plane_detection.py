"""RANSAC-based plane detection and classification.

The detector iteratively fits planes to the point cloud and classifies each
detected plane as floor, ceiling, or wall based on its normal orientation.

Coplanar but spatially separated surfaces (e.g. two walls on opposite sides
of a hallway) are split into individual segments using DBSCAN clustering.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from packages.core.types import BBox, DetectedPlane, PlaneKind, Vec3

logger = logging.getLogger(__name__)

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
    """Compute a robust AABB using percentile trimming.

    The 2nd / 98th percentile avoids a single outlier point stretching
    the bounding box far beyond the real surface.
    """
    inlier_pts = points[mask]
    if len(inlier_pts) < 20:
        # Too few points for percentile — fall back to exact bounds
        mins = inlier_pts.min(axis=0)
        maxs = inlier_pts.max(axis=0)
    else:
        mins = np.percentile(inlier_pts, 2, axis=0)
        maxs = np.percentile(inlier_pts, 98, axis=0)
    return BBox(
        min=Vec3(x=float(mins[0]), y=float(mins[1]), z=float(mins[2])),
        max=Vec3(x=float(maxs[0]), y=float(maxs[1]), z=float(maxs[2])),
    )


def _cluster_inliers(
    points: np.ndarray,
    cluster_eps: float = 0.3,
    min_cluster_size: int = 30,
) -> list[np.ndarray]:
    """Split a set of inlier points into spatially connected clusters.

    Uses a simple DBSCAN-like approach via a KD-tree: starting from an
    unvisited point, flood-fill all neighbours within *cluster_eps*.
    Returns a list of boolean masks, one per cluster.

    This ensures that two wall segments on the same geometric plane but
    separated by a doorway or hallway become separate detections.
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

        # BFS / flood-fill from point i
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


# ── coplanar plane fusing ────────────────────────────────────────────

def _should_fuse(
    a: DetectedPlane,
    b: DetectedPlane,
    normal_threshold: float,
    offset_threshold: float,
    spatial_gap: float,
) -> bool:
    """Return True if planes *a* and *b* are nearly coplanar and spatially close."""
    na = np.array([a.normal.x, a.normal.y, a.normal.z])
    nb = np.array([b.normal.x, b.normal.y, b.normal.z])

    # Normals must be nearly parallel
    dot = np.dot(na, nb)
    if abs(dot) < normal_threshold:
        return False

    # Perpendicular distance between the two parallel planes.
    # When normals point in opposite directions the offset signs flip.
    if dot < 0:
        plane_dist = abs(a.offset + b.offset)
    else:
        plane_dist = abs(a.offset - b.offset)
    if plane_dist > offset_threshold:
        return False

    # Spatial proximity – bounding boxes must overlap or be within *spatial_gap*
    if a.bounds and b.bounds:
        gap_x = max(0, max(a.bounds.min.x, b.bounds.min.x) - min(a.bounds.max.x, b.bounds.max.x))
        gap_y = max(0, max(a.bounds.min.y, b.bounds.min.y) - min(a.bounds.max.y, b.bounds.max.y))
        gap_z = max(0, max(a.bounds.min.z, b.bounds.min.z) - min(a.bounds.max.z, b.bounds.max.z))
        if max(gap_x, gap_y, gap_z) > spatial_gap:
            return False

    return True


def _merge_planes(a: DetectedPlane, b: DetectedPlane) -> DetectedPlane:
    """Merge two coplanar planes, keeping the larger one's orientation."""
    primary, secondary = (a, b) if a.inlier_count >= b.inlier_count else (b, a)

    merged_bounds: BBox | None = None
    if primary.bounds and secondary.bounds:
        merged_bounds = BBox(
            min=Vec3(
                x=min(primary.bounds.min.x, secondary.bounds.min.x),
                y=min(primary.bounds.min.y, secondary.bounds.min.y),
                z=min(primary.bounds.min.z, secondary.bounds.min.z),
            ),
            max=Vec3(
                x=max(primary.bounds.max.x, secondary.bounds.max.x),
                y=max(primary.bounds.max.y, secondary.bounds.max.y),
                z=max(primary.bounds.max.z, secondary.bounds.max.z),
            ),
        )
    else:
        merged_bounds = primary.bounds or secondary.bounds

    return DetectedPlane(
        kind=primary.kind,
        normal=primary.normal,
        offset=primary.offset,
        inlier_count=primary.inlier_count + secondary.inlier_count,
        bounds=merged_bounds,
    )


def _fuse_coplanar_planes(
    planes: list[DetectedPlane],
    *,
    normal_threshold: float = 0.95,
    offset_threshold: float = 0.15,
    spatial_gap: float = 0.5,
) -> list[DetectedPlane]:
    """Merge nearly-coplanar, spatially-overlapping planes into single walls.

    Two planes are fused when:

    * their normals are nearly parallel (dot product > *normal_threshold*),
    * the perpendicular distance between them is < *offset_threshold* metres,
    * their bounding boxes overlap or are within *spatial_gap* metres.

    The pass repeats until no more merges occur (transitive closure).
    """
    if len(planes) <= 1:
        return planes

    fused = list(planes)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(fused):
            j = i + 1
            while j < len(fused):
                if _should_fuse(fused[i], fused[j], normal_threshold, offset_threshold, spatial_gap):
                    logger.debug(
                        "Fusing plane %d (%d pts) ← plane %d (%d pts)",
                        i, fused[i].inlier_count, j, fused[j].inlier_count,
                    )
                    fused[i] = _merge_planes(fused[i], fused[j])
                    fused.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    if len(planes) != len(fused):
        logger.info(
            "  \U0001f517 Fused %d wall segments down to %d", len(planes), len(fused),
        )
    return fused


# ── public API ───────────────────────────────────────────────────────

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
    original = points  # keep for bounds calculation
    planes: list[DetectedPlane] = []
    min_inliers = max(int(len(points) * min_inlier_ratio), 3)

    # We need to keep track of which original indices are still in play
    active_mask = np.ones(len(points), dtype=bool)

    for iteration in range(max_planes):
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

        # Map inlier_mask back to original indices
        active_indices = np.where(active_mask)[0]
        original_inlier_mask = np.zeros(len(original), dtype=bool)
        original_inlier_mask[active_indices[inlier_mask]] = True

        # Cluster the inlier points to split spatially separated segments
        inlier_points = original[original_inlier_mask]
        clusters = _cluster_inliers(
            inlier_points,
            cluster_eps=cluster_eps,
            min_cluster_size=min_cluster_size,
        )

        # Map each cluster back to original-space indices
        inlier_original_indices = np.where(original_inlier_mask)[0]

        for cluster_mask in clusters:
            cluster_original_mask = np.zeros(len(original), dtype=bool)
            cluster_original_mask[inlier_original_indices[cluster_mask]] = True
            bounds = _inlier_bounds(original, cluster_original_mask)

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

        # Remove ALL inliers from the working set (even small clusters)
        remaining = remaining[~inlier_mask]
        active_mask[active_indices[inlier_mask]] = False

    # DISABLED: Floor/ceiling relabelling — only walls for now
    # _relabel_horizontal_planes(planes)

    # Filter to walls only
    planes = [p for p in planes if p.kind == PlaneKind.WALL]

    # Fuse nearly-coplanar overlapping wall segments into single walls
    planes = _fuse_coplanar_planes(planes)

    return planes
