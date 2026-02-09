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


# ── manual wall refinement ───────────────────────────────────────────

def refine_wall_from_corners(
    points: np.ndarray,
    corners: list[list[float]],
    *,
    search_radius: float = 0.3,
    distance_threshold: float = 0.05,
    ransac_iterations: int = 2000,
) -> DetectedPlane:
    """Fit a wall plane from user-picked corners against the actual point cloud.

    1. Build a plane from the user's corners (least-squares).
    2. Collect all cloud points within *search_radius* of the AABB defined
       by the corners.
    3. Among those candidates, run RANSAC seeded toward the user plane to
       find the best-fit wall surface.
    4. Return a :class:`DetectedPlane` snapped to the real geometry.

    This is the "click 4 corners" feature: the user roughly marks where
    a missing wall is, and the algorithm refines it against the scan.
    """
    corners_arr = np.array(corners, dtype=np.float64)  # (K, 3)

    if len(corners_arr) < 3:
        raise ValueError("Need at least 3 corner points to define a plane")

    # ── 1. derive initial plane from the corners ──────────────────────
    centroid = corners_arr.mean(axis=0)
    centered = corners_arr - centroid
    _, _, vt = np.linalg.svd(centered)
    init_normal = vt[-1]  # smallest singular value = normal direction
    init_normal /= np.linalg.norm(init_normal)
    init_offset = float(np.dot(init_normal, centroid))

    # ── 2. gather candidate points near the user region ──────────────
    # expand AABB by search_radius in every direction
    rgn_min = corners_arr.min(axis=0) - search_radius
    rgn_max = corners_arr.max(axis=0) + search_radius

    in_box = np.all((points >= rgn_min) & (points <= rgn_max), axis=1)
    candidates = points[in_box]

    if len(candidates) < 10:
        # Not enough nearby points — just use the user corners directly
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

    # ── 3. pre-filter: keep only points close to the initial plane ───
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

    # ── 4. RANSAC on candidates ──────────────────────────────────────
    result = _fit_plane_ransac(
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
        # fall back to initial plane
        logger.info("RANSAC found no better plane — using corner-derived plane")
        normal = init_normal
        offset = init_offset
        inlier_pts = candidates[np.abs(candidates @ init_normal - init_offset) < distance_threshold]
        if len(inlier_pts) < 3:
            inlier_pts = corners_arr

    # ── 5. compute bounds from inliers ───────────────────────────────
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


# ── wall intersection trimming ───────────────────────────────────────

def trim_wall_at_intersection(
    wall: DetectedPlane,
    clipper: DetectedPlane,
) -> DetectedPlane:
    """Clip *wall*'s bounding box so it doesn't protrude past *clipper*.

    The clipper defines a half-space via its plane equation (n · x = d).
    We figure out which side of the clipper holds the bulk of the wall,
    then trim the wall's AABB to that side.

    This handles the common case where two walls meet at a corner and one
    of them overshoots past the other.
    """
    if wall.bounds is None or clipper.bounds is None:
        return wall  # nothing to trim

    cn = np.array([clipper.normal.x, clipper.normal.y, clipper.normal.z])
    cd = clipper.offset
    cn_norm = np.linalg.norm(cn)
    if cn_norm < 1e-12:
        return wall
    cn = cn / cn_norm
    cd = cd / cn_norm

    # Wall AABB corners
    wb = wall.bounds
    wmin = np.array([wb.min.x, wb.min.y, wb.min.z])
    wmax = np.array([wb.max.x, wb.max.y, wb.max.z])
    wcenter = (wmin + wmax) / 2.0

    # Which side of the clipper plane is the wall centre on?
    center_side = np.dot(cn, wcenter) - cd  # positive = same side as normal

    # Find which axis of the wall AABB the clipper normal is most aligned with.
    # We'll clamp along that axis.
    abs_cn = np.abs(cn)
    clip_axis = int(np.argmax(abs_cn))

    # The intersection coordinate along clip_axis: solve cn · p = cd
    # for p[clip_axis], assuming p is on the wall centre for the other axes.
    if abs(cn[clip_axis]) < 1e-9:
        return wall  # clipper is parallel to this axis – no meaningful clip

    other_sum = sum(cn[j] * wcenter[j] for j in range(3) if j != clip_axis)
    intersection_val = (cd - other_sum) / cn[clip_axis]

    # Clamp the wall's AABB: keep the side that contains the centre
    new_min = list(wmin)
    new_max = list(wmax)

    if center_side >= 0:
        # centre is on the + side of clipper → clamp minimum
        new_min[clip_axis] = max(new_min[clip_axis], intersection_val)
    else:
        # centre is on the − side → clamp maximum
        new_max[clip_axis] = min(new_max[clip_axis], intersection_val)

    # Safety: don't let min exceed max
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


# ── wall normalization / snapping ────────────────────────────────────

_SNAP_ANGLES_DEG = [0, 45, 90, 135, 180]


def _snap_angle(angle_deg: float) -> float:
    """Snap an angle (degrees) to the nearest canonical angle."""
    best = min(_SNAP_ANGLES_DEG, key=lambda a: abs(angle_deg - a))
    return best


def normalize_walls(
    planes: list[DetectedPlane],
    *,
    thickness_percentile: float = 50.0,
    height_percentile: float = 75.0,
) -> list[DetectedPlane]:
    """Clean up wall geometry: snap angles, unify thickness, align heights.

    Steps:

    1. **Compute dominant directions** – find the principal wall normal and
       snap every wall normal to the nearest 45° increment relative to it.
    2. **Normalise thickness** – compute the median wall thickness (the
       thin AABB axis) and set all walls to that value, centred on their
       current plane.
    3. **Align heights** – use a representative wall height and apply it
       uniformly, snapping the base to the lowest detected base.
    """
    if len(planes) < 2:
        return planes

    # Gather normals projected onto XY (horizontal plane)
    normals_2d = []
    for p in planes:
        n = np.array([p.normal.x, p.normal.y, p.normal.z])
        n2d = n[:2]
        length = np.linalg.norm(n2d)
        if length > 1e-6:
            normals_2d.append(n2d / length)
        else:
            normals_2d.append(np.array([1.0, 0.0]))

    # Dominant direction: the normal with the most inliers
    dominant_idx = max(range(len(planes)), key=lambda i: planes[i].inlier_count)
    ref_angle = float(np.degrees(np.arctan2(normals_2d[dominant_idx][1], normals_2d[dominant_idx][0])))

    # --- collect per-wall metrics ---
    thicknesses = []
    heights = []
    bases = []
    for p in planes:
        if p.bounds is None:
            continue
        b = p.bounds
        dims = [b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z]
        # thickness = thinnest dimension, height = z-extent
        thicknesses.append(min(dims[0], dims[1]))  # thin axis in XY
        heights.append(dims[2])
        bases.append(b.min.z)

    target_thickness = float(np.percentile(thicknesses, thickness_percentile)) if thicknesses else 0.1
    target_height = float(np.percentile(heights, height_percentile)) if heights else 2.5
    common_base = float(np.percentile(bases, 25)) if bases else 0.0  # lower quartile = floor level

    # Ensure minimums
    target_thickness = max(target_thickness, 0.04)
    target_height = max(target_height, 0.5)

    logger.info(
        "Normalize: thickness=%.3fm, height=%.2fm, base=%.2fm, ref_angle=%.1f°",
        target_thickness, target_height, common_base, ref_angle,
    )

    result: list[DetectedPlane] = []
    for i, plane in enumerate(planes):
        if plane.bounds is None:
            result.append(plane)
            continue

        b = plane.bounds
        n = np.array([plane.normal.x, plane.normal.y, plane.normal.z])

        # ── 1. snap normal to nearest 45° ─────────────────────
        n2d = normals_2d[i]
        raw_angle = float(np.degrees(np.arctan2(n2d[1], n2d[0])))
        # relative to dominant, snap, then back to absolute
        relative = (raw_angle - ref_angle) % 180.0
        snapped_rel = _snap_angle(relative)
        snapped_abs = ref_angle + snapped_rel
        snapped_rad = np.radians(snapped_abs)
        new_nx = float(np.cos(snapped_rad))
        new_ny = float(np.sin(snapped_rad))
        new_nz = float(n[2])  # keep vertical component
        nnorm = np.sqrt(new_nx**2 + new_ny**2 + new_nz**2)
        if nnorm > 1e-9:
            new_nx /= nnorm; new_ny /= nnorm; new_nz /= nnorm

        new_normal = Vec3(x=new_nx, y=new_ny, z=new_nz)

        # ── 2. recompute offset so plane still passes through centre ──
        cx = (b.min.x + b.max.x) / 2
        cy = (b.min.y + b.max.y) / 2
        cz = (b.min.z + b.max.z) / 2
        new_offset = new_nx * cx + new_ny * cy + new_nz * cz

        # ── 3. normalise thickness ────────────────────────────
        dims = np.array([b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z])
        thin_xy = int(np.argmin(dims[:2]))  # 0 or 1
        center = np.array([cx, cy, cz])

        new_min = np.array([b.min.x, b.min.y, b.min.z])
        new_max = np.array([b.max.x, b.max.y, b.max.z])

        # Set thin axis to target_thickness, centred
        new_min[thin_xy] = center[thin_xy] - target_thickness / 2
        new_max[thin_xy] = center[thin_xy] + target_thickness / 2

        # ── 4. normalise height ───────────────────────────────
        new_min[2] = common_base
        new_max[2] = common_base + target_height

        new_bounds = BBox(
            min=Vec3(x=float(new_min[0]), y=float(new_min[1]), z=float(new_min[2])),
            max=Vec3(x=float(new_max[0]), y=float(new_max[1]), z=float(new_max[2])),
        )

        result.append(DetectedPlane(
            kind=plane.kind,
            normal=new_normal,
            offset=float(new_offset),
            inlier_count=plane.inlier_count,
            bounds=new_bounds,
        ))

    logger.info("  ✅ Normalized %d walls", len(result))
    return result
