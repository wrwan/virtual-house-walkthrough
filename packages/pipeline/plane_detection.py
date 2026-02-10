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

        # ── 2. recompute offset so plane passes through the original
        #    plane position, not the AABB centre.  We keep the original
        #    offset for the thin-axis centring in step 3 and only update
        #    the offset for the long axes.
        cx = (b.min.x + b.max.x) / 2
        cy = (b.min.y + b.max.y) / 2
        cz = (b.min.z + b.max.z) / 2

        # Use original normal + offset to find where the *original* plane
        # sits along the thin axis.  We must use the original normal here
        # because the snapped normal may have flipped direction, which would
        # invert the sign of plane_pos.
        dims = np.array([b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z])
        thin_xy = int(np.argmin(dims[:2]))  # 0 or 1
        nn = np.array([new_nx, new_ny, new_nz])
        orig_n = np.array([plane.normal.x, plane.normal.y, plane.normal.z])
        center = np.array([cx, cy, cz])

        if abs(orig_n[thin_xy]) > 1e-9:
            other_sum = sum(orig_n[k] * center[k] for k in range(3) if k != thin_xy)
            plane_pos = (plane.offset - other_sum) / orig_n[thin_xy]
        else:
            plane_pos = center[thin_xy]

        # Recompute offset for the *snapped* normal, using plane_pos on
        # the thin axis and AABB centres on the other axes.
        center_for_offset = center.copy()
        center_for_offset[thin_xy] = plane_pos
        new_offset = float(np.dot(nn, center_for_offset))

        # ── 3. normalise thickness, centred on actual plane ───
        new_min = np.array([b.min.x, b.min.y, b.min.z])
        new_max = np.array([b.max.x, b.max.y, b.max.z])

        # Set thin axis to target_thickness, centred on the plane
        new_min[thin_xy] = plane_pos - target_thickness / 2
        new_max[thin_xy] = plane_pos + target_thickness / 2

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

    # ── 5. snap wall endpoints to meet at corners ─────────────
    result = _snap_corners(result, snap_distance=1.0)

    # ── 6. merge walls that overlap after normalization ───────
    result = _merge_overlapping_walls(result)

    logger.info("  ✅ Normalized %d walls", len(result))
    return result


def _wall_long_axis(bounds: BBox) -> int:
    """Return the axis index (0=X, 1=Y) of the wall's long (length) direction."""
    dx = bounds.max.x - bounds.min.x
    dy = bounds.max.y - bounds.min.y
    return 0 if dx >= dy else 1


def _wall_thin_axis(bounds: BBox) -> int:
    """Return the axis index (0=X, 1=Y) of the wall's thin (thickness) direction."""
    dx = bounds.max.x - bounds.min.x
    dy = bounds.max.y - bounds.min.y
    return 0 if dx < dy else 1


def _merge_overlapping_walls(
    planes: list[DetectedPlane],
    overlap_threshold: float = 0.7,
    containment_threshold: float = 0.8,
) -> list[DetectedPlane]:
    """Merge walls that substantially overlap after normalization.

    Two parallel walls are merged when:
    - They are roughly parallel (normals within ~15°).
    - Their thin-axis centres are close (within combined half-thicknesses).
    - The smaller wall's long-axis span is mostly contained within the larger's.

    Additionally, any wall whose XY footprint is mostly contained inside
    another wall's footprint is removed (catches perpendicular fragments /
    "pillar" artefacts sitting inside a thicker wall).

    The larger wall absorbs the smaller and its bounds expand to cover both.
    """
    if len(planes) < 2:
        return planes

    alive = list(range(len(planes)))  # indices of walls still present

    # Pre-compute per-wall data
    infos: list[dict | None] = []
    for p in planes:
        if p.bounds is None:
            infos.append(None)
            continue
        b = p.bounds
        bmin = np.array([b.min.x, b.min.y, b.min.z])
        bmax = np.array([b.max.x, b.max.y, b.max.z])
        dims = bmax - bmin
        thin_xy = int(np.argmin(dims[:2]))
        long_xy = 1 - thin_xy
        n2d = np.array([p.normal.x, p.normal.y])
        n2d_len = np.linalg.norm(n2d)
        if n2d_len > 1e-6:
            n2d = n2d / n2d_len
        infos.append({
            "bmin": bmin,
            "bmax": bmax,
            "dims": dims,
            "thin": thin_xy,
            "long": long_xy,
            "n2d": n2d,
            "length": dims[long_xy],
            "thickness": dims[thin_xy],
            "thin_center": (bmin[thin_xy] + bmax[thin_xy]) / 2.0,
        })

    # Sort alive by wall length descending so larger walls absorb smaller ones
    alive.sort(key=lambda i: -(infos[i]["length"] if infos[i] else 0))

    removed: set[int] = set()

    for idx_a, a in enumerate(alive):
        if a in removed:
            continue
        info_a = infos[a]
        if info_a is None:
            continue

        for idx_b in range(idx_a + 1, len(alive)):
            b_idx = alive[idx_b]
            if b_idx in removed:
                continue
            info_b = infos[b_idx]
            if info_b is None:
                continue

            # Check parallel: dot product of 2D normals should be ~1
            dot = abs(float(np.dot(info_a["n2d"], info_b["n2d"])))
            if dot < 0.95:  # ~18° tolerance
                continue

            # Must share the same thin axis orientation
            if info_a["thin"] != info_b["thin"]:
                continue

            thin = info_a["thin"]
            long_ax = info_a["long"]

            # Check thin-axis proximity: centres should be within combined half-thicknesses
            max_gap = (info_a["thickness"] + info_b["thickness"]) / 2.0
            thin_dist = abs(info_a["thin_center"] - info_b["thin_center"])
            if thin_dist > max_gap:
                continue

            # Check long-axis overlap: how much of b is inside a?
            a_min_l = info_a["bmin"][long_ax]
            a_max_l = info_a["bmax"][long_ax]
            b_min_l = info_b["bmin"][long_ax]
            b_max_l = info_b["bmax"][long_ax]

            overlap_min = max(a_min_l, b_min_l)
            overlap_max = min(a_max_l, b_max_l)
            overlap_len = max(0.0, overlap_max - overlap_min)

            b_length = info_b["length"]
            if b_length < 1e-6:
                removed.add(b_idx)
                continue

            overlap_ratio = overlap_len / b_length
            if overlap_ratio < overlap_threshold:
                continue

            # Merge: expand a to cover b's long-axis span, keep a's thin axis
            info_a["bmin"][long_ax] = min(a_min_l, b_min_l)
            info_a["bmax"][long_ax] = max(a_max_l, b_max_l)
            info_a["length"] = info_a["bmax"][long_ax] - info_a["bmin"][long_ax]

            # Accumulate inlier counts
            planes[a] = DetectedPlane(
                kind=planes[a].kind,
                normal=planes[a].normal,
                offset=planes[a].offset,
                inlier_count=planes[a].inlier_count + planes[b_idx].inlier_count,
                bounds=planes[a].bounds,  # will be rebuilt below
            )

            removed.add(b_idx)

    # --- Second pass: remove walls whose XY footprint is contained inside
    #     another wall's footprint (catches perpendicular pillars / fragments)
    for idx_a, a in enumerate(alive):
        if a in removed:
            continue
        ia = infos[a]
        if ia is None:
            continue

        for idx_b in range(idx_a + 1, len(alive)):
            b_idx = alive[idx_b]
            if b_idx in removed:
                continue
            ib = infos[b_idx]
            if ib is None:
                continue

            # XY overlap rectangle
            ox_min = max(ia["bmin"][0], ib["bmin"][0])
            ox_max = min(ia["bmax"][0], ib["bmax"][0])
            oy_min = max(ia["bmin"][1], ib["bmin"][1])
            oy_max = min(ia["bmax"][1], ib["bmax"][1])

            if ox_min >= ox_max or oy_min >= oy_max:
                continue  # no overlap at all

            overlap_area = (ox_max - ox_min) * (oy_max - oy_min)

            b_area = (ib["bmax"][0] - ib["bmin"][0]) * (ib["bmax"][1] - ib["bmin"][1])
            a_area = (ia["bmax"][0] - ia["bmin"][0]) * (ia["bmax"][1] - ia["bmin"][1])

            # If b (smaller by length sort) is inside a, remove b
            if b_area > 1e-9 and overlap_area / b_area >= containment_threshold:
                removed.add(b_idx)
                continue

            # If a is inside b (rare, since sorted by length), remove a
            if a_area > 1e-9 and overlap_area / a_area >= containment_threshold:
                removed.add(a)
                break

    # Rebuild output
    out: list[DetectedPlane] = []
    for i in alive:
        if i in removed:
            continue
        info = infos[i]
        if info is None:
            out.append(planes[i])
            continue
        bmin = info["bmin"]
        bmax = info["bmax"]
        new_bounds = BBox(
            min=Vec3(x=float(bmin[0]), y=float(bmin[1]), z=float(bmin[2])),
            max=Vec3(x=float(bmax[0]), y=float(bmax[1]), z=float(bmax[2])),
        )
        out.append(DetectedPlane(
            kind=planes[i].kind,
            normal=planes[i].normal,
            offset=planes[i].offset,
            inlier_count=planes[i].inlier_count,
            bounds=new_bounds,
        ))

    merged_count = len(planes) - len(out)
    if merged_count:
        logger.info("  🔀 Merged %d overlapping walls → %d remain", merged_count, len(out))

    return out


def _snap_corners(
    planes: list[DetectedPlane],
    snap_distance: float = 1.0,
) -> list[DetectedPlane]:
    """Extend or shrink wall endpoints so perpendicular walls meet at corners.

    For each wall, look at both endpoints along its long axis.  If a nearby
    wall is roughly perpendicular and its thin-axis centre is close to this
    endpoint, snap this wall's endpoint to the other wall's centre-plane
    along the thin axis.  This makes the walls meet cleanly.

    The algorithm also extends walls slightly so they overlap by half the
    other wall's thickness — producing the typical architectural corner
    where one wall butts into the face of the perpendicular wall.
    """
    if len(planes) < 2:
        return planes

    # Pre-compute per-wall info: long axis, thin axis, centres, extents
    infos = []
    for p in planes:
        if p.bounds is None:
            infos.append(None)
            continue
        b = p.bounds
        bmin = np.array([b.min.x, b.min.y, b.min.z])
        bmax = np.array([b.max.x, b.max.y, b.max.z])
        long_ax = _wall_long_axis(b)
        thin_ax = _wall_thin_axis(b)
        center = (bmin + bmax) / 2.0
        infos.append({
            "bmin": bmin.copy(),
            "bmax": bmax.copy(),
            "long": long_ax,
            "thin": thin_ax,
            "center": center,
            "thickness": bmax[thin_ax] - bmin[thin_ax],
        })

    # For each wall, try to snap each long-axis endpoint to a perpendicular wall
    for i, info_i in enumerate(infos):
        if info_i is None:
            continue
        long_i = info_i["long"]
        thin_i = info_i["thin"]

        for j, info_j in enumerate(infos):
            if j == i or info_j is None:
                continue

            long_j = info_j["long"]
            thin_j = info_j["thin"]

            # Only snap walls whose long axes are different (i.e. roughly perpendicular)
            if long_i == long_j:
                continue

            # Wall i's long axis == wall j's thin axis (they are perpendicular)
            # Check if j's centre along its thin axis is near i's endpoint
            # along i's long axis.
            j_center_on_i_long = info_j["center"][long_i]

            # Distance from j's thin-axis centre to i's min/max long-axis endpoint
            i_min_end = info_i["bmin"][long_i]
            i_max_end = info_i["bmax"][long_i]

            # Also check that the walls overlap in the OTHER axis (thin_i = long_j)
            # i.e. wall j's extent along long_j overlaps wall i's thin-axis centre
            i_thin_center = info_i["center"][thin_i]
            j_min_long = info_j["bmin"][long_j]
            j_max_long = info_j["bmax"][long_j]

            if not (j_min_long - snap_distance <= i_thin_center <= j_max_long + snap_distance):
                continue  # walls are not in the same neighbourhood

            half_j_thick = info_j["thickness"] / 2.0
            max_trim = snap_distance * 0.5  # allow trimming up to half the snap window

            # Snap i's min endpoint — only if extending or moderately trimming.
            # A large positive trim means the wall passes THROUGH the corner
            # and the endpoint is on the far side — don't snap that.
            if abs(i_min_end - j_center_on_i_long) <= snap_distance:
                snapped_val = j_center_on_i_long - half_j_thick
                trim = snapped_val - i_min_end  # positive = trimming, negative = extending
                if trim <= max_trim:
                    info_i["bmin"][long_i] = snapped_val

            # Snap i's max endpoint
            if abs(i_max_end - j_center_on_i_long) <= snap_distance:
                snapped_val = j_center_on_i_long + half_j_thick
                trim = i_max_end - snapped_val  # positive = trimming, negative = extending
                if trim <= max_trim:
                    info_i["bmax"][long_i] = snapped_val
                snapped_val = j_center_on_i_long + half_j_thick
                info_i["bmax"][long_i] = snapped_val

    # Also snap perpendicular walls' long-axis endpoints to each other's faces
    # e.g. wall j should extend to reach i's face too
    # (handled symmetrically because we iterate all i,j pairs above)

    # Rebuild planes with snapped bounds
    out: list[DetectedPlane] = []
    for i, plane in enumerate(planes):
        info = infos[i]
        if info is None:
            out.append(plane)
            continue

        bmin = info["bmin"]
        bmax = info["bmax"]

        # Safety: ensure min < max
        for k in range(3):
            if bmin[k] > bmax[k]:
                bmin[k], bmax[k] = bmax[k], bmin[k]

        new_bounds = BBox(
            min=Vec3(x=float(bmin[0]), y=float(bmin[1]), z=float(bmin[2])),
            max=Vec3(x=float(bmax[0]), y=float(bmax[1]), z=float(bmax[2])),
        )

        out.append(DetectedPlane(
            kind=plane.kind,
            normal=plane.normal,
            offset=plane.offset,
            inlier_count=plane.inlier_count,
            bounds=new_bounds,
        ))

    snapped_count = sum(
        1 for i, info in enumerate(infos)
        if info is not None and (
            not np.allclose(info["bmin"], [planes[i].bounds.min.x, planes[i].bounds.min.y, planes[i].bounds.min.z], atol=1e-6)
            or not np.allclose(info["bmax"], [planes[i].bounds.max.x, planes[i].bounds.max.y, planes[i].bounds.max.z], atol=1e-6)
        )
        if planes[i].bounds is not None
    )
    if snapped_count:
        logger.info("  🔗 Snapped corners on %d / %d walls", snapped_count, len(planes))

    return out
