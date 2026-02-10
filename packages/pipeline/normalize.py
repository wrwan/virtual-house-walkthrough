"""Wall normalization — angle snapping, thickness/height unification, corner snapping."""

from __future__ import annotations

import logging

import numpy as np

from packages.core.types import BBox, DetectedPlane, PlaneKind, Vec3

logger = logging.getLogger(__name__)

_SNAP_ANGLES_DEG = [0, 45, 90, 135, 180]


def _snap_angle(angle_deg: float) -> float:
    """Snap an angle (degrees) to the nearest canonical angle."""
    best = min(_SNAP_ANGLES_DEG, key=lambda a: abs(angle_deg - a))
    return best


def wall_long_axis(bounds: BBox) -> int:
    """Return the axis index (0=X, 1=Y) of the wall's long (length) direction."""
    dx = bounds.max.x - bounds.min.x
    dy = bounds.max.y - bounds.min.y
    return 0 if dx >= dy else 1


def wall_thin_axis(bounds: BBox) -> int:
    """Return the axis index (0=X, 1=Y) of the wall's thin (thickness) direction."""
    dx = bounds.max.x - bounds.min.x
    dy = bounds.max.y - bounds.min.y
    return 0 if dx < dy else 1


def normalize_walls(
    planes: list[DetectedPlane],
    *,
    thickness_percentile: float = 50.0,
    height_percentile: float = 75.0,
) -> list[DetectedPlane]:
    """Clean up wall geometry: snap angles, unify thickness, align heights.

    Only WALL planes are modified. FLOOR and CEILING planes are passed
    through unchanged.

    Steps:

    1. **Compute dominant directions** – find the principal wall normal and
       snap every wall normal to the nearest 45° increment relative to it.
    2. **Normalise thickness** – compute the median wall thickness (the
       thin AABB axis) and set all walls to that value, centred on their
       current plane.
    3. **Align heights** – use a representative wall height and apply it
       uniformly, snapping the base to the lowest detected base.
    """
    walls = [p for p in planes if p.kind == PlaneKind.WALL]
    non_walls = [p for p in planes if p.kind != PlaneKind.WALL]

    if len(walls) < 2:
        return walls + non_walls

    # Gather normals projected onto XY (horizontal plane)
    normals_2d = []
    for p in walls:
        n = np.array([p.normal.x, p.normal.y, p.normal.z])
        n2d = n[:2]
        length = np.linalg.norm(n2d)
        if length > 1e-6:
            normals_2d.append(n2d / length)
        else:
            normals_2d.append(np.array([1.0, 0.0]))

    # Dominant direction: the normal with the most inliers
    dominant_idx = max(range(len(walls)), key=lambda i: walls[i].inlier_count)
    ref_angle = float(np.degrees(np.arctan2(normals_2d[dominant_idx][1], normals_2d[dominant_idx][0])))

    # --- collect per-wall metrics ---
    thicknesses = []
    heights = []
    bases = []
    for p in walls:
        if p.bounds is None:
            continue
        b = p.bounds
        dims = [b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z]
        thicknesses.append(min(dims[0], dims[1]))
        heights.append(dims[2])
        bases.append(b.min.z)

    target_thickness = float(np.percentile(thicknesses, thickness_percentile)) if thicknesses else 0.1
    target_height = float(np.percentile(heights, height_percentile)) if heights else 2.5
    common_base = float(np.percentile(bases, 25)) if bases else 0.0

    target_thickness = max(target_thickness, 0.04)
    target_height = max(target_height, 0.5)

    logger.info(
        "Normalize: thickness=%.3fm, height=%.2fm, base=%.2fm, ref_angle=%.1f°",
        target_thickness, target_height, common_base, ref_angle,
    )

    result: list[DetectedPlane] = []
    for i, plane in enumerate(walls):
        if plane.bounds is None:
            result.append(plane)
            continue

        b = plane.bounds
        n = np.array([plane.normal.x, plane.normal.y, plane.normal.z])

        # ── 1. snap normal to nearest 45° ─────────────────────
        n2d = normals_2d[i]
        raw_angle = float(np.degrees(np.arctan2(n2d[1], n2d[0])))
        relative = (raw_angle - ref_angle) % 180.0
        snapped_rel = _snap_angle(relative)
        snapped_abs = ref_angle + snapped_rel
        snapped_rad = np.radians(snapped_abs)
        new_nx = float(np.cos(snapped_rad))
        new_ny = float(np.sin(snapped_rad))
        new_nz = float(n[2])
        nnorm = np.sqrt(new_nx**2 + new_ny**2 + new_nz**2)
        if nnorm > 1e-9:
            new_nx /= nnorm; new_ny /= nnorm; new_nz /= nnorm

        new_normal = Vec3(x=new_nx, y=new_ny, z=new_nz)

        # ── 2. recompute offset ───────────────────────────────
        cx = (b.min.x + b.max.x) / 2
        cy = (b.min.y + b.max.y) / 2
        cz = (b.min.z + b.max.z) / 2

        dims = np.array([b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z])
        thin_xy = int(np.argmin(dims[:2]))
        nn = np.array([new_nx, new_ny, new_nz])
        orig_n = np.array([plane.normal.x, plane.normal.y, plane.normal.z])
        center = np.array([cx, cy, cz])

        if abs(orig_n[thin_xy]) > 1e-9:
            other_sum = sum(orig_n[k] * center[k] for k in range(3) if k != thin_xy)
            plane_pos = (plane.offset - other_sum) / orig_n[thin_xy]
        else:
            plane_pos = center[thin_xy]

        center_for_offset = center.copy()
        center_for_offset[thin_xy] = plane_pos
        new_offset = float(np.dot(nn, center_for_offset))

        # ── 3. normalise thickness, centred on actual plane ───
        new_min = np.array([b.min.x, b.min.y, b.min.z])
        new_max = np.array([b.max.x, b.max.y, b.max.z])

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

    logger.info("  ✅ Normalized %d walls", len(result))
    return result + non_walls


def _snap_corners(
    planes: list[DetectedPlane],
    snap_distance: float = 1.0,
) -> list[DetectedPlane]:
    """Extend or shrink wall endpoints so perpendicular walls meet at corners."""
    if len(planes) < 2:
        return planes

    infos = []
    for p in planes:
        if p.bounds is None:
            infos.append(None)
            continue
        b = p.bounds
        bmin = np.array([b.min.x, b.min.y, b.min.z])
        bmax = np.array([b.max.x, b.max.y, b.max.z])
        long_ax = wall_long_axis(b)
        thin_ax = wall_thin_axis(b)
        center = (bmin + bmax) / 2.0
        infos.append({
            "bmin": bmin.copy(),
            "bmax": bmax.copy(),
            "long": long_ax,
            "thin": thin_ax,
            "center": center,
            "thickness": bmax[thin_ax] - bmin[thin_ax],
        })

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

            if long_i == long_j:
                continue

            j_center_on_i_long = info_j["center"][long_i]

            i_min_end = info_i["bmin"][long_i]
            i_max_end = info_i["bmax"][long_i]

            i_thin_center = info_i["center"][thin_i]
            j_min_long = info_j["bmin"][long_j]
            j_max_long = info_j["bmax"][long_j]

            if not (j_min_long - snap_distance <= i_thin_center <= j_max_long + snap_distance):
                continue

            half_j_thick = info_j["thickness"] / 2.0
            max_trim = snap_distance * 0.5

            if abs(i_min_end - j_center_on_i_long) <= snap_distance:
                snapped_val = j_center_on_i_long - half_j_thick
                trim = snapped_val - i_min_end
                if trim <= max_trim:
                    info_i["bmin"][long_i] = snapped_val

            if abs(i_max_end - j_center_on_i_long) <= snap_distance:
                snapped_val = j_center_on_i_long + half_j_thick
                trim = i_max_end - snapped_val
                if trim <= max_trim:
                    info_i["bmax"][long_i] = snapped_val
                snapped_val = j_center_on_i_long + half_j_thick
                info_i["bmax"][long_i] = snapped_val

    out: list[DetectedPlane] = []
    for i, plane in enumerate(planes):
        info = infos[i]
        if info is None:
            out.append(plane)
            continue

        bmin = info["bmin"]
        bmax = info["bmax"]

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


def merge_overlapping_walls(
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
    another wall's footprint is removed.
    """
    if len(planes) < 2:
        return planes

    alive = list(range(len(planes)))

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

            dot = abs(float(np.dot(info_a["n2d"], info_b["n2d"])))
            if dot < 0.95:
                continue

            if info_a["thin"] != info_b["thin"]:
                continue

            thin = info_a["thin"]
            long_ax = info_a["long"]

            max_gap = (info_a["thickness"] + info_b["thickness"]) / 2.0
            thin_dist = abs(info_a["thin_center"] - info_b["thin_center"])
            if thin_dist > max_gap:
                continue

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

            info_a["bmin"][long_ax] = min(a_min_l, b_min_l)
            info_a["bmax"][long_ax] = max(a_max_l, b_max_l)
            info_a["length"] = info_a["bmax"][long_ax] - info_a["bmin"][long_ax]

            planes[a] = DetectedPlane(
                kind=planes[a].kind,
                normal=planes[a].normal,
                offset=planes[a].offset,
                inlier_count=planes[a].inlier_count + planes[b_idx].inlier_count,
                bounds=planes[a].bounds,
            )

            removed.add(b_idx)

    # Second pass: remove walls whose XY footprint is contained inside another
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

            ox_min = max(ia["bmin"][0], ib["bmin"][0])
            ox_max = min(ia["bmax"][0], ib["bmax"][0])
            oy_min = max(ia["bmin"][1], ib["bmin"][1])
            oy_max = min(ia["bmax"][1], ib["bmax"][1])

            if ox_min >= ox_max or oy_min >= oy_max:
                continue

            overlap_area = (ox_max - ox_min) * (oy_max - oy_min)

            b_area = (ib["bmax"][0] - ib["bmin"][0]) * (ib["bmax"][1] - ib["bmin"][1])
            a_area = (ia["bmax"][0] - ia["bmin"][0]) * (ia["bmax"][1] - ia["bmin"][1])

            if b_area > 1e-9 and overlap_area / b_area >= containment_threshold:
                removed.add(b_idx)
                continue

            if a_area > 1e-9 and overlap_area / a_area >= containment_threshold:
                removed.add(a)
                break

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
