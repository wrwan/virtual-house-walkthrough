"""Coplanar plane fusing — merge nearly-coplanar overlapping segments."""

from __future__ import annotations

import logging

import numpy as np

from packages.core.types import BBox, DetectedPlane, Vec3

logger = logging.getLogger(__name__)


def should_fuse(
    a: DetectedPlane,
    b: DetectedPlane,
    normal_threshold: float,
    offset_threshold: float,
    spatial_gap: float,
) -> bool:
    """Return True if planes *a* and *b* are nearly coplanar and spatially close."""
    na = np.array([a.normal.x, a.normal.y, a.normal.z])
    nb = np.array([b.normal.x, b.normal.y, b.normal.z])

    dot = np.dot(na, nb)
    if abs(dot) < normal_threshold:
        return False

    if dot < 0:
        plane_dist = abs(a.offset + b.offset)
    else:
        plane_dist = abs(a.offset - b.offset)
    if plane_dist > offset_threshold:
        return False

    if a.bounds and b.bounds:
        gap_x = max(0, max(a.bounds.min.x, b.bounds.min.x) - min(a.bounds.max.x, b.bounds.max.x))
        gap_y = max(0, max(a.bounds.min.y, b.bounds.min.y) - min(a.bounds.max.y, b.bounds.max.y))
        gap_z = max(0, max(a.bounds.min.z, b.bounds.min.z) - min(a.bounds.max.z, b.bounds.max.z))
        if max(gap_x, gap_y, gap_z) > spatial_gap:
            return False

    return True


def merge_planes(a: DetectedPlane, b: DetectedPlane) -> DetectedPlane:
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


def fuse_coplanar_planes(
    planes: list[DetectedPlane],
    *,
    normal_threshold: float = 0.95,
    offset_threshold: float = 0.15,
    spatial_gap: float = 0.5,
) -> list[DetectedPlane]:
    """Merge nearly-coplanar, spatially-overlapping planes into single walls.

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
                if should_fuse(fused[i], fused[j], normal_threshold, offset_threshold, spatial_gap):
                    logger.debug(
                        "Fusing plane %d (%d pts) ← plane %d (%d pts)",
                        i, fused[i].inlier_count, j, fused[j].inlier_count,
                    )
                    fused[i] = merge_planes(fused[i], fused[j])
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
