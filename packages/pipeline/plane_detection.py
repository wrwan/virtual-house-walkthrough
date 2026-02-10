"""RANSAC-based plane detection and classification.

This module is a **re-export facade** -- the implementation has been split
into focused sub-modules for maintainability:

* :mod:packages.pipeline.ransac    -- RANSAC single-plane fitting
* :mod:packages.pipeline.classify  -- classification, clustering, bounds
* :mod:packages.pipeline.fuse      -- coplanar plane fusing
* :mod:packages.pipeline.normalize -- wall normalization & corner snapping
* :mod:packages.pipeline.detect    -- iterative detection, trimming, manual refinement

All public symbols are re-exported here so existing `from
packages.pipeline.plane_detection import ...` statements continue to work.
"""

from __future__ import annotations

# Re-export public API exactly as before
from packages.pipeline.detect import (  # noqa: F401
    detect_planes,
    refine_wall_from_corners,
    trim_wall_at_intersection,
)
from packages.pipeline.normalize import (  # noqa: F401
    merge_overlapping_walls,
    normalize_walls,
    wall_long_axis,
    wall_thin_axis,
)
