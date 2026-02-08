"""Tests for RANSAC plane detection."""

from __future__ import annotations

import numpy as np

from packages.core.types import PlaneKind
from packages.pipeline.plane_detection import detect_planes


class TestDetectPlanes:
    def test_finds_floor_and_ceiling(self, simple_room_points: np.ndarray):
        planes = detect_planes(
            simple_room_points,
            max_planes=10,
            distance_threshold=0.02,
            seed=42,
        )
        kinds = {p.kind for p in planes}
        assert PlaneKind.FLOOR in kinds, "Should detect a floor"
        assert PlaneKind.CEILING in kinds, "Should detect a ceiling"

    def test_finds_walls(self, simple_room_points: np.ndarray):
        planes = detect_planes(
            simple_room_points,
            max_planes=10,
            distance_threshold=0.02,
            seed=42,
        )
        wall_count = sum(1 for p in planes if p.kind == PlaneKind.WALL)
        assert wall_count >= 2, f"Expected at least 2 walls, got {wall_count}"

    def test_total_plane_count(self, simple_room_points: np.ndarray):
        planes = detect_planes(
            simple_room_points,
            max_planes=10,
            distance_threshold=0.02,
            seed=42,
        )
        # The synthetic room has 6 planes; we should find most of them
        assert len(planes) >= 4, f"Expected ≥4 planes, got {len(planes)}"

    def test_empty_cloud(self):
        planes = detect_planes(np.empty((0, 3)), seed=0)
        assert planes == []

    def test_inlier_counts_positive(self, simple_room_points: np.ndarray):
        planes = detect_planes(simple_room_points, seed=42)
        for p in planes:
            assert p.inlier_count > 0
