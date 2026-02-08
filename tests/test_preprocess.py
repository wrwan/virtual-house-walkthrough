"""Tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np

from packages.pipeline.preprocess import compute_bounds, voxel_downsample


class TestComputeBounds:
    def test_simple(self):
        pts = np.array([[0, 0, 0], [1, 2, 3], [-1, -2, -3]], dtype=np.float64)
        bbox = compute_bounds(pts)
        assert bbox.min.x == -1.0
        assert bbox.max.z == 3.0


class TestVoxelDownsample:
    def test_reduces_point_count(self):
        rng = np.random.default_rng(0)
        pts = rng.random((10_000, 3))
        ds = voxel_downsample(pts, voxel_size=0.1)
        assert len(ds) < len(pts)
        assert ds.shape[1] == 3

    def test_single_voxel(self):
        pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]], dtype=np.float64)
        ds = voxel_downsample(pts, voxel_size=0.1)
        assert len(ds) == 1
        np.testing.assert_allclose(ds[0], [0.015, 0.015, 0.015])

    def test_invalid_voxel_size(self):
        import pytest

        with pytest.raises(ValueError):
            voxel_downsample(np.zeros((5, 3)), voxel_size=0)
