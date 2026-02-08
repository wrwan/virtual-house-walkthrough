"""Tests for point-cloud loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from packages.pipeline.loader import load_ply, load_point_cloud


def _write_ply(path: Path, points: np.ndarray) -> None:
    """Helper: write an (N, 3) array as a binary PLY file."""
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    structured = np.empty(len(points), dtype=dtype)
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    el = PlyElement.describe(structured, "vertex")
    PlyData([el], text=False).write(str(path))


class TestLoadPly:
    def test_round_trip(self, tmp_path: Path):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        ply_file = tmp_path / "test.ply"
        _write_ply(ply_file, pts)

        loaded = load_ply(ply_file)
        assert loaded.shape == (2, 3)
        np.testing.assert_allclose(loaded, pts, atol=1e-5)

    def test_load_point_cloud_dispatch(self, tmp_path: Path):
        pts = np.random.default_rng(0).random((10, 3)).astype(np.float32)
        ply_file = tmp_path / "cloud.ply"
        _write_ply(ply_file, pts)

        loaded = load_point_cloud(ply_file)
        assert loaded.shape == (10, 3)

    def test_unsupported_extension(self, tmp_path: Path):
        fake = tmp_path / "file.xyz"
        fake.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_point_cloud(fake)
