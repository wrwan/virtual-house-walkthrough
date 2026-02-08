"""Tests for point-cloud loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pye57
import pytest
from plyfile import PlyData, PlyElement

from packages.pipeline.loader import load_e57, load_ply, load_point_cloud


def _write_ply(path: Path, points: np.ndarray) -> None:
    """Helper: write an (N, 3) array as a binary PLY file."""
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    structured = np.empty(len(points), dtype=dtype)
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    el = PlyElement.describe(structured, "vertex")
    PlyData([el], text=False).write(str(path))


def _write_e57(path: Path, points: np.ndarray) -> None:
    """Helper: write an (N, 3) array as an E57 file."""
    e57 = pye57.E57(str(path), mode="w")
    data = {
        "cartesianX": points[:, 0].astype(np.float64),
        "cartesianY": points[:, 1].astype(np.float64),
        "cartesianZ": points[:, 2].astype(np.float64),
    }
    e57.write_scan_raw(data)
    e57.close()


class TestLoadPly:
    def test_round_trip(self, tmp_path: Path):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        ply_file = tmp_path / "test.ply"
        _write_ply(ply_file, pts)

        result = load_ply(ply_file)
        assert result["positions"].shape == (2, 3)
        np.testing.assert_allclose(result["positions"], pts, atol=1e-5)
        assert result["colors"] is None  # no RGB in this test file

    def test_load_point_cloud_dispatch(self, tmp_path: Path):
        pts = np.random.default_rng(0).random((10, 3)).astype(np.float32)
        ply_file = tmp_path / "cloud.ply"
        _write_ply(ply_file, pts)

        result = load_point_cloud(ply_file)
        assert result["positions"].shape == (10, 3)


class TestLoadE57:
    def test_round_trip(self, tmp_path: Path):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        e57_file = tmp_path / "test.e57"
        _write_e57(e57_file, pts)

        result = load_e57(e57_file)
        assert result["positions"].shape == (2, 3)
        np.testing.assert_allclose(result["positions"], pts, atol=1e-10)
        assert result["colors"] is None  # no RGB in this test file

    def test_load_point_cloud_dispatch(self, tmp_path: Path):
        pts = np.random.default_rng(0).random((10, 3))
        e57_file = tmp_path / "cloud.e57"
        _write_e57(e57_file, pts)

        result = load_point_cloud(e57_file)
        assert result["positions"].shape == (10, 3)
        np.testing.assert_allclose(result["positions"], pts, atol=1e-10)


class TestUnsupportedFormat:
    def test_unsupported_extension(self, tmp_path: Path):
        fake = tmp_path / "file.xyz"
        fake.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_point_cloud(fake)
