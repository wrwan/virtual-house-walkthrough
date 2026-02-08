"""End-to-end test for the processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from packages.core.types import ParametricModel
from packages.pipeline.process import process_scan, process_scan_to_json


def _write_ply(path: Path, points: np.ndarray) -> None:
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    structured = np.empty(len(points), dtype=dtype)
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    el = PlyElement.describe(structured, "vertex")
    PlyData([el], text=False).write(str(path))


class TestProcessScan:
    def test_produces_valid_model(self, simple_room_points: np.ndarray, tmp_path: Path):
        ply_file = tmp_path / "room.ply"
        _write_ply(ply_file, simple_room_points)

        model = process_scan(ply_file, voxel_size=0.05, seed=42)

        assert isinstance(model, ParametricModel)
        assert model.version == "0.1.0"
        assert model.units == "metres"
        assert model.point_count == len(simple_room_points)
        assert len(model.planes) >= 4

    def test_json_output(self, simple_room_points: np.ndarray, tmp_path: Path):
        ply_file = tmp_path / "room.ply"
        _write_ply(ply_file, simple_room_points)
        out_json = tmp_path / "model.json"

        json_str = process_scan_to_json(ply_file, output_path=out_json, seed=42)

        assert out_json.exists()
        data = json.loads(json_str)
        assert "planes" in data
        assert "bounds" in data
        # Validate it round-trips through Pydantic
        model = ParametricModel.model_validate(data)
        assert model.source_file == "room.ply"
