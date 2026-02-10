"""End-to-end pipeline: load a point cloud → produce a parametric model JSON."""

from __future__ import annotations

import logging
from pathlib import Path

from packages.core.types import ParametricModel
from packages.pipeline.loader import load_point_cloud
from packages.pipeline.parametric import build_parametric_model
from packages.pipeline.plane_detection import detect_planes, normalize_walls
from packages.pipeline.preprocess import compute_bounds, voxel_downsample

logger = logging.getLogger(__name__)


def process_scan(
    input_path: str | Path,
    *,
    voxel_size: float = 0.05,
    max_planes: int = 50,
    ransac_iterations: int = 1000,
    distance_threshold: float = 0.02,
    seed: int | None = None,
) -> ParametricModel:
    """Run the full processing pipeline on a single point-cloud file.

    1. Load the file.
    2. Down-sample with a voxel grid.
    3. Detect planes via RANSAC.
    4. Classify planes (floor / ceiling / wall).
    5. Assemble a :class:`ParametricModel`.
    """
    input_path = Path(input_path)
    logger.info("Loading %s …", input_path.name)
    cloud_data = load_point_cloud(input_path)
    raw_points = cloud_data["positions"]
    logger.info("Loaded %d points", len(raw_points))

    logger.info("Down-sampling (voxel_size=%.3f) …", voxel_size)
    points = voxel_downsample(raw_points, voxel_size=voxel_size)
    logger.info("Down-sampled to %d points", len(points))

    bounds = compute_bounds(points)

    logger.info("Detecting planes (max=%d) …", max_planes)
    planes = detect_planes(
        points,
        max_planes=max_planes,
        max_iterations=ransac_iterations,
        distance_threshold=distance_threshold,
        seed=seed,
    )
    logger.info("Detected %d planes", len(planes))

    logger.info("Normalizing walls …")
    planes = normalize_walls(planes)
    logger.info("Normalized to %d walls", len(planes))

    model = build_parametric_model(
        source_file=input_path.name,
        point_count=len(raw_points),
        bounds=bounds,
        planes=planes,
    )
    return model


def process_scan_to_json(
    input_path: str | Path,
    output_path: str | Path | None = None,
    **kwargs,
) -> str:
    """Run the pipeline and write the parametric model to a JSON file.

    Returns the JSON string.
    """
    model = process_scan(input_path, **kwargs)
    json_str = model.model_dump_json(indent=2)

    if output_path is None:
        output_path = Path(input_path).with_suffix(".parametric.json")
    Path(output_path).write_text(json_str)
    logger.info("Wrote parametric model → %s", output_path)
    return json_str
