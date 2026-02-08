"""Build a ParametricModel from detected planes and point-cloud metadata."""

from __future__ import annotations

from packages.core.types import BBox, DetectedPlane, ParametricModel


def build_parametric_model(
    *,
    source_file: str,
    point_count: int,
    bounds: BBox,
    planes: list[DetectedPlane],
) -> ParametricModel:
    """Assemble pipeline outputs into a :class:`ParametricModel`."""
    return ParametricModel(
        source_file=source_file,
        point_count=point_count,
        bounds=bounds,
        planes=planes,
    )
