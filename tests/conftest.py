"""Shared test fixtures – synthetic point clouds for a simple room."""

from __future__ import annotations

import numpy as np
import pytest


def _make_plane_points(
    normal: np.ndarray,
    offset: float,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    extent: float = 5.0,
    n: int = 500,
    noise: float = 0.005,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate *n* points on a plane with a bit of Gaussian noise."""
    rng = rng or np.random.default_rng(0)
    centre = normal * offset
    u = rng.uniform(-extent / 2, extent / 2, size=(n, 1))
    v = rng.uniform(-extent / 2, extent / 2, size=(n, 1))
    pts = centre + u * u_axis + v * v_axis
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


@pytest.fixture()
def simple_room_points() -> np.ndarray:
    """A synthetic box-room: floor, ceiling, and four walls.

    Room is roughly 5 m × 5 m × 3 m, centred on the origin.
    Z is up.  Floor at z=0, ceiling at z=3.
    """
    rng = np.random.default_rng(42)
    planes: list[np.ndarray] = []

    # Floor (normal +Z, offset 0)
    planes.append(
        _make_plane_points(
            np.array([0, 0, 1.0]), 0.0,
            np.array([1, 0, 0.0]), np.array([0, 1, 0.0]),
            extent=5.0, n=600, rng=rng,
        )
    )
    # Ceiling (normal -Z, offset -3)  → we store normal pointing *into* room
    planes.append(
        _make_plane_points(
            np.array([0, 0, -1.0]), -3.0,
            np.array([1, 0, 0.0]), np.array([0, 1, 0.0]),
            extent=5.0, n=600, rng=rng,
        )
    )
    # Wall -X
    planes.append(
        _make_plane_points(
            np.array([-1, 0, 0.0]), -2.5,
            np.array([0, 1, 0.0]), np.array([0, 0, 1.0]),
            extent=5.0, n=400, rng=rng,
        )
    )
    # Wall +X
    planes.append(
        _make_plane_points(
            np.array([1, 0, 0.0]), 2.5,
            np.array([0, 1, 0.0]), np.array([0, 0, 1.0]),
            extent=5.0, n=400, rng=rng,
        )
    )
    # Wall -Y
    planes.append(
        _make_plane_points(
            np.array([0, -1, 0.0]), -2.5,
            np.array([1, 0, 0.0]), np.array([0, 0, 1.0]),
            extent=5.0, n=400, rng=rng,
        )
    )
    # Wall +Y
    planes.append(
        _make_plane_points(
            np.array([0, 1, 0.0]), 2.5,
            np.array([1, 0, 0.0]), np.array([0, 0, 1.0]),
            extent=5.0, n=400, rng=rng,
        )
    )

    return np.vstack(planes)
