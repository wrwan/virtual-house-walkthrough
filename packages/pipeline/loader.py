"""Load point-cloud files into a NumPy (N, 3) array.

Supported formats
-----------------
* **PLY** – via the ``plyfile`` library (always available).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_ply(path: str | Path) -> np.ndarray:
    """Read a binary or ASCII PLY file and return an (N, 3) float64 array."""
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xs = np.asarray(vertex["x"], dtype=np.float64)
    ys = np.asarray(vertex["y"], dtype=np.float64)
    zs = np.asarray(vertex["z"], dtype=np.float64)
    return np.column_stack((xs, ys, zs))


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Auto-detect format and return (N, 3) points.

    Raises ``ValueError`` for unsupported extensions.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".ply":
        return load_ply(p)
    raise ValueError(
        f"Unsupported point-cloud format '{ext}'. Supported: .ply"
    )
