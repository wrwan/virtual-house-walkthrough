"""Load point-cloud files into a NumPy (N, 3) array.

Supported formats
-----------------
* **PLY** – via the ``plyfile`` library.
* **E57** – via the ``pye57`` library (ASTM E2807 standard, used by BLK360 etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pye57
from plyfile import PlyData


def load_ply(path: str | Path) -> np.ndarray:
    """Read a binary or ASCII PLY file and return an (N, 3) float64 array."""
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xs = np.asarray(vertex["x"], dtype=np.float64)
    ys = np.asarray(vertex["y"], dtype=np.float64)
    zs = np.asarray(vertex["z"], dtype=np.float64)
    return np.column_stack((xs, ys, zs))


def load_e57(path: str | Path, scan_index: int = 0) -> np.ndarray:
    """Read an E57 file and return an (N, 3) float64 array.

    Parameters
    ----------
    path : str | Path
        Path to the ``.e57`` file.
    scan_index : int, optional
        Which scan (``Data3D`` entry) to read when the file contains
        multiple scans.  Defaults to ``0`` (the first scan).

    Returns
    -------
    np.ndarray
        An (N, 3) float64 array of XYZ point coordinates.
    """
    e57 = pye57.E57(str(path))
    try:
        raw = e57.read_scan_raw(scan_index)

        # E57 files use cartesianX/Y/Z for point coordinates
        xs = np.asarray(raw["cartesianX"], dtype=np.float64)
        ys = np.asarray(raw["cartesianY"], dtype=np.float64)
        zs = np.asarray(raw["cartesianZ"], dtype=np.float64)
        return np.column_stack((xs, ys, zs))
    finally:
        e57.close()


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Auto-detect format and return (N, 3) points.

    Raises ``ValueError`` for unsupported extensions.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".ply":
        return load_ply(p)
    if ext == ".e57":
        return load_e57(p)
    raise ValueError(
        f"Unsupported point-cloud format '{ext}'. Supported: .ply, .e57"
    )
