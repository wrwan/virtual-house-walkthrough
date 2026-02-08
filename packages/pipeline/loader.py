"""Load point-cloud files into a NumPy (N, 3) array.

Supported formats
-----------------
* **PLY** – via the ``plyfile`` library.
* **E57** – via the ``pye57`` library (ASTM E2807 standard, used by BLK360 etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pye57
from plyfile import PlyData

logger = logging.getLogger(__name__)


def load_ply(path: str | Path) -> dict:
    """Read a binary or ASCII PLY file and return positions + optional RGB.

    Returns a dict with:
      - 'positions': (N, 3) float64 array of XYZ coordinates
      - 'colors': (N, 3) float64 array of RGB values in [0, 1], or None
    """
    logger.info(f"📄 Reading PLY file...")
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xs = np.asarray(vertex["x"], dtype=np.float64)
    ys = np.asarray(vertex["y"], dtype=np.float64)
    zs = np.asarray(vertex["z"], dtype=np.float64)
    positions = np.column_stack((xs, ys, zs))
    logger.info(f"✅ PLY file loaded: {len(positions):,} vertices")

    # Check for RGB color data (supports multiple naming conventions)
    colors = None
    prop_names = [p.name for p in vertex.properties]
    logger.info(f"📋 PLY properties: {prop_names}")

    # Common PLY color property names
    red_candidates = ["red", "diffuse_red", "r"]
    green_candidates = ["green", "diffuse_green", "g"]
    blue_candidates = ["blue", "diffuse_blue", "b"]

    r_name = next((n for n in red_candidates if n in prop_names), None)
    g_name = next((n for n in green_candidates if n in prop_names), None)
    b_name = next((n for n in blue_candidates if n in prop_names), None)

    if r_name and g_name and b_name:
        logger.info(f"🎨 Found color properties: {r_name}, {g_name}, {b_name}")
        r = np.asarray(vertex[r_name], dtype=np.float64)
        g = np.asarray(vertex[g_name], dtype=np.float64)
        b = np.asarray(vertex[b_name], dtype=np.float64)
        # Normalise to [0, 1] – PLY typically stores 0-255 uint8
        if r.max() > 1.0:
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0
        colors = np.column_stack((r, g, b))
        logger.info(f"🎨 RGB color data loaded for {len(colors):,} points")
    else:
        logger.info(f"⚪ No RGB color data in PLY file")

    return {"positions": positions, "colors": colors}


def load_e57(path: str | Path, scan_index: int = 0) -> dict:
    """Read an E57 file and return positions + optional RGB.

    Parameters
    ----------
    path : str | Path
        Path to the ``.e57`` file.
    scan_index : int, optional
        Which scan (``Data3D`` entry) to read when the file contains
        multiple scans.  Defaults to ``0`` (the first scan).

    Returns
    -------
    dict
        A dict with 'positions' (N, 3) float64 and 'colors' (N, 3) float64 or None.
    """
    logger.info(f"📄 Opening E57 file (scan index {scan_index})...")
    e57 = pye57.E57(str(path))
    try:
        logger.info(f"📖 Reading scan data from E57...")
        raw = e57.read_scan_raw(scan_index)

        # E57 files use cartesianX/Y/Z for point coordinates
        logger.info(f"🔄 Converting cartesian coordinates...")
        xs = np.asarray(raw["cartesianX"], dtype=np.float64)
        ys = np.asarray(raw["cartesianY"], dtype=np.float64)
        zs = np.asarray(raw["cartesianZ"], dtype=np.float64)
        positions = np.column_stack((xs, ys, zs))
        logger.info(f"✅ E57 file loaded: {len(positions):,} points")

        # Check for RGB color data
        colors = None
        available_keys = list(raw.keys())
        logger.info(f"📋 E57 fields: {available_keys}")
        if "colorRed" in raw and "colorGreen" in raw and "colorBlue" in raw:
            r = np.asarray(raw["colorRed"], dtype=np.float64)
            g = np.asarray(raw["colorGreen"], dtype=np.float64)
            b = np.asarray(raw["colorBlue"], dtype=np.float64)
            if r.max() > 1.0:
                r = r / 255.0
                g = g / 255.0
                b = b / 255.0
            colors = np.column_stack((r, g, b))
            logger.info(f"🎨 RGB color data found for {len(colors):,} points")
        else:
            logger.info(f"⚪ No RGB color data in E57 file")

        return {"positions": positions, "colors": colors}
    finally:
        e57.close()
        logger.info(f"🔒 E57 file closed")


def load_point_cloud(path: str | Path) -> dict:
    """Auto-detect format and return a dict with 'positions' and optional 'colors'.

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
