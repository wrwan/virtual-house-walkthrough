"""FastAPI application for the Digital Twin platform.

Serves uploaded point-cloud files, runs the processing pipeline, and
provides the parametric model JSON + raw point data to the viewer.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from packages.core.types import DetectedPlane, ParametricModel
from packages.pipeline.loader import load_point_cloud
from packages.pipeline.plane_detection import (
    normalize_walls,
    refine_wall_from_corners,
    trim_wall_at_intersection,
)
from packages.pipeline.process import process_scan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Digital Twin API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permissive for local development; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store (single-scan MVP) ────────────────────────────────
_state: dict = {
    "points": None,       # np.ndarray (N, 3) or None
    "colors": None,       # np.ndarray (N, 3) or None — RGB in [0, 1]
    "model": None,        # ParametricModel or None
    "source_file": None,  # original filename
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_scan(file: UploadFile = File(...)):
    """Upload a point-cloud file (PLY or E57), process it, and store results."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".ply", ".e57"):
        raise HTTPException(400, f"Unsupported format '{suffix}'. Use .ply or .e57")

    logger.info(f"📥 Receiving file: {file.filename} ({suffix})")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        logger.info(f"💾 Saving to temporary file: {tmp.name}")
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Load raw points
        logger.info(f"📂 Loading point cloud from {file.filename}...")
        cloud_data = load_point_cloud(tmp_path)
        points = cloud_data["positions"]
        colors = cloud_data["colors"]
        logger.info(f"✅ Loaded {len(points):,} points")
        if colors is not None:
            logger.info(f"🎨 RGB color data available")

        # Run processing pipeline
        logger.info(f"⚙️  Running processing pipeline...")
        model = process_scan(tmp_path, seed=42)
        model.source_file = file.filename  # use original name, not temp path
        logger.info(f"✅ Processing complete: {len(model.planes)} planes detected")

        # Store in memory
        _state["points"] = points
        _state["colors"] = colors
        _state["model"] = model
        _state["source_file"] = file.filename

        logger.info(f"🎉 Upload and processing successful!")
        return {
            "filename": file.filename,
            "point_count": len(points),
            "planes_detected": len(model.planes),
        }
    except Exception as e:
        logger.exception("❌ Processing failed")
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        logger.info(f"🧹 Cleaning up temporary file")
        tmp_path.unlink(missing_ok=True)


@app.get("/points")
def get_points():
    """Return point cloud as a flat JSON array of [x, y, z, ...] coordinates
    plus optional RGB colors.

    For performance with large clouds we send a flat Float32 list that the
    viewer can directly load into a Three.js BufferGeometry.
    """
    if _state["points"] is None:
        raise HTTPException(404, "No scan uploaded yet")

    pts: np.ndarray = _state["points"]
    colors: np.ndarray | None = _state["colors"]
    logger.info(f"📊 Preparing {len(pts):,} points for viewer")

    # # Downsample for the viewer if the cloud is very large
    # max_viewer_points = 200_000
    # if len(pts) > max_viewer_points:
    #     step = len(pts) // max_viewer_points
    #     indices = np.arange(0, len(pts), step)[:max_viewer_points]
    #     pts = pts[indices]
    #     if colors is not None:
    #         colors = colors[indices]
    #     logger.info(f"⬇️  Downsampled to {len(pts):,} points for performance")

    flat_positions = pts.astype(np.float32).ravel().tolist()

    result = {"count": len(pts), "positions": flat_positions, "has_colors": colors is not None}
    if colors is not None:
        flat_colors = colors.astype(np.float32).ravel().tolist()
        result["colors"] = flat_colors
        logger.info(f"🎨 Sending {len(pts):,} points with RGB colors to viewer")
    else:
        logger.info(f"✅ Sending {len(pts):,} points (no RGB) to viewer")

    return result


@app.get("/model")
def get_model():
    """Return the parametric model JSON (planes, bounds, etc.)."""
    if _state["model"] is None:
        raise HTTPException(404, "No scan uploaded yet")

    model: ParametricModel = _state["model"]
    logger.info(f"📐 Sending parametric model with {len(model.planes)} planes")
    return JSONResponse(content=json.loads(model.model_dump_json()))


from pydantic import BaseModel as PydanticBaseModel


class ManualWallRequest(PydanticBaseModel):
    """Body for the manual wall endpoint."""
    corners: list[list[float]]  # [[x, y, z], ...] — 3 or 4 corners
    search_radius: float = 0.3


@app.post("/manual-wall")
def add_manual_wall(req: ManualWallRequest):
    """User picks corners in the viewer → refine against the point cloud."""
    if _state["points"] is None:
        raise HTTPException(404, "No scan uploaded yet")
    if len(req.corners) < 3:
        raise HTTPException(400, "Need at least 3 corners")

    logger.info(f"🖐️  Manual wall requested with {len(req.corners)} corners")
    try:
        plane = refine_wall_from_corners(
            _state["points"],
            req.corners,
            search_radius=req.search_radius,
        )
    except Exception as e:
        logger.exception("Manual wall refinement failed")
        raise HTTPException(500, f"Refinement failed: {e}")

    # Append to the current model
    model: ParametricModel = _state["model"]
    model.planes.append(plane)
    logger.info(f"✅ Manual wall added — model now has {len(model.planes)} planes")

    return JSONResponse(content=json.loads(plane.model_dump_json()))


@app.delete("/wall/{index}")
def delete_wall(index: int):
    """Delete a wall by its index in the planes list."""
    if _state["model"] is None:
        raise HTTPException(404, "No scan uploaded yet")

    model: ParametricModel = _state["model"]
    if index < 0 or index >= len(model.planes):
        raise HTTPException(400, f"Invalid wall index {index} (have {len(model.planes)} planes)")

    removed = model.planes.pop(index)
    logger.info(f"🗑️  Deleted wall {index} ({removed.inlier_count} pts) — {len(model.planes)} planes remaining")
    return {"deleted": index, "remaining": len(model.planes)}


class TrimWallRequest(PydanticBaseModel):
    """Body for the wall trimming endpoint."""
    wall_index: int
    clipper_index: int


@app.post("/trim-wall")
def trim_wall(req: TrimWallRequest):
    """Trim a wall so it doesn't poke through another wall.

    The wall at *wall_index* is clipped at its intersection with the wall
    at *clipper_index*.  The clipped wall is updated in-place.
    """
    if _state["model"] is None:
        raise HTTPException(404, "No scan uploaded yet")

    model: ParametricModel = _state["model"]
    n = len(model.planes)
    if req.wall_index < 0 or req.wall_index >= n:
        raise HTTPException(400, f"Invalid wall_index {req.wall_index}")
    if req.clipper_index < 0 or req.clipper_index >= n:
        raise HTTPException(400, f"Invalid clipper_index {req.clipper_index}")
    if req.wall_index == req.clipper_index:
        raise HTTPException(400, "wall_index and clipper_index must differ")

    wall = model.planes[req.wall_index]
    clipper = model.planes[req.clipper_index]

    logger.info(f"✂️  Trimming wall {req.wall_index} at wall {req.clipper_index}")
    try:
        trimmed = trim_wall_at_intersection(wall, clipper)
    except Exception as e:
        logger.exception("Trim failed")
        raise HTTPException(500, f"Trim failed: {e}")

    model.planes[req.wall_index] = trimmed
    logger.info("✅ Wall trimmed successfully")
    return JSONResponse(content=json.loads(trimmed.model_dump_json()))


@app.post("/normalize-walls")
def normalize_walls_endpoint():
    """Snap walls to clean angles and uniform thickness."""
    if _state["model"] is None:
        raise HTTPException(404, "No scan uploaded yet")

    model: ParametricModel = _state["model"]
    logger.info(f"📐 Normalizing {len(model.planes)} walls")

    try:
        model.planes = normalize_walls(model.planes)
    except Exception as e:
        logger.exception("Normalization failed")
        raise HTTPException(500, f"Normalization failed: {e}")

    logger.info(f"✅ Normalized to {len(model.planes)} walls")
    return JSONResponse(content=json.loads(model.model_dump_json()))
