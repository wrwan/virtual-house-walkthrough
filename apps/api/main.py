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

from packages.core.types import ParametricModel
from packages.pipeline.loader import load_point_cloud
from packages.pipeline.process import process_scan

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

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Load raw points
        points = load_point_cloud(tmp_path)

        # Run processing pipeline
        model = process_scan(tmp_path, seed=42)
        model.source_file = file.filename  # use original name, not temp path

        # Store in memory
        _state["points"] = points
        _state["model"] = model
        _state["source_file"] = file.filename

        return {
            "filename": file.filename,
            "point_count": len(points),
            "planes_detected": len(model.planes),
        }
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/points")
def get_points():
    """Return point cloud as a flat JSON array of [x, y, z, ...] coordinates.

    For performance with large clouds we send a flat Float32 list that the
    viewer can directly load into a Three.js BufferGeometry.
    """
    if _state["points"] is None:
        raise HTTPException(404, "No scan uploaded yet")

    pts: np.ndarray = _state["points"]

    # Downsample for the viewer if the cloud is very large
    max_viewer_points = 200_000
    if len(pts) > max_viewer_points:
        step = len(pts) // max_viewer_points
        pts = pts[::step][:max_viewer_points]

    flat = pts.astype(np.float32).ravel().tolist()
    return {"count": len(pts), "positions": flat}


@app.get("/model")
def get_model():
    """Return the parametric model JSON (planes, bounds, etc.)."""
    if _state["model"] is None:
        raise HTTPException(404, "No scan uploaded yet")

    model: ParametricModel = _state["model"]
    return JSONResponse(content=json.loads(model.model_dump_json()))
