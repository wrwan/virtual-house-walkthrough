"""Pydantic models for the parametric model and pipeline artefacts.

The parametric model is the structured JSON output of the scan-processing
pipeline.  It describes the detected planes (walls, floor, ceiling), room
extents, and – later – openings, simulation overlays, etc.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── tiny helpers ──────────────────────────────────────────────────────
class Vec3(BaseModel):
    """A 3-component vector (x, y, z) in metres."""

    x: float
    y: float
    z: float


class BBox(BaseModel):
    """Axis-aligned bounding box."""

    min: Vec3
    max: Vec3


# ── plane / surface types ────────────────────────────────────────────
class PlaneKind(str, Enum):
    FLOOR = "floor"
    CEILING = "ceiling"
    WALL = "wall"
    UNKNOWN = "unknown"


class DetectedPlane(BaseModel):
    """A plane detected by RANSAC, classified by orientation."""

    kind: PlaneKind
    normal: Vec3
    offset: float = Field(description="Signed distance from origin (Hesse normal form: n·x = d)")
    inlier_count: int = 0
    bounds: Optional[BBox] = None


# ── parametric model ─────────────────────────────────────────────────
class ParametricModel(BaseModel):
    """Top-level parametric model produced by the processing pipeline."""

    version: str = "0.1.0"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    units: str = "metres"
    source_file: str = ""
    point_count: int = 0
    bounds: Optional[BBox] = None
    planes: list[DetectedPlane] = Field(default_factory=list)


# ── scan status (used later by the API / DB layer) ───────────────────
class ScanStatus(str, Enum):
    CREATED = "CREATED"
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
