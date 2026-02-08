"""Tests for the FastAPI backend (apps/api/main.py)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from plyfile import PlyData, PlyElement

from apps.api.main import _state, app


@pytest.fixture()
def client():
    """Fresh test client with cleared state."""
    _state["points"] = None
    _state["model"] = None
    _state["source_file"] = None
    return TestClient(app)


def _make_ply_bytes(points: np.ndarray) -> bytes:
    """Create an in-memory binary PLY file and return its bytes."""
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    structured = np.empty(len(points), dtype=dtype)
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    el = PlyElement.describe(structured, "vertex")
    buf = io.BytesIO()
    PlyData([el], text=False).write(buf)
    return buf.getvalue()


def _sample_room_ply() -> bytes:
    """Small synthetic room (floor + ceiling + 4 walls) as PLY bytes."""
    rng = np.random.default_rng(42)
    planes = []
    # Floor
    u, v = rng.uniform(-2, 2, (300, 1)), rng.uniform(-2, 2, (300, 1))
    planes.append(np.hstack([u, v, np.zeros((300, 1))]) + rng.normal(0, 0.005, (300, 3)))
    # Ceiling
    u, v = rng.uniform(-2, 2, (300, 1)), rng.uniform(-2, 2, (300, 1))
    planes.append(np.hstack([u, v, 3 * np.ones((300, 1))]) + rng.normal(0, 0.005, (300, 3)))
    # Walls
    for axis, sign in [(0, -1), (0, 1), (1, -1), (1, 1)]:
        u_ax = rng.uniform(-2, 2, (200, 1))
        v_ax = rng.uniform(0, 3, (200, 1))
        wall = np.zeros((200, 3))
        wall[:, axis] = sign * 2
        wall[:, 1 - axis] = u_ax.ravel()
        wall[:, 2] = v_ax.ravel()
        wall += rng.normal(0, 0.005, (200, 3))
        planes.append(wall)
    return _make_ply_bytes(np.vstack(planes).astype(np.float32))


class TestHealth:
    def test_health(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestUpload:
    def test_upload_ply(self, client: TestClient):
        ply_bytes = _sample_room_ply()
        r = client.post(
            "/upload", files={"file": ("room.ply", ply_bytes, "application/octet-stream")}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["point_count"] > 0
        assert data["planes_detected"] >= 4

    def test_upload_bad_format(self, client: TestClient):
        r = client.post("/upload", files={"file": ("bad.xyz", b"junk", "application/octet-stream")})
        assert r.status_code == 400


class TestEndpoints:
    def test_points_before_upload(self, client: TestClient):
        r = client.get("/points")
        assert r.status_code == 404

    def test_model_before_upload(self, client: TestClient):
        r = client.get("/model")
        assert r.status_code == 404

    def test_points_after_upload(self, client: TestClient):
        ply_bytes = _sample_room_ply()
        client.post("/upload", files={"file": ("room.ply", ply_bytes, "application/octet-stream")})
        r = client.get("/points")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] > 0
        assert len(data["positions"]) == data["count"] * 3

    def test_model_after_upload(self, client: TestClient):
        ply_bytes = _sample_room_ply()
        client.post("/upload", files={"file": ("room.ply", ply_bytes, "application/octet-stream")})
        r = client.get("/model")
        assert r.status_code == 200
        model = r.json()
        assert "planes" in model
        assert "bounds" in model
        assert model["units"] == "metres"
        # Should have wall planes
        wall_count = sum(1 for p in model["planes"] if p["kind"] == "wall")
        assert wall_count >= 2
