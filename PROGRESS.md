# Implementation Progress

> Auto-generated status of what has been built and what remains.
> Last updated: 2026-02-08

---

## ✅ What Has Been Implemented

### 0) Repo & Standards

| Item | Status |
|------|--------|
| Monorepo structure (`packages/core`, `packages/pipeline`, `apps/api`, `viewer`) | ✅ Done |
| `pyproject.toml` with deps (numpy, plyfile, pye57, pydantic, click, scipy, fastapi, uvicorn) | ✅ Done |
| `ruff` config (line-length 100, py310, E/F/W/I rules) | ✅ Done |
| `pytest` config (testpaths = tests/) | ✅ Done |
| Makefile (`make install`, `make test`, `make lint`, `make api`, `make viewer`) | ✅ Done |
| `.env.example` (Postgres, Redis, MinIO placeholders) | ✅ Done |
| `mypy` config | ❌ Not done |
| pre-commit hooks | ❌ Not done |
| `apps/worker` (Celery) | ❌ Not done |
| `packages/sims` | ❌ Not done |
| `infra/` (docker-compose) | ❌ Not done |
| `scripts/` (dev helpers) | ❌ Not done |

### 1) Local Infrastructure (Docker Compose)

**Not implemented.** No `docker-compose.yml`, no Postgres/Redis/MinIO containers.
The current stack runs without any infrastructure — the API uses in-memory state, no database, no object storage.

### 2) Data Model (Core Entities)

| Item | Status |
|------|--------|
| Pydantic types: `Vec3`, `BBox`, `PlaneKind`, `DetectedPlane`, `ParametricModel`, `ScanStatus` | ✅ Done |
| SQLAlchemy models (Project, Scan, Artifact, RoomModel, SimulationRun, ViewerSession, ViewerEvent) | ❌ Not done |
| Alembic migrations | ❌ Not done |
| Seed script | ❌ Not done |

### 3) Object Storage (MinIO) Abstraction

**Not implemented.** No `storage.py`, no MinIO client. Files are stored in-memory only.

### 4) FastAPI (Local API)

| Item | Status |
|------|--------|
| FastAPI app with CORS | ✅ Done |
| `/health` endpoint | ✅ Done |
| `POST /upload` — multipart upload of PLY/E57, runs pipeline, stores in memory | ✅ Done |
| `GET /points` — returns point cloud as flat float32 array (with large-cloud downsampling) | ✅ Done |
| `GET /model` — returns parametric model JSON | ✅ Done |
| Config management from env | ❌ Not done (hardcoded) |
| DB session dependency | ❌ Not done |
| Storage client dependency | ❌ Not done |
| Project/Scan CRUD endpoints | ❌ Not done |
| Processing orchestration endpoints (`/process`, `/status`) | ❌ Not done |
| Simulation trigger endpoints (`/simulate/sun`, `/simulate/airflow`) | ❌ Not done |
| Analytics ingestion endpoints (`/viewer/sessions`, `/viewer/events`) | ❌ Not done |

### 5) Worker (Celery) + Pipeline Jobs

#### 5.1 Worker Setup
**Not implemented.** No Celery app, no Redis broker. Processing runs synchronously inside the API request.

#### 5.2 Core Processing Pipeline

| Item | Status |
|------|--------|
| Point-cloud loader: PLY format (plyfile) | ✅ Done |
| Point-cloud loader: E57 format (pye57) | ✅ Done |
| Auto-detect format by extension | ✅ Done |
| Voxel-grid downsampling (centroid-based) | ✅ Done |
| Normal estimation (PCA on k-nearest neighbours via scipy cKDTree) | ✅ Done |
| Bounding box computation | ✅ Done |
| RANSAC plane detection (iterative, inlier removal) | ✅ Done |
| Plane classification: floor / ceiling / wall (by normal orientation + height) | ✅ Done |
| Parametric model JSON assembly (`ParametricModel` with planes + bounds) | ✅ Done |
| CLI entry-point: `dt-pipeline process <file> [-o output.json]` | ✅ Done |
| End-to-end pipeline: `process_scan()` and `process_scan_to_json()` | ✅ Done |
| Room segmentation | ❌ Not done |
| E57 → LAZ/PLY conversion via PDAL | ❌ Not done |
| Downsampled preview PLY generation | ❌ Not done |

#### 5.3 Opening Detection (windows/doors)
**Not implemented.** No void-rectangle detection, no occupancy grid, no opening heuristics.

#### 5.4 Manual Tagging Fallback
**Not implemented.** No manual opening CRUD endpoints.

### 6) Simulations

**Not implemented.** No `packages/sims/`, no sun simulation, no airflow heuristic.

### 7) Viewer (Local Demo)

| Item | Status |
|------|--------|
| Vite + Three.js app (`viewer/`) | ✅ Done |
| Drag-and-drop file upload UI | ✅ Done |
| Point cloud rendering with height-based colour ramp | ✅ Done |
| Orbit controls (camera navigation) | ✅ Done |
| Detected planes as wireframe + translucent fill overlays | ✅ Done |
| Colour coding: floor=green, ceiling=blue, walls=orange | ✅ Done |
| Toggle controls per plane type (walls/floor/ceiling/all) | ✅ Done |
| HUD with scan stats (file, points, planes, walls, floor, ceiling) | ✅ Done |
| Reset camera button | ✅ Done |
| Upload-new button | ✅ Done |
| Grid + axes helpers | ✅ Done |
| Vite proxy `/api/*` → FastAPI on port 8000 | ✅ Done |
| Grounded first-person navigation | ❌ Not done (orbit only) |
| User height slider | ❌ Not done |
| Floor clamp / collision | ❌ Not done |
| Window/door bounding box overlay | ❌ Not done |
| Sun map overlay (floor heat) | ❌ Not done |
| Airflow arrows overlay | ❌ Not done |
| Hero snapshots panel | ❌ Not done |
| Analytics instrumentation (session, events, dwell-time) | ❌ Not done |

### 8) Analytics Aggregation

**Not implemented.** No aggregation queries, no dashboard.

### 9) Refurb / Empty States

**Not implemented.** No state system, no endpoints.

### 10) Window Replacement / Parametric Edits

**Not implemented.** No edit endpoints, no viewer support.

### 11) Test Data & Reproducible Demo

| Item | Status |
|------|--------|
| Synthetic room fixture in tests (floor + ceiling + 4 walls) | ✅ Done |
| `data/README.md` describing sample E57 | ❌ Not done |
| Ingest script | ❌ Not done |
| One-command demo script | ❌ Not done |
| Snapshot/schema tests for parametric JSON | ❌ Not done |

### 12) Documentation

| Item | Status |
|------|--------|
| `README.md` with setup + usage instructions | ✅ Done |
| `PROGRESS.md` (this file) — what is / isn't implemented | ✅ Done |
| `docs/ROADMAP.md` — full TODO/plan | ✅ Done |
| Architecture doc (components diagram, data flow) | ❌ Not done |
| Accuracy + limitations doc | ❌ Not done |

### 13) Nice-to-haves

**None implemented.** Multi-room segmentation, ML opening detection, Gaussian splats, mobile UX, exports — all future work.

---

## Tests

| Test file | Tests | Status |
|-----------|-------|--------|
| `tests/test_loader.py` | PLY round-trip, E57 round-trip, dispatch, unsupported format | ✅ 5 pass |
| `tests/test_preprocess.py` | bounds, downsample, single-voxel, invalid voxel | ✅ 4 pass |
| `tests/test_plane_detection.py` | finds floor+ceiling, finds walls, total count, empty cloud, inlier counts | ✅ 5 pass |
| `tests/test_process.py` | produces valid model, JSON output | ✅ 2 pass |
| `tests/test_api.py` | health, upload PLY, bad format, points before/after upload, model before/after upload | ✅ 7 pass |
| **Total** | | **23 tests passing** |

---

## Tech Stack (as implemented)

| Layer | Technology |
|-------|-----------|
| **Backend API** | FastAPI 0.100+, Uvicorn |
| **Pipeline** | NumPy, SciPy, plyfile, pye57 |
| **Types** | Pydantic v2 |
| **CLI** | Click |
| **Viewer** | Vite, Three.js |
| **Linting** | Ruff |
| **Testing** | Pytest, httpx |
| **Language** | Python ≥3.10, JavaScript (ES modules) |

---

## MVP Checklist (from roadmap)

| Criteria | Status |
|----------|--------|
| Upload E57 | ✅ (also PLY) |
| Convert + preview artifact produced | ⚠️ Partial — pipeline processes inline, no separate artifact storage |
| Parametric planes detected (walls/floor/ceiling) | ✅ |
| Manual openings added OR auto openings mostly working | ❌ |
| Sun sim produces per-room results + overlay | ❌ |
| Airflow heuristic produces per-room results + overlay | ❌ |
| Viewer shows model + grounded navigation + height slider | ⚠️ Partial — orbit camera, no grounded nav/height slider |
| Viewer logs analytics events | ❌ |
| Basic analytics summary endpoint works | ❌ |
