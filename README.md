# Digital Twin Real Estate Platform (Local-First, Codespaces-Ready)

This repo runs a full local-first stack:
- FastAPI (Python API)
- Celery worker (jobs)
- Postgres (DB)
- Redis (queue)
- MinIO (S3-compatible object storage)
- Optional: Adminer (DB UI), Flower (Celery UI)
- (Optional) Viewer (web)

It is designed to run in **GitHub Codespaces** and also run **locally** with the same commands.

---

## Quick Start (GitHub Codespaces)

### 1) Open in Codespaces
1. Click **Code → Codespaces → Create codespace on main**
2. Wait for the devcontainer to build.

### 2) Start services
In the terminal:

```bash
make up


# Digital Twin Real Estate Platform (Local-First) — Extensive TODO

## 0) Repo & Standards
- [ ] Create monorepo structure:
  - [ ] `apps/api` (FastAPI)
  - [ ] `apps/worker` (Celery)
  - [ ] `packages/core` (shared types + geometry primitives)
  - [ ] `packages/pipeline` (processing steps)
  - [ ] `packages/sims` (sun + airflow)
  - [ ] `infra` (docker-compose, init scripts)
  - [ ] `scripts` (dev helpers)
- [ ] Add `pyproject.toml` with:
  - [ ] FastAPI, uvicorn, pydantic, SQLAlchemy, alembic
  - [ ] celery, redis
  - [ ] minio client (S3 compatible)
  - [ ] open3d, numpy
  - [ ] astral or pysolar for sun position
  - [ ] pytest, ruff, mypy
- [ ] Add code quality:
  - [ ] `ruff` config
  - [ ] `mypy` config
  - [ ] `pytest` config
  - [ ] pre-commit hooks
- [ ] Add Makefile (or just scripts):
  - [ ] `make up` (docker compose up)
  - [ ] `make api` (run FastAPI)
  - [ ] `make worker` (run Celery)
  - [ ] `make test`
- [ ] Add `.env.example` for local secrets (DB creds, MinIO keys)

---

## 1) Local Infrastructure (Docker Compose)
- [ ] Create `infra/docker-compose.yml` with:
  - [ ] Postgres
  - [ ] Redis
  - [ ] MinIO (S3-compatible)
  - [ ] Adminer (DB admin UI)
  - [ ] Optional: Flower (Celery monitor)
- [ ] Add `infra/init-db.sql`:
  - [ ] create db + user
  - [ ] set extensions if needed
- [ ] Add healthchecks to all services
- [ ] Add persistent volumes for postgres/minio
- [ ] Document startup steps in `README.md`

---

## 2) Data Model (Core Entities)
- [ ] Define SQLAlchemy models:
  - [ ] `Project` (aka listing)
    - fields: id, name, address (optional), created_at
  - [ ] `Scan`
    - fields: id, project_id, source_type (BLK360/iPhone/etc), raw_object_key, status, created_at
  - [ ] `Artifact`
    - fields: id, scan_id, kind (E57, LAZ, PLY, GLB, PARAM_JSON, SIM_JSON), object_key, metadata_json, created_at
  - [ ] `RoomModel`
    - fields: id, scan_id, param_model_object_key, version, created_at
  - [ ] `SimulationRun`
    - fields: id, scan_id, type (SUN/AIRFLOW), params_json, output_object_key, status, created_at
  - [ ] `ViewerSession`
    - fields: id, scan_id, started_at, ended_at, user_agent, device_type
  - [ ] `ViewerEvent`
    - fields: id, session_id, ts, type, payload_json
- [ ] Create Alembic migrations for all tables
- [ ] Add basic seed script:
  - [ ] create sample Project
  - [ ] create Scan referencing a sample E57 (if present)

---

## 3) Object Storage (MinIO) Abstraction
- [ ] Implement `packages/core/storage.py`:
  - [ ] init MinIO client from env
  - [ ] ensure bucket exists on startup
  - [ ] functions:
    - [ ] `put_file(local_path) -> object_key`
    - [ ] `put_bytes(bytes) -> object_key`
    - [ ] `get_file(object_key, local_path)`
    - [ ] `presigned_get_url(object_key, expiry_seconds)`
- [ ] Define consistent object key layout:
  - [ ] `projects/{project_id}/scans/{scan_id}/raw/{filename}`
  - [ ] `projects/{project_id}/scans/{scan_id}/artifacts/{kind}/{filename}`
  - [ ] `projects/{project_id}/scans/{scan_id}/models/{version}/param.json`
  - [ ] `projects/{project_id}/scans/{scan_id}/sims/{sim_id}/output.json`

---

## 4) FastAPI (Local API)
### 4.1 Core setup
- [ ] Create FastAPI app:
  - [ ] config management (env)
  - [ ] DB session dependency
  - [ ] storage client dependency
  - [ ] CORS for local viewer
- [ ] Add `/health` endpoint

### 4.2 CRUD endpoints
- [ ] Projects:
  - [ ] `POST /projects` create project
  - [ ] `GET /projects` list projects
  - [ ] `GET /projects/{id}` get project
- [ ] Scans:
  - [ ] `POST /projects/{id}/scans` create scan (metadata only)
  - [ ] `GET /projects/{id}/scans` list scans
  - [ ] `GET /scans/{scan_id}` get scan
- [ ] Upload raw E57:
  - [ ] `POST /scans/{scan_id}/upload` multipart upload E57
  - [ ] store in MinIO
  - [ ] update Scan.status = `UPLOADED`
  - [ ] create Artifact(kind=E57)
  - [ ] return presigned URL + artifact info

### 4.3 Processing orchestration endpoints
- [ ] Start pipeline:
  - [ ] `POST /scans/{scan_id}/process`
  - [ ] enqueue Celery job `pipeline.process_scan(scan_id)`
  - [ ] set Scan.status = `PROCESSING`
- [ ] Job status:
  - [ ] `GET /scans/{scan_id}/status`
  - [ ] show artifacts present + current status
- [ ] Start simulations:
  - [ ] `POST /scans/{scan_id}/simulate/sun` (params: lat, lon, date, start_time, end_time, step_minutes)
  - [ ] `POST /scans/{scan_id}/simulate/airflow` (params: wind_dir optional, open_windows config optional)
  - [ ] enqueue Celery jobs
- [ ] Analytics ingestion:
  - [ ] `POST /viewer/sessions` create session
  - [ ] `POST /viewer/events` batch ingest events
  - [ ] `GET /scans/{scan_id}/analytics/summary` aggregated stats

---

## 5) Worker (Celery) + Pipeline Jobs
### 5.1 Worker setup
- [ ] Create Celery app in `apps/worker`:
  - [ ] Redis broker
  - [ ] result backend
  - [ ] task routing
  - [ ] structured logging

### 5.2 Core processing job
- [ ] Implement `pipeline.process_scan(scan_id)`:
  - [ ] download E57 from MinIO to temp dir
  - [ ] convert E57 -> LAZ (or PLY) via PDAL pipeline
    - [ ] store LAZ/PLY to MinIO as Artifact
  - [ ] generate a downsampled preview PLY for fast viewer load
    - [ ] store preview artifact
  - [ ] compute normals (Open3D)
  - [ ] plane detection (walls/floor/ceiling) (RANSAC)
  - [ ] build parametric model JSON:
    - [ ] global scale units = meters
    - [ ] floor plane
    - [ ] ceiling plane
    - [ ] wall planes with extents
  - [ ] room segmentation (v1 simple):
    - [ ] single-room fallback
    - [ ] store as “room graph v0”
  - [ ] store parametric JSON artifact
  - [ ] update Scan.status = `PROCESSED`
  - [ ] on error: store error log artifact + set status `FAILED`

### 5.3 Opening detection (windows/doors) v1
- [ ] Implement `pipeline.detect_openings(scan_id)` (can be part of process_scan or separate):
  - [ ] for each wall plane:
    - [ ] project nearby points to 2D wall coordinates
    - [ ] grid occupancy map
    - [ ] detect “void rectangles”
    - [ ] filter by plausible dimensions:
      - [ ] doors near floor (y ~ 0)
      - [ ] windows above sill height
    - [ ] output bounding rectangles
  - [ ] write openings into parametric model
  - [ ] store updated parametric JSON (version bump)
- [ ] Add a config file for heuristics:
  - [ ] min/max opening width/height
  - [ ] min sill height
  - [ ] tolerance for noisy glass points

### 5.4 Manual tagging fallback (must-have for MVP)
- [ ] Implement API + model support for manual override:
  - [ ] `POST /scans/{scan_id}/openings` add opening
  - [ ] `PATCH /scans/{scan_id}/openings/{opening_id}` edit opening
  - [ ] `DELETE /scans/{scan_id}/openings/{opening_id}`
- [ ] Ensure simulations use manual override data when present

---

## 6) Simulations
### 6.1 Sun/daylight simulation (v1)
- [ ] Create `packages/sims/sun.py`:
  - [ ] compute sun direction vector from lat/lon/date/time (astral/pysolar)
  - [ ] ray-cast through each window/opening into room volume
  - [ ] produce outputs:
    - [ ] `sun_hours_per_room`
    - [ ] `direct_sun_map` (coarse grid on floor)
    - [ ] `glare_risk_zones` (simple heuristic)
- [ ] Create Celery task `sims.run_sun_sim(scan_id, params)`
  - [ ] load parametric model
  - [ ] run sim
  - [ ] store output JSON to MinIO
  - [ ] record SimulationRun row in DB

### 6.2 Airflow/ventilation heuristic (v1)
- [ ] Create `packages/sims/airflow.py`:
  - [ ] compute room volume from planes/height
  - [ ] compute opening areas + distribution
  - [ ] cross-vent potential score (openings on opposing walls)
  - [ ] airflow “paths” (graph arrows, not CFD)
  - [ ] outputs:
    - [ ] `ventilation_score_per_room`
    - [ ] `cross_breeze_pairs`
    - [ ] `stagnation_zones` (heuristic)
- [ ] Create Celery task `sims.run_airflow_sim(scan_id, params)`
  - [ ] store output JSON + SimulationRun row

---

## 7) Viewer (Local Demo) — Minimal but Real
> Goal: a local viewer that loads a preview point cloud/mesh, shows rooms, toggles overlays, and sends analytics.

- [ ] Create minimal web viewer (choose one):
  - [ ] Option A: React + Three.js
  - [ ] Option B: plain Vite + Three.js (faster)
- [ ] Load artifacts:
  - [ ] fetch presigned URL for preview PLY/GLB
  - [ ] render point cloud or mesh
- [ ] Basic navigation:
  - [ ] grounded first-person (no flying)
  - [ ] user height slider (cm)
  - [ ] collision-ish floor clamp (simple)
- [ ] Overlay toggles (v1):
  - [ ] show/hide walls planes (wireframe)
  - [ ] show windows/doors bounding boxes
  - [ ] show sun map overlay (floor heat overlay)
  - [ ] show airflow arrows overlay
- [ ] “Hero snapshots” panel (placeholder for later AI refurb):
  - [ ] show captured screenshots list (static for now)
- [ ] Analytics instrumentation:
  - [ ] create session via API
  - [ ] send periodic camera pose samples (e.g., 5–10 Hz or throttled)
  - [ ] send click events on room/hotspot UI
  - [ ] send dwell-time per room based on volume containment

---

## 8) Analytics Aggregation (Agent Dashboard v1)
- [ ] Implement backend aggregation queries:
  - [ ] total sessions, unique sessions
  - [ ] average session duration
  - [ ] dwell time per room
  - [ ] heatmap bins (2D floor grid counts)
  - [ ] drop-off points (where sessions end)
- [ ] Create `GET /scans/{scan_id}/analytics/summary` response schema
- [ ] Create minimal dashboard page (can be in viewer or separate):
  - [ ] sessions over time
  - [ ] dwell per room bar chart
  - [ ] top hotspots list
  - [ ] toggle usage stats (empty/refurb/time-of-day once implemented)

---

## 9) Refurb / Empty States (MVP-friendly plan)
> Start with “hero-frame transforms” later; for now build the plumbing.

- [ ] Define “state system” in DB/model:
  - [ ] states: `ORIGINAL`, `EMPTY`, `STAGED_A`, `REFURB_A`, etc.
  - [ ] per-room assets list
- [ ] Create endpoints:
  - [ ] `POST /scans/{scan_id}/states` create state
  - [ ] `GET /scans/{scan_id}/states` list states
- [ ] Add placeholder artifacts for future:
  - [ ] per-room images for each state
  - [ ] optional per-room mesh overrides later

---

## 10) Window Replacement / Parametric Edits (Core Differentiator)
- [ ] Implement parametric edits API:
  - [ ] `POST /scans/{scan_id}/edits/window_replace`
    - [ ] target opening_id
    - [ ] replacement type (double-glazed, enlarge, convert_to_door, skylight)
    - [ ] update param model
- [ ] Update viewer to show replacement variants (bounding box + label)
- [ ] Ensure sun/airflow sims rerun against edited model

---

## 11) Test Data & Reproducible Demo
- [ ] Add `data/README.md` describing where to place sample E57
- [ ] Add script to ingest a sample:
  - [ ] create project/scan
  - [ ] upload sample file
  - [ ] trigger process + sims
- [ ] Add “one command demo” script:
  - [ ] start services
  - [ ] ingest sample
  - [ ] open viewer URL
- [ ] Add snapshot tests for:
  - [ ] parametric JSON schema validation
  - [ ] plane detection stability on sample data

---

## 12) Documentation (Must for agents)
- [ ] Write `README.md`:
  - [ ] setup prerequisites
  - [ ] how to run locally
  - [ ] how to ingest E57
  - [ ] how to run pipeline
  - [ ] how to run sims
  - [ ] how to open viewer
- [ ] Add architecture doc:
  - [ ] components diagram
  - [ ] data flow
  - [ ] artifact formats
- [ ] Add “Accuracy + limitations” doc:
  - [ ] glass/mirrors issues
  - [ ] confidence levels in detection/sims

---

## 13) Nice-to-haves (After MVP)
- [ ] Multi-room segmentation (real room graph)
- [ ] Better opening detection using ML segmentation
- [ ] Gaussian splat viewer pipeline
- [ ] Automatic orientation estimation (north alignment)
- [ ] Mobile-friendly viewer UX
- [ ] Real-time collaboration notes for agents
- [ ] Export packages:
  - [ ] PDF summary (light, airflow, attention)
  - [ ] marketing images bundle

---

# MVP Definition (so scope doesn’t explode)
## MVP = local demo success criteria
- [ ] Upload E57
- [ ] Convert + preview artifact produced
- [ ] Parametric planes detected (walls/floor/ceiling)
- [ ] Manual openings added OR auto openings mostly working
- [ ] Sun sim produces per-room results + overlay
- [ ] Airflow heuristic produces per-room results + overlay
- [ ] Viewer shows model + grounded navigation + height slider



- [ ] Viewer logs analytics events
- [ ] Basic analytics summary endpoint works
