# Digital Twin Real Estate Platform

Upload a point-cloud scan (PLY or E57), detect interior walls/floor/ceiling via RANSAC, and visualise the result in a 3D web viewer.

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18

### 1) Install Python dependencies

```bash
pip install -e ".[dev]"
```

### 2) Start the API server

```bash
make api
# → runs on http://localhost:8000
```

### 3) Start the viewer

In a second terminal:

```bash
make viewer
# → runs on http://localhost:5173
```

### 4) Open the viewer

Go to **http://localhost:5173**, drag-and-drop a `.ply` or `.e57` file, and see:

- The point cloud rendered with a height-based colour ramp
- Detected planes overlaid as wireframe boxes (green = floor, blue = ceiling, orange = walls)
- Toggle controls for each overlay type

## CLI Usage

You can also process a scan from the command line without the API/viewer:

```bash
dt-pipeline process scan.ply -o model.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<input>.parametric.json` | Output JSON path |
| `--voxel-size` | `0.05` | Voxel grid size in metres |
| `--max-planes` | `10` | Maximum planes to detect |
| `--seed` | `42` | RANSAC random seed |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PLY/E57 file (multipart) — runs the processing pipeline |
| `GET` | `/points` | Point cloud as flat float32 array for Three.js |
| `GET` | `/model` | Parametric model JSON (detected planes, bounds, metadata) |

## Project Structure

```
├── apps/api/            FastAPI backend
│   └── main.py          API endpoints (upload, points, model)
├── packages/
│   ├── core/
│   │   └── types.py     Pydantic models (Vec3, BBox, DetectedPlane, ParametricModel)
│   └── pipeline/
│       ├── cli.py       Click CLI entry-point
│       ├── loader.py    PLY + E57 file loading
│       ├── preprocess.py  Voxel downsampling, normal estimation, bounds
│       ├── plane_detection.py  RANSAC plane detection + classification
│       ├── parametric.py  Parametric model builder
│       └── process.py   End-to-end pipeline orchestration
├── viewer/              Vite + Three.js web viewer
│   ├── index.html       Viewer page with upload UI + HUD + controls
│   ├── src/main.js      Three.js scene, point cloud + plane rendering
│   └── vite.config.js   Dev server + API proxy config
├── tests/               Pytest test suite (23 tests)
├── docs/
│   └── ROADMAP.md       Full project TODO / roadmap
├── PROGRESS.md          What is implemented vs. what remains
├── pyproject.toml       Python project config + dependencies
└── Makefile             Dev commands
```

## Make Targets

| Command | Description |
|---------|-------------|
| `make install` | Install Python deps in editable mode |
| `make test` | Run pytest |
| `make lint` | Run ruff linter |
| `make api` | Start FastAPI dev server (port 8000) |
| `make viewer` | Start Vite dev server (port 5173) |

## Running Tests

```bash
make test
# or
python -m pytest tests/ -v
```

## What's Implemented / What's Not

See **[PROGRESS.md](PROGRESS.md)** for a full breakdown of implemented vs. remaining features.

See **[docs/ROADMAP.md](docs/ROADMAP.md)** for the complete project roadmap.
