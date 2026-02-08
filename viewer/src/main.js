/**
 * Digital Twin Viewer — main entry point.
 *
 * Renders a point cloud + detected-plane overlays using Three.js.
 * Talks to the FastAPI backend at /api/* (proxied by Vite dev server).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API = '/api';

// ── Three.js scene setup ────────────────────────────────────────────
const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x1a1a2e);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(5, 5, 5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;

// Ambient + directional light for plane meshes
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// Grid helper (faint)
const grid = new THREE.GridHelper(20, 40, 0x333355, 0x222244);
scene.add(grid);

// Axes helper
scene.add(new THREE.AxesHelper(2));

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ── Groups for toggling ─────────────────────────────────────────────
const pointCloudGroup = new THREE.Group();
pointCloudGroup.name = 'pointCloud';
scene.add(pointCloudGroup);

const planeGroup = new THREE.Group();
planeGroup.name = 'planes';
scene.add(planeGroup);

// Per-kind sub-groups for individual toggles
const wallGroup = new THREE.Group(); wallGroup.name = 'walls';
const floorGroup = new THREE.Group(); floorGroup.name = 'floor';
const ceilingGroup = new THREE.Group(); ceilingGroup.name = 'ceiling';
planeGroup.add(wallGroup, floorGroup, ceilingGroup);

// ── Colours ─────────────────────────────────────────────────────────
const PLANE_COLORS = {
  floor: 0x4caf50,
  ceiling: 0x2196f3,
  wall: 0xff9800,
  unknown: 0x999999,
};

// ── Upload flow ─────────────────────────────────────────────────────
const overlay = document.getElementById('upload-overlay');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusEl = document.getElementById('upload-status');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  statusEl.textContent = `Uploading ${file.name}…`;
  try {
    const form = new FormData();
    form.append('file', file);

    const res = await fetch(`${API}/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const info = await res.json();
    statusEl.textContent = `Processing complete — ${info.point_count.toLocaleString()} points, ${info.planes_detected} planes`;

    // Load data into the scene
    await Promise.all([loadPoints(), loadModel(info)]);

    // Hide overlay, show HUD + controls
    overlay.classList.add('hidden');
    document.getElementById('hud').style.display = '';
    document.getElementById('controls').style.display = '';
  } catch (err) {
    statusEl.textContent = `❌ ${err.message}`;
  }
}

// ── Load point cloud ────────────────────────────────────────────────
async function loadPoints() {
  const res = await fetch(`${API}/points`);
  const data = await res.json();
  const positions = new Float32Array(data.positions);

  // Clear previous
  while (pointCloudGroup.children.length) pointCloudGroup.remove(pointCloudGroup.children[0]);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  // Colour each point by height (Z value) for visual depth
  const count = positions.length / 3;
  const colors = new Float32Array(count * 3);
  let minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const z = positions[i * 3 + 2];
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const rangeZ = maxZ - minZ || 1;
  for (let i = 0; i < count; i++) {
    const t = (positions[i * 3 + 2] - minZ) / rangeZ;
    // Cool-to-warm colour ramp
    colors[i * 3] = 0.2 + 0.6 * t;       // R
    colors[i * 3 + 1] = 0.3 + 0.4 * (1 - Math.abs(t - 0.5) * 2); // G
    colors[i * 3 + 2] = 0.9 - 0.7 * t;   // B
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, sizeAttenuation: true });
  const cloud = new THREE.Points(geometry, material);
  pointCloudGroup.add(cloud);

  // Fit camera to point cloud
  geometry.computeBoundingSphere();
  const { center, radius } = geometry.boundingSphere;
  controls.target.copy(center);
  camera.position.set(center.x + radius * 1.5, center.y + radius * 1.5, center.z + radius * 1.5);
  controls.update();
}

// ── Load parametric model (planes) ──────────────────────────────────
async function loadModel(uploadInfo) {
  const res = await fetch(`${API}/model`);
  const model = await res.json();

  // Clear previous planes
  while (wallGroup.children.length) wallGroup.remove(wallGroup.children[0]);
  while (floorGroup.children.length) floorGroup.remove(floorGroup.children[0]);
  while (ceilingGroup.children.length) ceilingGroup.remove(ceilingGroup.children[0]);

  let wallCount = 0, floorCount = 0, ceilingCount = 0;

  for (const plane of model.planes) {
    if (!plane.bounds) continue;
    const b = plane.bounds;
    const sx = b.max.x - b.min.x;
    const sy = b.max.y - b.min.y;
    const sz = b.max.z - b.min.z;
    const cx = (b.max.x + b.min.x) / 2;
    const cy = (b.max.y + b.min.y) / 2;
    const cz = (b.max.z + b.min.z) / 2;

    const color = PLANE_COLORS[plane.kind] || PLANE_COLORS.unknown;

    // Wireframe box showing plane extent
    const boxGeo = new THREE.BoxGeometry(Math.max(sx, 0.01), Math.max(sy, 0.01), Math.max(sz, 0.01));
    const edges = new THREE.EdgesGeometry(boxGeo);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color, linewidth: 2 }));
    line.position.set(cx, cy, cz);

    // Semi-transparent face fill
    const fillMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, side: THREE.DoubleSide });
    const fillMesh = new THREE.Mesh(boxGeo.clone(), fillMat);
    fillMesh.position.set(cx, cy, cz);

    const group = plane.kind === 'wall' ? wallGroup
                : plane.kind === 'floor' ? floorGroup
                : plane.kind === 'ceiling' ? ceilingGroup
                : planeGroup;

    group.add(line);
    group.add(fillMesh);

    if (plane.kind === 'wall') wallCount++;
    else if (plane.kind === 'floor') floorCount++;
    else if (plane.kind === 'ceiling') ceilingCount++;
  }

  // Update HUD
  document.getElementById('hud-file').textContent = model.source_file || uploadInfo?.filename || '—';
  document.getElementById('hud-points').textContent = (model.point_count || 0).toLocaleString();
  document.getElementById('hud-planes').textContent = model.planes.length;
  document.getElementById('hud-walls').textContent = wallCount;
  document.getElementById('hud-floor').textContent = floorCount;
  document.getElementById('hud-ceiling').textContent = ceilingCount;
}

// ── Toggle controls ─────────────────────────────────────────────────
document.getElementById('toggle-points').addEventListener('change', (e) => {
  pointCloudGroup.visible = e.target.checked;
});
document.getElementById('toggle-planes').addEventListener('change', (e) => {
  planeGroup.visible = e.target.checked;
});
document.getElementById('toggle-walls').addEventListener('change', (e) => {
  wallGroup.visible = e.target.checked;
});
document.getElementById('toggle-floor').addEventListener('change', (e) => {
  floorGroup.visible = e.target.checked;
});
document.getElementById('toggle-ceiling').addEventListener('change', (e) => {
  ceilingGroup.visible = e.target.checked;
});

document.getElementById('btn-reset').addEventListener('click', () => {
  const cloud = pointCloudGroup.children[0];
  if (cloud?.geometry?.boundingSphere) {
    const { center, radius } = cloud.geometry.boundingSphere;
    controls.target.copy(center);
    camera.position.set(center.x + radius * 1.5, center.y + radius * 1.5, center.z + radius * 1.5);
    controls.update();
  }
});

document.getElementById('btn-new').addEventListener('click', () => {
  overlay.classList.remove('hidden');
  document.getElementById('hud').style.display = 'none';
  document.getElementById('controls').style.display = 'none';
});
