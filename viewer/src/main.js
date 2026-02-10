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

// Drawn wall previews (declared early so animate loop can reference it)
var drawnWalls = [];
var editLabelDiv = null;

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateFloatingLabels();
  renderer.render(scene, camera);
}
animate();

function updateFloatingLabels() {
  for (const entry of drawnWalls) {
    const worldPos = entry.topLocal.clone();
    planeGroup.localToWorld(worldPos);
    worldPos.y += 0.15;

    const projected = worldPos.clone().project(camera);
    if (projected.z > 1) { entry.labelDiv.style.display = 'none'; continue; }

    entry.labelDiv.style.display = '';
    const x = (projected.x * 0.5 + 0.5) * window.innerWidth;
    const y = (-projected.y * 0.5 + 0.5) * window.innerHeight;
    entry.labelDiv.style.left = `${x}px`;
    entry.labelDiv.style.top = `${y}px`;
  }

  // Update edit-mode floating delete label
  if (editLabelDiv && editLabelDiv._wallEntry) {
    const we = editLabelDiv._wallEntry;
    if (!we.fillObj) { removeEditLabel(); return; }
    const b = we.plane.bounds;
    if (!b) { removeEditLabel(); return; }
    const topLocal = new THREE.Vector3(
      (b.max.x + b.min.x) / 2,
      (b.max.y + b.min.y) / 2,
      b.max.z
    );
    const worldPos = topLocal.clone();
    planeGroup.localToWorld(worldPos);
    worldPos.y += 0.15;

    const projected = worldPos.clone().project(camera);
    if (projected.z > 1) { editLabelDiv.style.display = 'none'; return; }

    editLabelDiv.style.display = '';
    const x = (projected.x * 0.5 + 0.5) * window.innerWidth;
    const y = (-projected.y * 0.5 + 0.5) * window.innerHeight;
    editLabelDiv.style.left = `${x}px`;
    editLabelDiv.style.top = `${y}px`;
  }
}

// ── Groups for toggling ─────────────────────────────────────────────
const pointCloudGroup = new THREE.Group();
pointCloudGroup.name = 'pointCloud';
// Rotate point cloud group 90 degrees around X-axis to make floor horizontal
pointCloudGroup.rotation.x = -Math.PI / 2;
scene.add(pointCloudGroup);

const planeGroup = new THREE.Group();
planeGroup.name = 'planes';
// Rotate plane group to match point cloud orientation
planeGroup.rotation.x = -Math.PI / 2;
scene.add(planeGroup);

// Per-kind sub-groups for individual toggles
const wallGroup = new THREE.Group(); wallGroup.name = 'walls';
// const floorGroup = new THREE.Group(); floorGroup.name = 'floor';
// const ceilingGroup = new THREE.Group(); ceilingGroup.name = 'ceiling';
// planeGroup.add(wallGroup, floorGroup, ceilingGroup);
planeGroup.add(wallGroup);

// Track wall objects for selection/deletion
// Each entry: { index, lineObj, fillObj, plane }
let wallObjects = [];

// ── Colours ─────────────────────────────────────────────────────────
const PLANE_COLORS = {
  floor: 0x4caf50,
  ceiling: 0x2196f3,
  wall: 0xff9800,
  manual: 0xe91e63,
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
  console.log(`[UPLOAD] Starting upload of ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
  statusEl.textContent = `Uploading ${file.name}…`;
  
  try {
    const form = new FormData();
    form.append('file', file);

    console.log('[UPLOAD] Sending file to server...');
    const uploadStart = Date.now();
    const res = await fetch(`${API}/upload`, { method: 'POST', body: form });
    const uploadTime = ((Date.now() - uploadStart) / 1000).toFixed(2);
    console.log(`[UPLOAD] Upload completed in ${uploadTime}s`);
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    
    console.log('[PROCESSING] Parsing response...');
    const info = await res.json();
    console.log(`[PROCESSING] Processing complete! ${info.point_count.toLocaleString()} points, ${info.planes_detected} planes detected`);
    statusEl.textContent = `Processing complete — ${info.point_count.toLocaleString()} points, ${info.planes_detected} planes`;

    // Load data into the scene
    console.log('[SCENE] Loading point cloud and model data...');
    const loadStart = Date.now();
    await Promise.all([loadPoints(), loadModel(info)]);
    const loadTime = ((Date.now() - loadStart) / 1000).toFixed(2);
    console.log(`[SCENE] Scene loaded in ${loadTime}s`);

    // Hide overlay, show HUD + controls
    overlay.classList.add('hidden');
    document.getElementById('hud').style.display = '';
    document.getElementById('controls').style.display = '';
    console.log('[COMPLETE] Viewer ready!');
  } catch (err) {
    console.error('[ERROR]', err);
    statusEl.textContent = `❌ ${err.message}`;
  }
}

// ── Load point cloud ────────────────────────────────────────────────
async function loadPoints() {
  console.log('[POINTS] Fetching point cloud data from server...');
  const res = await fetch(`${API}/points`);
  const data = await res.json();
  console.log(`[POINTS] Received ${data.count.toLocaleString()} points (has_colors: ${data.has_colors})`);
  
  const positions = new Float32Array(data.positions);

  // Clear previous
  while (pointCloudGroup.children.length) pointCloudGroup.remove(pointCloudGroup.children[0]);

  console.log('[POINTS] Building geometry...');
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const count = positions.length / 3;
  let colors;

  if (data.has_colors && data.colors) {
    // Use real RGB colors from the point cloud file
    console.log('[POINTS] Using RGB colors from file...');
    colors = new Float32Array(data.colors);
    console.log(`[POINTS] Applied ${count.toLocaleString()} RGB colors`);
  } else {
    // Fallback: colour each point by height (Z value) for visual depth
    console.log('[POINTS] No RGB data — computing height-based colors...');
    colors = new Float32Array(count * 3);
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const z = positions[i * 3 + 2];
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    const rangeZ = maxZ - minZ || 1;
    console.log(`[POINTS] Height range: ${minZ.toFixed(2)}m to ${maxZ.toFixed(2)}m`);
    
    for (let i = 0; i < count; i++) {
      const t = (positions[i * 3 + 2] - minZ) / rangeZ;
      // Cool-to-warm colour ramp
      colors[i * 3] = 0.2 + 0.6 * t;       // R
      colors[i * 3 + 1] = 0.3 + 0.4 * (1 - Math.abs(t - 0.5) * 2); // G
      colors[i * 3 + 2] = 0.9 - 0.7 * t;   // B
    }
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, sizeAttenuation: true });
  const cloud = new THREE.Points(geometry, material);
  pointCloudGroup.add(cloud);

  // Fit camera to point cloud
  console.log('[POINTS] Adjusting camera position...');
  geometry.computeBoundingSphere();
  const { center, radius } = geometry.boundingSphere;
  controls.target.copy(center);
  camera.position.set(center.x + radius * 1.5, center.y + radius * 1.5, center.z + radius * 1.5);
  controls.update();
  console.log('[POINTS] Point cloud loaded successfully');
}

// ── Load parametric model (planes) ──────────────────────────────────
async function loadModel(uploadInfo) {
  console.log('[MODEL] Fetching parametric model from server...');
  const res = await fetch(`${API}/model`);
  const model = await res.json();
  console.log(`[MODEL] Received model with ${model.planes.length} planes`);

  // Clear previous planes
  while (wallGroup.children.length) wallGroup.remove(wallGroup.children[0]);
  // while (floorGroup.children.length) floorGroup.remove(floorGroup.children[0]);
  // while (ceilingGroup.children.length) ceilingGroup.remove(ceilingGroup.children[0]);

  let wallCount = 0;
  // let floorCount = 0, ceilingCount = 0;
  wallObjects = [];

  console.log('[MODEL] Building plane visualizations (walls only)...');
  for (let planeIdx = 0; planeIdx < model.planes.length; planeIdx++) {
    const plane = model.planes[planeIdx];
    // Skip non-wall planes
    if (plane.kind !== 'wall') {
      console.log(`[MODEL] Skipping ${plane.kind} plane`);
      continue;
    }
    
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

    // Tag meshes with wall index for raycasting
    line.userData.wallIndex = planeIdx;
    fillMesh.userData.wallIndex = planeIdx;

    const group = plane.kind === 'wall' ? wallGroup
                : planeGroup;

    group.add(line);
    group.add(fillMesh);

    wallObjects.push({ index: planeIdx, lineObj: line, fillObj: fillMesh, plane });

    if (plane.kind === 'wall') wallCount++;
  }

  console.log(`[MODEL] Created ${wallCount} walls (floor/ceiling disabled)`);

  // Update HUD
  document.getElementById('hud-file').textContent = model.source_file || uploadInfo?.filename || '—';
  document.getElementById('hud-points').textContent = (model.point_count || 0).toLocaleString();
  document.getElementById('hud-planes').textContent = model.planes.length;
  document.getElementById('hud-walls').textContent = wallCount;
  document.getElementById('hud-floor').textContent = '—';
  document.getElementById('hud-ceiling').textContent = '—';
  console.log('[MODEL] Model loaded successfully');
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
// document.getElementById('toggle-floor').addEventListener('change', (e) => {
//   floorGroup.visible = e.target.checked;
// });
// document.getElementById('toggle-ceiling').addEventListener('change', (e) => {
//   ceilingGroup.visible = e.target.checked;
// });

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

// ── Manual wall drawing mode ────────────────────────────────────────
const manualWallBtn = document.getElementById('btn-add-wall');
const manualWallStatus = document.getElementById('manual-wall-status');
const floatingLabels = document.getElementById('floating-labels');
const generateAllBtn = document.getElementById('btn-generate-all');

let manualMode = false;
let pickedCorners = [];       // world-space Vec3[] for current pick
let cornerMarkers = [];       // Three.js meshes for current pick
let edgeLines = [];            // Three.js line segments for current pick
const cornerGroup = new THREE.Group();
cornerGroup.name = 'cornerMarkers';
scene.add(cornerGroup);

// Drawn wall previews awaiting generation
// (drawnWalls declared early near animate loop)
let drawnWallId = 0;
const previewGroup = new THREE.Group();
previewGroup.name = 'wallPreviews';
planeGroup.add(previewGroup);

manualWallBtn.addEventListener('click', () => {
  if (manualMode) {
    exitManualMode();
  } else {
    enterManualMode();
  }
});

generateAllBtn.addEventListener('click', () => {
  generateAllDrawnWalls();
});

function enterManualMode() {
  manualMode = true;
  pickedCorners = [];
  clearCurrentCornerVisuals();
  manualWallBtn.classList.add('active-mode');
  manualWallBtn.textContent = 'Cancel Drawing';
  manualWallStatus.textContent = 'Click 4 points on the wall (0/4)';
  controls.enabled = true;
  renderer.domElement.addEventListener('pointerdown', onManualPointerDown);
}

function exitManualMode() {
  manualMode = false;
  pickedCorners = [];
  clearCurrentCornerVisuals();
  manualWallBtn.classList.remove('active-mode');
  manualWallBtn.textContent = '+ Add Wall';
  if (drawnWalls.length > 0) {
    manualWallStatus.textContent = `${drawnWalls.length} wall(s) drawn — Generate or click + Add Wall to draw more.`;
  } else {
    manualWallStatus.textContent = '';
  }
  renderer.domElement.removeEventListener('pointerdown', onManualPointerDown);
}

function clearCurrentCornerVisuals() {
  while (cornerGroup.children.length) cornerGroup.remove(cornerGroup.children[0]);
  cornerMarkers = [];
  edgeLines = [];
}

// Raycaster against the point cloud
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.05;
const mouse = new THREE.Vector2();

let pointerDownPos = null;

function onManualPointerDown(e) {
  if (!manualMode) return;
  pointerDownPos = { x: e.clientX, y: e.clientY };
  renderer.domElement.addEventListener('pointerup', onManualPointerUp, { once: true });
}

function onManualPointerUp(e) {
  if (!manualMode || !pointerDownPos) return;
  const dx = e.clientX - pointerDownPos.x;
  const dy = e.clientY - pointerDownPos.y;
  if (Math.sqrt(dx * dx + dy * dy) > 5) return;
  if (e.button !== 0) return;

  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const intersections = raycaster.intersectObjects(pointCloudGroup.children, true);
  if (intersections.length === 0) return;

  const hit = intersections[0].point.clone();
  console.log(`[DRAW] Corner ${pickedCorners.length + 1}: (${hit.x.toFixed(3)}, ${hit.y.toFixed(3)}, ${hit.z.toFixed(3)})`);

  pickedCorners.push(hit);
  addCornerMarker(hit);

  if (pickedCorners.length > 1) {
    addEdgeLine(pickedCorners[pickedCorners.length - 2], hit);
  }

  manualWallStatus.textContent = `Click 4 points on the wall (${pickedCorners.length}/4)`;

  if (pickedCorners.length >= 4) {
    addEdgeLine(pickedCorners[3], pickedCorners[0]);
    createWallPreview([...pickedCorners]);
    pickedCorners = [];
    clearCurrentCornerVisuals();
    manualWallStatus.textContent = `${drawnWalls.length} wall(s) drawn — pick 4 more or Generate.`;
  }
}

function addCornerMarker(pos) {
  const geo = new THREE.SphereGeometry(0.04, 12, 12);
  const mat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
  const sphere = new THREE.Mesh(geo, mat);
  sphere.position.copy(pos);
  cornerGroup.add(sphere);
  cornerMarkers.push(sphere);
}

function addEdgeLine(a, b) {
  const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
  const mat = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 2 });
  const line = new THREE.Line(geo, mat);
  cornerGroup.add(line);
  edgeLines.push(line);
}

// ── Wall preview management ─────────────────────────────────────────

function createWallPreview(worldCorners) {
  const invMatrix = new THREE.Matrix4().copy(planeGroup.matrixWorld).invert();
  const localCorners = worldCorners.map(p => p.clone().applyMatrix4(invMatrix));

  const min = localCorners[0].clone();
  const max = localCorners[0].clone();
  for (const c of localCorners) { min.min(c); max.max(c); }

  const sx = max.x - min.x;
  const sy = max.y - min.y;
  const sz = max.z - min.z;
  const cx = (max.x + min.x) / 2;
  const cy = (max.y + min.y) / 2;
  const cz = (max.z + min.z) / 2;

  const grp = new THREE.Group();

  // Dashed wireframe
  const boxGeo = new THREE.BoxGeometry(Math.max(sx, 0.01), Math.max(sy, 0.01), Math.max(sz, 0.01));
  const edges = new THREE.EdgesGeometry(boxGeo);
  const lineMat = new THREE.LineDashedMaterial({ color: 0x00e5ff, dashSize: 0.1, gapSize: 0.05, linewidth: 2 });
  const lineObj = new THREE.LineSegments(edges, lineMat);
  lineObj.computeLineDistances();
  lineObj.position.set(cx, cy, cz);
  grp.add(lineObj);

  // Transparent fill
  const fillMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
  const fillObj = new THREE.Mesh(boxGeo.clone(), fillMat);
  fillObj.position.set(cx, cy, cz);
  grp.add(fillObj);

  // Corner spheres
  for (const c of localCorners) {
    const geo = new THREE.SphereGeometry(0.04, 8, 8);
    const s = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
    s.position.copy(c);
    grp.add(s);
  }
  // Edge lines
  for (let i = 0; i < localCorners.length; i++) {
    const a = localCorners[i], b = localCorners[(i + 1) % localCorners.length];
    const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
    grp.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x00e5ff })));
  }

  previewGroup.add(grp);

  // Corners for API submission (in pointCloudGroup local space)
  const pcInv = new THREE.Matrix4().copy(pointCloudGroup.matrixWorld).invert();
  const apiCorners = worldCorners.map(p => {
    const local = p.clone().applyMatrix4(pcInv);
    return [local.x, local.y, local.z];
  });

  // Floating HTML label
  const id = drawnWallId++;
  const labelDiv = document.createElement('div');
  labelDiv.className = 'wall-label';
  labelDiv.innerHTML = `
    <button class="btn-gen" data-id="${id}">Generate</button>
    <button class="btn-del" data-id="${id}">Delete</button>
  `;
  labelDiv.querySelector('.btn-gen').addEventListener('click', () => generateDrawnWall(id));
  labelDiv.querySelector('.btn-del').addEventListener('click', () => deleteDrawnWall(id));
  floatingLabels.appendChild(labelDiv);

  drawnWalls.push({
    id,
    corners: apiCorners,
    previewGrp: grp,
    labelDiv,
    topLocal: new THREE.Vector3(cx, cy, max.z),
  });

  generateAllBtn.style.display = '';
}

function deleteDrawnWall(id) {
  const idx = drawnWalls.findIndex(w => w.id === id);
  if (idx === -1) return;
  const entry = drawnWalls[idx];

  previewGroup.remove(entry.previewGrp);
  entry.previewGrp.traverse(child => { if (child.geometry) child.geometry.dispose(); });
  floatingLabels.removeChild(entry.labelDiv);

  drawnWalls.splice(idx, 1);
  generateAllBtn.style.display = drawnWalls.length > 0 ? '' : 'none';
  manualWallStatus.textContent = drawnWalls.length > 0
    ? `${drawnWalls.length} wall(s) drawn.`
    : (manualMode ? 'Click 4 points on the wall (0/4)' : '');
}

async function generateDrawnWall(id) {
  const idx = drawnWalls.findIndex(w => w.id === id);
  if (idx === -1) return;
  const entry = drawnWalls[idx];

  const genBtn = entry.labelDiv.querySelector('.btn-gen');
  const delBtn = entry.labelDiv.querySelector('.btn-del');
  genBtn.disabled = true;  genBtn.textContent = '…';
  delBtn.disabled = true;

  try {
    const res = await fetch(`${API}/manual-wall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corners: entry.corners }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    const plane = await res.json();
    console.log('[GENERATE] Wall created:', plane);

    const serverIdx = parseInt(document.getElementById('hud-planes').textContent) || 0;
    addPlaneToScene(plane, true, serverIdx);

    const wc = parseInt(document.getElementById('hud-walls').textContent) || 0;
    document.getElementById('hud-walls').textContent = wc + 1;
    const pc = parseInt(document.getElementById('hud-planes').textContent) || 0;
    document.getElementById('hud-planes').textContent = pc + 1;

    deleteDrawnWall(id);
    manualWallStatus.textContent = `Wall generated! (${plane.inlier_count} inliers). ${drawnWalls.length} drawn.`;
  } catch (err) {
    console.error('[GENERATE]', err);
    genBtn.disabled = false;  genBtn.textContent = 'Generate';
    delBtn.disabled = false;
    manualWallStatus.textContent = `Error: ${err.message}`;
  }
}

async function generateAllDrawnWalls() {
  if (drawnWalls.length === 0) return;
  generateAllBtn.disabled = true;
  generateAllBtn.textContent = 'Generating…';
  const total = drawnWalls.length;
  manualWallStatus.textContent = `Generating ${total} wall(s)…`;

  const ids = drawnWalls.map(w => w.id);
  let success = 0, failed = 0;

  for (const id of ids) {
    try {
      await generateDrawnWall(id);
      success++;
      manualWallStatus.textContent = `Generating… (${success}/${total})`;
    } catch { failed++; }
  }

  generateAllBtn.disabled = false;
  generateAllBtn.textContent = 'Generate All Drawn';
  generateAllBtn.style.display = drawnWalls.length > 0 ? '' : 'none';
  manualWallStatus.textContent = `Generated ${success} wall(s)${failed ? `, ${failed} failed` : ''}.`;
}

/** Add a single plane object to the 3D scene. */
function addPlaneToScene(plane, isManual = false, serverIndex = -1) {
  if (!plane.bounds) return;
  const b = plane.bounds;
  const sx = b.max.x - b.min.x;
  const sy = b.max.y - b.min.y;
  const sz = b.max.z - b.min.z;
  const cx = (b.max.x + b.min.x) / 2;
  const cy = (b.max.y + b.min.y) / 2;
  const cz = (b.max.z + b.min.z) / 2;

  const color = isManual ? PLANE_COLORS.manual : (PLANE_COLORS[plane.kind] || PLANE_COLORS.unknown);

  const boxGeo = new THREE.BoxGeometry(Math.max(sx, 0.01), Math.max(sy, 0.01), Math.max(sz, 0.01));
  const edges = new THREE.EdgesGeometry(boxGeo);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color, linewidth: 2 }));
  line.position.set(cx, cy, cz);

  const fillMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
  const fillMesh = new THREE.Mesh(boxGeo.clone(), fillMat);
  fillMesh.position.set(cx, cy, cz);

  // Tag with server index
  const idx = serverIndex >= 0 ? serverIndex : wallObjects.length;
  line.userData.wallIndex = idx;
  fillMesh.userData.wallIndex = idx;

  wallGroup.add(line);
  wallGroup.add(fillMesh);

  wallObjects.push({ index: idx, lineObj: line, fillObj: fillMesh, plane });
}

// ── Wall selection / deletion / edit mode ───────────────────────────
const actionStatus = document.getElementById('action-status');
let selectedWallEntry = null;      // current wallObjects entry or null
let trimMode = false;
let trimFirstWall = null;         // wallObjects entry of the first pick
let editMode = false;

/** Highlight a wall red (selected) or reset to orange. */
function setWallHighlight(entry, highlighted) {
  if (!entry) return;
  const color = highlighted ? 0xff1744 : (PLANE_COLORS.wall);
  entry.lineObj.material.color.setHex(color);
  entry.fillObj.material.color.setHex(color);
  entry.fillObj.material.opacity = highlighted ? 0.3 : 0.12;
}

/** Find the wallObjects entry for a given wallIndex. */
function findWallEntry(wallIndex) {
  return wallObjects.find(w => w.index === wallIndex) || null;
}

/** Raycast against wall fill meshes (not point cloud). */
function raycastWalls(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const fillMeshes = wallObjects.map(w => w.fillObj);
  const hits = raycaster.intersectObjects(fillMeshes, false);
  if (hits.length === 0) return null;
  const wallIdx = hits[0].object.userData.wallIndex;
  return findWallEntry(wallIdx);
}

// ── Edit walls mode ─────────────────────────────────────────────────
const editBtn = document.getElementById('btn-edit-walls');

editBtn.addEventListener('click', () => {
  if (editMode) {
    exitEditMode();
  } else {
    enterEditMode();
  }
});

function enterEditMode() {
  if (manualMode || trimMode) return;
  editMode = true;
  editBtn.classList.add('active-mode');
  editBtn.textContent = 'Done Editing';
  actionStatus.textContent = 'Click any wall to select it for deletion.';
  actionStatus.style.color = '#ff1744';
  renderer.domElement.addEventListener('pointerdown', onEditPointerDown);
}

function exitEditMode() {
  editMode = false;
  if (selectedWallEntry) { setWallHighlight(selectedWallEntry, false); selectedWallEntry = null; }
  removeEditLabel();
  editBtn.classList.remove('active-mode');
  editBtn.textContent = 'Edit Walls';
  actionStatus.textContent = '';
  actionStatus.style.color = '#aaa';
  renderer.domElement.removeEventListener('pointerdown', onEditPointerDown);
}

let editPointerDown = null;

function onEditPointerDown(e) {
  if (!editMode) return;
  editPointerDown = { x: e.clientX, y: e.clientY };
  renderer.domElement.addEventListener('pointerup', onEditPointerUp, { once: true });
}

function onEditPointerUp(e) {
  if (!editMode || !editPointerDown) return;
  const dx = e.clientX - editPointerDown.x;
  const dy = e.clientY - editPointerDown.y;
  if (Math.sqrt(dx * dx + dy * dy) > 5) return;
  if (e.button !== 0) return;

  const entry = raycastWalls(e);

  // Deselect previous
  if (selectedWallEntry) {
    setWallHighlight(selectedWallEntry, false);
    removeEditLabel();
  }

  if (entry && entry !== selectedWallEntry) {
    selectedWallEntry = entry;
    setWallHighlight(entry, true);
    showEditLabel(entry);
    actionStatus.textContent = `Wall ${entry.index} selected.`;
    actionStatus.style.color = '#ff1744';
  } else {
    selectedWallEntry = null;
    actionStatus.textContent = 'Click any wall to select it.';
    actionStatus.style.color = '#ff1744';
  }
}

function showEditLabel(entry) {
  removeEditLabel();
  const div = document.createElement('div');
  div.className = 'wall-edit-label';
  div.innerHTML = `<button>Delete Wall</button>`;
  div.querySelector('button').addEventListener('click', () => deleteSelectedWall());
  floatingLabels.appendChild(div);
  editLabelDiv = div;
  // Store info needed for positioning
  editLabelDiv._wallEntry = entry;
}

function removeEditLabel() {
  if (editLabelDiv) {
    floatingLabels.removeChild(editLabelDiv);
    editLabelDiv = null;
  }
}

async function deleteSelectedWall() {
  if (!selectedWallEntry) return;
  const idx = selectedWallEntry.index;
  actionStatus.textContent = `Deleting wall ${idx}…`;
  removeEditLabel();

  try {
    const res = await fetch(`${API}/wall/${idx}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    wallGroup.remove(selectedWallEntry.lineObj);
    wallGroup.remove(selectedWallEntry.fillObj);
    selectedWallEntry.lineObj.geometry.dispose();
    selectedWallEntry.fillObj.geometry.dispose();

    wallObjects = wallObjects.filter(w => w !== selectedWallEntry);
    for (const w of wallObjects) {
      if (w.index > idx) {
        w.index--;
        w.lineObj.userData.wallIndex--;
        w.fillObj.userData.wallIndex--;
      }
    }

    selectedWallEntry = null;
    actionStatus.textContent = `Wall ${idx} deleted`;
    actionStatus.style.color = '#4caf50';
    document.getElementById('hud-walls').textContent = wallObjects.length;
    document.getElementById('hud-planes').textContent = wallObjects.length;

    setTimeout(() => {
      if (editMode) {
        actionStatus.textContent = 'Click any wall to select it.';
        actionStatus.style.color = '#ff1744';
      }
    }, 1500);
  } catch (err) {
    console.error('[DELETE]', err);
    actionStatus.textContent = `Error: ${err.message}`;
    actionStatus.style.color = '#ff1744';
  }
}

// ── Trim intersection mode ──────────────────────────────────────────
const trimBtn = document.getElementById('btn-trim-wall');

trimBtn.addEventListener('click', () => {
  if (trimMode) {
    exitTrimMode();
  } else {
    enterTrimMode();
  }
});

function enterTrimMode() {
  if (manualMode) return;
  trimMode = true;
  trimFirstWall = null;
  trimBtn.classList.add('active-mode');
  trimBtn.textContent = 'Cancel Trim';
  actionStatus.textContent = 'Click the wall to trim (1/2)';
  actionStatus.style.color = '#e65100';
  renderer.domElement.addEventListener('pointerdown', onTrimPointerDown);
}

function exitTrimMode() {
  trimMode = false;
  if (trimFirstWall) { setWallHighlight(trimFirstWall, false); trimFirstWall = null; }
  if (selectedWallEntry) { setWallHighlight(selectedWallEntry, false); selectedWallEntry = null; }
  trimBtn.classList.remove('active-mode');
  trimBtn.textContent = 'Trim Intersection';
  actionStatus.textContent = '';
  actionStatus.style.color = '#aaa';
  renderer.domElement.removeEventListener('pointerdown', onTrimPointerDown);
}

let trimPointerDown = null;

function onTrimPointerDown(e) {
  if (!trimMode) return;
  trimPointerDown = { x: e.clientX, y: e.clientY };
  renderer.domElement.addEventListener('pointerup', onTrimPointerUp, { once: true });
}

function onTrimPointerUp(e) {
  if (!trimMode || !trimPointerDown) return;
  const dx = e.clientX - trimPointerDown.x;
  const dy = e.clientY - trimPointerDown.y;
  if (Math.sqrt(dx * dx + dy * dy) > 5) return;
  if (e.button !== 0) return;

  const entry = raycastWalls(e);
  if (!entry) return;

  if (!trimFirstWall) {
    trimFirstWall = entry;
    setWallHighlight(entry, true);
    actionStatus.textContent = `Wall ${entry.index} selected — now click the clipper wall (2/2)`;
  } else if (entry.index !== trimFirstWall.index) {
    // We have both walls — submit trim
    actionStatus.textContent = 'Trimming…';
    submitTrim(trimFirstWall.index, entry.index);
  }
}

async function submitTrim(wallIndex, clipperIndex) {
  try {
    const res = await fetch(`${API}/trim-wall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wall_index: wallIndex, clipper_index: clipperIndex }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    const trimmedPlane = await res.json();
    console.log('[TRIM] Wall trimmed:', trimmedPlane);

    // Update the visual: remove old, add new
    const entry = wallObjects.find(w => w.index === wallIndex);
    if (entry) {
      wallGroup.remove(entry.lineObj);
      wallGroup.remove(entry.fillObj);
      entry.lineObj.geometry.dispose();
      entry.fillObj.geometry.dispose();
      wallObjects = wallObjects.filter(w => w !== entry);
    }

    addPlaneToScene(trimmedPlane, false, wallIndex);
    actionStatus.textContent = `Wall ${wallIndex} trimmed at wall ${clipperIndex}`;
    actionStatus.style.color = '#4caf50';
  } catch (err) {
    console.error('[TRIM]', err);
    actionStatus.textContent = `Error: ${err.message}`;
    actionStatus.style.color = '#ff1744';
  }

  setTimeout(() => exitTrimMode(), 1500);
}

// ── Normalize walls ─────────────────────────────────────────────────
const normalizeBtn = document.getElementById('btn-normalize');

normalizeBtn.addEventListener('click', async () => {
  if (manualMode || trimMode) return;

  normalizeBtn.disabled = true;
  actionStatus.textContent = 'Normalizing walls…';
  actionStatus.style.color = '#6c63ff';

  try {
    const res = await fetch(`${API}/normalize-walls`, { method: 'POST' });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    const model = await res.json();
    console.log('[NORMALIZE] Model updated:', model.planes.length, 'planes');

    // Full reload of wall visuals
    while (wallGroup.children.length) wallGroup.remove(wallGroup.children[0]);
    wallObjects = [];

    let wallCount = 0;
    for (let i = 0; i < model.planes.length; i++) {
      const plane = model.planes[i];
      if (plane.kind !== 'wall') continue;
      addPlaneToScene(plane, false, i);
      wallCount++;
    }

    document.getElementById('hud-walls').textContent = wallCount;
    document.getElementById('hud-planes').textContent = model.planes.length;

    actionStatus.textContent = `Walls normalized (${wallCount} walls)`;
    actionStatus.style.color = '#4caf50';
  } catch (err) {
    console.error('[NORMALIZE]', err);
    actionStatus.textContent = `Error: ${err.message}`;
    actionStatus.style.color = '#ff1744';
  }

  normalizeBtn.disabled = false;
  setTimeout(() => { actionStatus.textContent = ''; actionStatus.style.color = '#aaa'; }, 3000);
});
