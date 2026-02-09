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

// ── Manual wall corner picking ──────────────────────────────────────
const manualWallBtn = document.getElementById('btn-add-wall');
const manualWallStatus = document.getElementById('manual-wall-status');
let manualMode = false;
let pickedCorners = [];       // world-space Vec3[]
let cornerMarkers = [];       // Three.js meshes
let edgeLines = [];            // Three.js line segments
const cornerGroup = new THREE.Group();
cornerGroup.name = 'cornerMarkers';
scene.add(cornerGroup);

manualWallBtn.addEventListener('click', () => {
  if (manualMode) {
    exitManualMode();
  } else {
    enterManualMode();
  }
});

function enterManualMode() {
  manualMode = true;
  pickedCorners = [];
  clearCornerVisuals();
  manualWallBtn.classList.add('active-mode');
  manualWallBtn.textContent = 'Cancel';
  manualWallStatus.textContent = 'Click 4 points on the wall (0/4)';
  controls.enabled = true;  // orbit stays on; we pick on click events
  renderer.domElement.addEventListener('pointerdown', onManualPointerDown);
}

function exitManualMode() {
  manualMode = false;
  pickedCorners = [];
  clearCornerVisuals();
  manualWallBtn.classList.remove('active-mode');
  manualWallBtn.textContent = '+ Add Wall';
  manualWallStatus.textContent = '';
  renderer.domElement.removeEventListener('pointerdown', onManualPointerDown);
}

function clearCornerVisuals() {
  while (cornerGroup.children.length) cornerGroup.remove(cornerGroup.children[0]);
  cornerMarkers = [];
  edgeLines = [];
}

// Raycaster against the point cloud
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.05;  // hit tolerance in world units
const mouse = new THREE.Vector2();

let pointerDownPos = null; // track drag vs click

function onManualPointerDown(e) {
  if (!manualMode) return;
  // record start position to distinguish drag from click
  pointerDownPos = { x: e.clientX, y: e.clientY };
  renderer.domElement.addEventListener('pointerup', onManualPointerUp, { once: true });
}

function onManualPointerUp(e) {
  if (!manualMode || !pointerDownPos) return;
  // if user dragged more than 5px, treat as orbit, not pick
  const dx = e.clientX - pointerDownPos.x;
  const dy = e.clientY - pointerDownPos.y;
  if (Math.sqrt(dx * dx + dy * dy) > 5) return;
  if (e.button !== 0) return;  // left click only

  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  // Intersect against all Points objects in the rotated pointCloudGroup
  const intersections = raycaster.intersectObjects(pointCloudGroup.children, true);
  if (intersections.length === 0) {
    console.log('[MANUAL] No intersection found');
    return;
  }

  // The intersection point is in world space already
  const hit = intersections[0].point.clone();
  console.log(`[MANUAL] Picked corner ${pickedCorners.length + 1}: (${hit.x.toFixed(3)}, ${hit.y.toFixed(3)}, ${hit.z.toFixed(3)})`);

  pickedCorners.push(hit);
  addCornerMarker(hit);

  if (pickedCorners.length > 1) {
    addEdgeLine(pickedCorners[pickedCorners.length - 2], hit);
  }

  manualWallStatus.textContent = `Click 4 points on the wall (${pickedCorners.length}/4)`;

  if (pickedCorners.length >= 4) {
    // Close the loop visually
    addEdgeLine(pickedCorners[3], pickedCorners[0]);
    manualWallStatus.textContent = 'Refining wall against point cloud…';
    submitManualWall();
  }
}

function addCornerMarker(pos) {
  const geo = new THREE.SphereGeometry(0.04, 12, 12);
  const mat = new THREE.MeshBasicMaterial({ color: 0xe91e63 });
  const sphere = new THREE.Mesh(geo, mat);
  sphere.position.copy(pos);
  cornerGroup.add(sphere);
  cornerMarkers.push(sphere);
}

function addEdgeLine(a, b) {
  const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
  const mat = new THREE.LineBasicMaterial({ color: 0xe91e63, linewidth: 2 });
  const line = new THREE.Line(geo, mat);
  cornerGroup.add(line);
  edgeLines.push(line);
}

async function submitManualWall() {
  // Convert picked world-space corners back to the point-cloud's local
  // coordinate system (undo the -90° X rotation on pointCloudGroup).
  // pointCloudGroup.rotation.x = -PI/2  →  inverse is +PI/2
  const invMatrix = new THREE.Matrix4().copy(pointCloudGroup.matrixWorld).invert();
  const corners = pickedCorners.map(p => {
    const local = p.clone().applyMatrix4(invMatrix);
    return [local.x, local.y, local.z];
  });

  console.log('[MANUAL] Submitting corners to /api/manual-wall:', corners);

  try {
    const res = await fetch(`${API}/manual-wall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corners }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }

    const plane = await res.json();
    console.log('[MANUAL] Wall refined:', plane);

    // Render the new wall — server index is planes.length - 1
    const serverIdx = parseInt(document.getElementById('hud-planes').textContent) || 0;
    addPlaneToScene(plane, true, serverIdx);

    manualWallStatus.textContent = `Wall added! (${plane.inlier_count} inlier points)`;

    // Update HUD wall count
    const currentCount = parseInt(document.getElementById('hud-walls').textContent) || 0;
    document.getElementById('hud-walls').textContent = currentCount + 1;
    const planesCount = parseInt(document.getElementById('hud-planes').textContent) || 0;
    document.getElementById('hud-planes').textContent = planesCount + 1;
  } catch (err) {
    console.error('[MANUAL] Error:', err);
    manualWallStatus.textContent = `\u274c ${err.message}`;
  }

  // Clean up corner markers after a delay so user can see them
  setTimeout(() => {
    clearCornerVisuals();
    exitManualMode();
  }, 1500);
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

// ── Wall selection / deletion / trim ────────────────────────────────
const actionStatus = document.getElementById('action-status');
let selectedWallEntry = null;      // current wallObjects entry or null
let trimMode = false;
let trimFirstWall = null;         // wallObjects entry of the first pick

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

// Click on canvas to select/deselect walls (when NOT in manual-add or trim mode)
renderer.domElement.addEventListener('dblclick', (e) => {
  if (manualMode || trimMode) return;

  const entry = raycastWalls(e);

  // Deselect previous
  if (selectedWallEntry) {
    setWallHighlight(selectedWallEntry, false);
  }

  if (entry && entry !== selectedWallEntry) {
    selectedWallEntry = entry;
    setWallHighlight(entry, true);
    actionStatus.textContent = `Wall ${entry.index} selected — press Delete or click Trim`;
    actionStatus.style.color = '#ff1744';
  } else {
    selectedWallEntry = null;
    actionStatus.textContent = '';
    actionStatus.style.color = '#aaa';
  }
});

// Delete key removes selected wall
window.addEventListener('keydown', async (e) => {
  if (e.key === 'Delete' && selectedWallEntry && !manualMode && !trimMode) {
    const idx = selectedWallEntry.index;
    actionStatus.textContent = `Deleting wall ${idx}…`;

    try {
      const res = await fetch(`${API}/wall/${idx}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }

      // Remove from scene
      wallGroup.remove(selectedWallEntry.lineObj);
      wallGroup.remove(selectedWallEntry.fillObj);
      selectedWallEntry.lineObj.geometry.dispose();
      selectedWallEntry.fillObj.geometry.dispose();

      wallObjects = wallObjects.filter(w => w !== selectedWallEntry);

      // Re-index: entries after the deleted one shift down
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

      // Update HUD
      document.getElementById('hud-walls').textContent = wallObjects.length;
      document.getElementById('hud-planes').textContent = wallObjects.length;

      setTimeout(() => { actionStatus.textContent = ''; actionStatus.style.color = '#aaa'; }, 2000);
    } catch (err) {
      console.error('[DELETE]', err);
      actionStatus.textContent = `Error: ${err.message}`;
      actionStatus.style.color = '#ff1744';
    }
  }
});

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
