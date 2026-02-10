/**
 * Wall tools — manual wall drawing (4-corner pick + preview),
 *              edit mode (select + delete), and trim intersection.
 */

import * as THREE from 'three';
import {
  API, renderer, camera, controls,
  planeGroup, pointCloudGroup, wallGroup, cornerGroup, previewGroup,
  wallObjects, setWallObjects, raycaster, mouse, PLANE_COLORS,
} from './state.js';
import { addPlaneToScene } from './model.js';
import { actionStatus } from './controls.js';

const floatingLabels = document.getElementById('floating-labels');

// ── Shared helpers ──────────────────────────────────────────────────

/** Highlight a wall red (selected) or reset to orange. */
function setWallHighlight(entry, highlighted) {
  if (!entry) return;
  const color = highlighted ? 0xff1744 : PLANE_COLORS.wall;
  entry.lineObj.material.color.setHex(color);
  entry.fillObj.material.color.setHex(color);
  entry.fillObj.material.opacity = highlighted ? 0.3 : 0.12;
}

/** Find the wallObjects entry for a given wallIndex. */
function findWallEntry(wallIndex) {
  return wallObjects.find(w => w.index === wallIndex) || null;
}

/** Raycast against wall fill meshes. */
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

// =====================================================================
// MANUAL WALL DRAWING
// =====================================================================

const manualWallBtn = document.getElementById('btn-add-wall');
const manualWallStatus = document.getElementById('manual-wall-status');
const generateAllBtn = document.getElementById('btn-generate-all');

let manualMode = false;
let pickedCorners = [];
let cornerMarkers = [];
let edgeLines = [];

// Drawn wall previews awaiting generation
var drawnWalls = [];
let drawnWallId = 0;

manualWallBtn.addEventListener('click', () => {
  if (manualMode) exitManualMode(); else enterManualMode();
});

generateAllBtn.addEventListener('click', () => generateAllDrawnWalls());

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

// =====================================================================
// EDIT WALLS MODE
// =====================================================================

const editBtn = document.getElementById('btn-edit-walls');
let editMode = false;
let selectedWallEntry = null;
var editLabelDiv = null;

editBtn.addEventListener('click', () => {
  if (editMode) exitEditMode(); else enterEditMode();
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

    const filtered = wallObjects.filter(w => w !== selectedWallEntry);
    for (const w of filtered) {
      if (w.index > idx) {
        w.index--;
        w.lineObj.userData.wallIndex--;
        w.fillObj.userData.wallIndex--;
      }
    }
    setWallObjects(filtered);

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

// =====================================================================
// TRIM INTERSECTION MODE
// =====================================================================

const trimBtn = document.getElementById('btn-trim-wall');
let trimMode = false;
let trimFirstWall = null;

trimBtn.addEventListener('click', () => {
  if (trimMode) exitTrimMode(); else enterTrimMode();
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

    const entry = wallObjects.find(w => w.index === wallIndex);
    if (entry) {
      wallGroup.remove(entry.lineObj);
      wallGroup.remove(entry.fillObj);
      entry.lineObj.geometry.dispose();
      entry.fillObj.geometry.dispose();
      setWallObjects(wallObjects.filter(w => w !== entry));
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

// =====================================================================
// FLOATING LABEL UPDATES (called from animate loop)
// =====================================================================

/**
 * Update screen-space positions of floating HTML labels.
 * Called every frame from the animate loop in main.js.
 */
export function updateFloatingLabels() {
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
