/**
 * Parametric model loading and plane visualisation.
 */

import * as THREE from 'three';
import {
  API, wallGroup, floorGroup, planeGroup,
  wallObjects, setWallObjects, PLANE_COLORS,
} from './state.js';

/** Fetch the parametric model and render all planes. */
export async function loadModel(uploadInfo) {
  console.log('[MODEL] Fetching parametric model from server...');
  const res = await fetch(`${API}/model`);
  const model = await res.json();
  console.log(`[MODEL] Received model with ${model.planes.length} planes`);

  // Clear previous planes
  while (wallGroup.children.length) wallGroup.remove(wallGroup.children[0]);
  while (floorGroup.children.length) floorGroup.remove(floorGroup.children[0]);

  let wallCount = 0;
  let floorCount = 0;
  setWallObjects([]);

  console.log('[MODEL] Building plane visualizations...');
  for (let planeIdx = 0; planeIdx < model.planes.length; planeIdx++) {
    const plane = model.planes[planeIdx];

    if (!plane.bounds) continue;
    addPlaneToScene(plane, false, planeIdx);

    if (plane.kind === 'wall') wallCount++;
    else if (plane.kind === 'floor') floorCount++;
  }

  console.log(`[MODEL] Created ${wallCount} walls, ${floorCount} floor`);

  // Update HUD
  document.getElementById('hud-file').textContent = model.source_file || uploadInfo?.filename || '—';
  document.getElementById('hud-points').textContent = (model.point_count || 0).toLocaleString();
  document.getElementById('hud-planes').textContent = model.planes.length;
  document.getElementById('hud-walls').textContent = wallCount;
  document.getElementById('hud-floor').textContent = floorCount;
  console.log('[MODEL] Model loaded successfully');
}

/** Add a single plane object to the 3D scene. */
export function addPlaneToScene(plane, isManual = false, serverIndex = -1) {
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

  const fillMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, side: THREE.DoubleSide });
  const fillMesh = new THREE.Mesh(boxGeo.clone(), fillMat);
  fillMesh.position.set(cx, cy, cz);

  // Tag with wall index for raycasting
  const idx = serverIndex >= 0 ? serverIndex : wallObjects.length;
  line.userData.wallIndex = idx;
  fillMesh.userData.wallIndex = idx;

  const group = plane.kind === 'wall' ? wallGroup
              : plane.kind === 'floor' ? floorGroup
              : planeGroup;

  group.add(line);
  group.add(fillMesh);

  wallObjects.push({ index: idx, lineObj: line, fillObj: fillMesh, plane });
}
