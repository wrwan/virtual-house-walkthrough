/**
 * UI controls — toggles, reset camera, upload new, and normalize walls.
 */

import {
  API, pointCloudGroup, planeGroup, wallGroup, floorGroup,
  wallObjects, setWallObjects, camera, controls,
} from './state.js';
import { addPlaneToScene } from './model.js';
import { showUploadOverlay } from './upload.js';

export const actionStatus = document.getElementById('action-status');

// ── Visibility toggles ──────────────────────────────────────────────
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

// ── Reset camera ────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', () => {
  const cloud = pointCloudGroup.children[0];
  if (cloud?.geometry?.boundingSphere) {
    const { center, radius } = cloud.geometry.boundingSphere;
    controls.target.copy(center);
    camera.position.set(center.x + radius * 1.5, center.y + radius * 1.5, center.z + radius * 1.5);
    controls.update();
  }
});

// ── Upload new ──────────────────────────────────────────────────────
document.getElementById('btn-new').addEventListener('click', () => showUploadOverlay());

// ── Normalize walls ─────────────────────────────────────────────────
const normalizeBtn = document.getElementById('btn-normalize');

normalizeBtn.addEventListener('click', async () => {
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
    setWallObjects([]);

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
