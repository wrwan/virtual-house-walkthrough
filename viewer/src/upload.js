/**
 * Upload flow — drag-drop / file-input → upload → trigger scene loading.
 */

import { API } from './state.js';
import { loadPoints } from './pointcloud.js';
import { loadModel } from './model.js';

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

/** Re-show the upload overlay (called by "Upload New" button). */
export function showUploadOverlay() {
  overlay.classList.remove('hidden');
  document.getElementById('hud').style.display = 'none';
  document.getElementById('controls').style.display = 'none';
}
