/**
 * Walkthrough mode — first-person exploration of the scanned space.
 *
 * Activate via the "Walkthrough" button → click on a floor surface to
 * teleport there → WASD + mouse-look for navigation.
 * Eye height adjustable via the side slider or scroll-wheel.
 */

import * as THREE from 'three';
import {
  renderer, camera, controls,
  floorGroup, pointCloudGroup,
} from './state.js';

// ── Configuration ───────────────────────────────────────────────────
const MOVE_SPEED      = 2.5;       // m/s walking speed
const ACCEL           = 10;        // acceleration factor
const DECEL           = 8;         // deceleration factor
const MOUSE_SENS      = 0.0018;    // radians per pixel
const MIN_PITCH       = -Math.PI / 3;
const MAX_PITCH       = Math.PI / 3;
const TRANSITION_SECS = 0.7;       // teleport animation duration
const DEFAULT_HEIGHT  = 1.65;      // metres above floor

// ── Internal state ──────────────────────────────────────────────────
let active        = false;   // walk-mode menu entered
let walking       = false;   // in first-person locomotion
let pointerLocked = false;
let eyeHeight     = DEFAULT_HEIGHT;
let floorY        = 0;

// Saved orbit camera (restored on exit)
const savedPos    = new THREE.Vector3();
const savedTarget = new THREE.Vector3();

// Smooth teleport
let transitioning = false;
let tProg         = 0;
const tFrom = new THREE.Vector3();
const tTo   = new THREE.Vector3();

// First-person look (YXZ → yaw then pitch)
const euler = new THREE.Euler(0, 0, 0, 'YXZ');

// Movement
const vel      = new THREE.Vector3();
const keysDown = new Set();

// Raycast (private — avoids touching the shared raycaster)
const ray = new THREE.Raycaster();
ray.params.Points.threshold = 0.1;

// ── DOM refs ────────────────────────────────────────────────────────
const walkBtn      = document.getElementById('btn-walkthrough');
const overlayEl    = document.getElementById('walk-overlay');
const crosshairEl  = document.getElementById('walk-crosshair');
const heightPanel  = document.getElementById('walk-height-panel');
const heightSlider = document.getElementById('walk-height');
const heightValEl  = document.getElementById('walk-height-value');
const hintEl       = document.getElementById('walk-hint');
const exitBtnEl    = document.getElementById('walk-exit');

// ── Public API (called from main.js animate loop) ───────────────────

/** True when first-person locomotion is active (controls.update should be skipped). */
export function isWalking() { return walking; }

/** Per-frame update — handles transition animation + WASD movement. */
export function updateWalk(delta) {
  if (!active) return;

  // ── Teleport transition ──────────────────────────────────────
  if (transitioning) {
    tProg += delta / TRANSITION_SECS;
    if (tProg >= 1) {
      tProg = 1;
      transitioning = false;
      camera.position.copy(tTo);
      hint('move');
    }
    const t = 1 - Math.pow(1 - Math.min(tProg, 1), 3); // ease-out cubic
    camera.position.lerpVectors(tFrom, tTo, t);
    camera.quaternion.setFromEuler(euler);
    return;
  }

  if (!walking) return;

  // ── Apply look angle ─────────────────────────────────────────
  camera.quaternion.setFromEuler(euler);

  // ── Compute XZ movement ──────────────────────────────────────
  const fwd = new THREE.Vector3();
  camera.getWorldDirection(fwd);
  fwd.y = 0;
  fwd.normalize();

  const right = new THREE.Vector3()
    .crossVectors(fwd, new THREE.Vector3(0, 1, 0))
    .normalize();

  const desired = new THREE.Vector3();
  if (keysDown.has('KeyW') || keysDown.has('ArrowUp'))    desired.add(fwd);
  if (keysDown.has('KeyS') || keysDown.has('ArrowDown'))  desired.sub(fwd);
  if (keysDown.has('KeyD') || keysDown.has('ArrowRight')) desired.add(right);
  if (keysDown.has('KeyA') || keysDown.has('ArrowLeft'))  desired.sub(right);

  if (desired.lengthSq() > 0) {
    desired.normalize().multiplyScalar(MOVE_SPEED);
    vel.lerp(desired, 1 - Math.exp(-ACCEL * delta));
  } else {
    vel.multiplyScalar(Math.exp(-DECEL * delta));
  }

  if (vel.lengthSq() < 1e-4) vel.set(0, 0, 0);

  camera.position.addScaledVector(vel, delta);
  camera.position.y = floorY + eyeHeight;
}

// ── Enter / exit ────────────────────────────────────────────────────

function enter() {
  // Block if a wall tool is active
  const toolIds = ['btn-add-wall', 'btn-edit-walls', 'btn-trim-wall'];
  if (toolIds.some(id => document.getElementById(id)?.classList.contains('active-mode'))) return;

  active  = true;
  walking = false;

  savedPos.copy(camera.position);
  savedTarget.copy(controls.target);

  walkBtn.textContent = 'Exit Walkthrough';
  walkBtn.classList.add('active-mode');
  overlayEl.classList.remove('hidden');
  crosshairEl.style.display  = 'none';
  heightPanel.style.display  = 'none';
  exitBtnEl.style.display    = 'none';
  hint('pick');

  heightSlider.value = eyeHeight;
  heightValEl.textContent = `${eyeHeight.toFixed(2)} m`;

  renderer.domElement.addEventListener('click', onCanvasClick);
  document.addEventListener('pointerlockchange', onLockChange);
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', onKeyUp);
  renderer.domElement.addEventListener('wheel', onWheel, { passive: false });
}

function exit() {
  active        = false;
  walking       = false;
  transitioning = false;
  vel.set(0, 0, 0);
  keysDown.clear();

  // Restore orbit camera
  camera.position.copy(savedPos);
  controls.target.copy(savedTarget);
  controls.enabled = true;
  controls.update();

  if (document.pointerLockElement) document.exitPointerLock();

  walkBtn.textContent = 'Walkthrough';
  walkBtn.classList.remove('active-mode');
  overlayEl.classList.add('hidden');

  renderer.domElement.removeEventListener('click', onCanvasClick);
  document.removeEventListener('pointerlockchange', onLockChange);
  document.removeEventListener('keydown', onKeyDown);
  document.removeEventListener('keyup', onKeyUp);
  renderer.domElement.removeEventListener('wheel', onWheel);
  document.removeEventListener('mousemove', onMouseLook);
}

// ── Button handlers ─────────────────────────────────────────────────
walkBtn.addEventListener('click', () => { active ? exit() : enter(); });
exitBtnEl.addEventListener('click', () => exit());

// ── Canvas click (floor pick / resume) ──────────────────────────────

function onCanvasClick(e) {
  if (pointerLocked || transitioning) return;

  const pointer = new THREE.Vector2(
    (e.clientX / window.innerWidth)  *  2 - 1,
    -(e.clientY / window.innerHeight) * 2 + 1,
  );
  ray.setFromCamera(pointer, camera);

  // Try floor fill-meshes first
  const meshes = [];
  floorGroup.traverse(c => { if (c.isMesh) meshes.push(c); });

  let hit = null;
  const fHits = ray.intersectObjects(meshes, false);
  if (fHits.length) hit = fHits[0].point;

  // Fallback: point cloud
  if (!hit) {
    const pHits = ray.intersectObjects(pointCloudGroup.children, true);
    if (pHits.length) hit = pHits[0].point;
  }

  if (hit) {
    teleportTo(hit);
  } else if (walking) {
    // No surface hit — just re-lock to resume walking
    Promise.resolve(renderer.domElement.requestPointerLock()).catch(() => {});
  }
}

function teleportTo(worldPt) {
  floorY = worldPt.y;
  tFrom.copy(camera.position);
  tTo.set(worldPt.x, worldPt.y + eyeHeight, worldPt.z);
  tProg = 0;
  transitioning = true;
  walking = true;
  controls.enabled = false;

  // Preserve current look direction
  euler.setFromQuaternion(camera.quaternion, 'YXZ');

  crosshairEl.style.display = '';
  heightPanel.style.display = '';
  hint('transitioning');

  // Must request pointer-lock inside click handler (user gesture)
  Promise.resolve(renderer.domElement.requestPointerLock()).catch(() => {});
}

// ── Pointer-lock lifecycle ──────────────────────────────────────────

function onLockChange() {
  pointerLocked = document.pointerLockElement === renderer.domElement;

  if (pointerLocked) {
    document.addEventListener('mousemove', onMouseLook);
    crosshairEl.style.display = '';
    exitBtnEl.style.display   = 'none';
    if (!transitioning) hint('move');
  } else {
    document.removeEventListener('mousemove', onMouseLook);
    keysDown.clear();
    vel.set(0, 0, 0);
    crosshairEl.style.display = 'none';
    if (walking) {
      exitBtnEl.style.display = '';
      hint('resume');
    }
  }
}

function onMouseLook(e) {
  if (!pointerLocked || !walking || transitioning) return;
  euler.y -= e.movementX * MOUSE_SENS;
  euler.x -= e.movementY * MOUSE_SENS;
  euler.x = Math.max(MIN_PITCH, Math.min(MAX_PITCH, euler.x));
}

// ── Keyboard ────────────────────────────────────────────────────────

function onKeyDown(e) {
  if (!active) return;
  keysDown.add(e.code);
}

function onKeyUp(e) {
  keysDown.delete(e.code);
}

// Release all keys when window loses focus (alt-tab, etc.)
window.addEventListener('blur', () => { keysDown.clear(); vel.set(0, 0, 0); });

// ── Eye-height ──────────────────────────────────────────────────────

heightSlider.addEventListener('input', () => {
  eyeHeight = parseFloat(heightSlider.value);
  heightValEl.textContent = `${eyeHeight.toFixed(2)} m`;
});

function onWheel(e) {
  if (!walking) return;
  e.preventDefault();
  eyeHeight = Math.max(0.5, Math.min(3.0, eyeHeight - e.deltaY * 0.001));
  heightSlider.value = eyeHeight;
  heightValEl.textContent = `${eyeHeight.toFixed(2)} m`;
}

// ── Hint bar ────────────────────────────────────────────────────────

function hint(mode) {
  switch (mode) {
    case 'pick':
      hintEl.textContent = 'Click on the floor to start walking';
      hintEl.style.display = '';
      break;
    case 'transitioning':
      hintEl.style.display = 'none';
      break;
    case 'move':
      hintEl.textContent = 'WASD to move  \u00b7  Mouse to look  \u00b7  Scroll to adjust height  \u00b7  Esc to pause';
      hintEl.style.display = '';
      break;
    case 'resume':
      hintEl.textContent = 'Click to resume  \u00b7  Click floor to teleport  \u00b7  Exit to return';
      hintEl.style.display = '';
      break;
    default:
      hintEl.textContent = '';
  }
}
