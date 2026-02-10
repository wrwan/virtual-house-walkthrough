/**
 * Shared scene state — Three.js objects, groups, and mutable app state.
 *
 * Every viewer module imports from here so there's a single source of
 * truth for the scene graph, camera, controls, and runtime state.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const API = '/api';

// ── Renderer & camera ───────────────────────────────────────────────
export const container = document.getElementById('canvas-container');
export const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x1a1a2e);
container.appendChild(renderer.domElement);

export const scene = new THREE.Scene();

export const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(5, 5, 5);

export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// Helpers
scene.add(new THREE.GridHelper(20, 40, 0x333355, 0x222244));
scene.add(new THREE.AxesHelper(2));

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Groups ──────────────────────────────────────────────────────────
export const pointCloudGroup = new THREE.Group();
pointCloudGroup.name = 'pointCloud';
pointCloudGroup.rotation.x = -Math.PI / 2;
scene.add(pointCloudGroup);

export const planeGroup = new THREE.Group();
planeGroup.name = 'planes';
planeGroup.rotation.x = -Math.PI / 2;
scene.add(planeGroup);

export const wallGroup = new THREE.Group(); wallGroup.name = 'walls';
export const floorGroup = new THREE.Group(); floorGroup.name = 'floor';
planeGroup.add(wallGroup, floorGroup);

export const cornerGroup = new THREE.Group();
cornerGroup.name = 'cornerMarkers';
scene.add(cornerGroup);

export const previewGroup = new THREE.Group();
previewGroup.name = 'wallPreviews';
planeGroup.add(previewGroup);

// ── Mutable state ───────────────────────────────────────────────────
/** Array of { index, lineObj, fillObj, plane } */
export let wallObjects = [];

/** Replace the wallObjects array (needed because `let` can't be re-assigned from outside). */
export function setWallObjects(arr) { wallObjects = arr; }

/** Raycaster shared across pointer-based tools */
export const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.05;
export const mouse = new THREE.Vector2();

// ── Colours ─────────────────────────────────────────────────────────
export const PLANE_COLORS = {
  floor: 0x4caf50,
  ceiling: 0x2196f3,
  wall: 0xff9800,
  manual: 0xe91e63,
  unknown: 0x999999,
};
