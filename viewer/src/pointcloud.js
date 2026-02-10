/**
 * Point cloud loading from API.
 */

import * as THREE from 'three';
import { API, pointCloudGroup, camera, controls } from './state.js';

/** Fetch and render the point cloud. */
export async function loadPoints() {
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
    console.log('[POINTS] Using RGB colors from file...');
    colors = new Float32Array(data.colors);
    console.log(`[POINTS] Applied ${count.toLocaleString()} RGB colors`);
  } else {
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
      colors[i * 3] = 0.2 + 0.6 * t;
      colors[i * 3 + 1] = 0.3 + 0.4 * (1 - Math.abs(t - 0.5) * 2);
      colors[i * 3 + 2] = 0.9 - 0.7 * t;
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
