/**
 * Digital Twin Viewer — main entry point (orchestrator).
 *
 * All logic lives in focused sub-modules; this file just wires them
 * together and starts the render loop.
 */

import * as THREE from 'three';
import { renderer, scene, camera, controls } from './state.js';

// Side-effect imports — each module registers its own event listeners.
import './upload.js';
import './controls.js';

// wallTools exports the per-frame label-updater used in animate().
import { updateFloatingLabels } from './wallTools.js';

// walkMode exports the per-frame walk updater.
import { isWalking, updateWalk } from './walkMode.js';

// -- Render loop -----------------------------------------------------
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  if (isWalking()) {
    updateWalk(delta);
  } else {
    controls.update();
  }

  updateFloatingLabels();
  renderer.render(scene, camera);
}
animate();
