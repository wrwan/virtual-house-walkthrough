/**
 * Digital Twin Viewer — main entry point (orchestrator).
 *
 * All logic lives in focused sub-modules; this file just wires them
 * together and starts the render loop.
 */

import { renderer, scene, camera, controls } from './state.js';

// Side-effect imports — each module registers its own event listeners.
import './upload.js';
import './controls.js';

// wallTools exports the per-frame label-updater used in animate().
import { updateFloatingLabels } from './wallTools.js';

// -- Render loop -----------------------------------------------------
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateFloatingLabels();
  renderer.render(scene, camera);
}
animate();
