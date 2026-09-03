/**
 * Main-thread cost of one frame of visibility work, CPU path versus
 * GPU-driven path, for the WebGPU backend.
 *
 * The renderer is stubbed: `compute()` records the dispatch and returns. That
 * is deliberate. What this measures is the JavaScript the library runs every
 * frame, which is the cost the GPU path exists to remove, and it is the same
 * number on any GPU. It says nothing about GPU-side time.
 *
 * Usage: node test/bench-culling.mjs [--counts 10000,100000] [--iters 20]
 */
import { BoxGeometry, MeshBasicNodeMaterial, PerspectiveCamera, Scene } from 'three/webgpu';
import { InstancedMesh2 } from '../dist/build/webgpu.js';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);

const COUNTS = (args.get('--counts') ?? '10000,50000,100000,250000,500000,1000000').split(',').map(Number);
const ITERS = Number(args.get('--iters') ?? 20);
const SCENARIOS = [
  { label: 'most visible', spread: 150 },
  { label: 'half visible', spread: 700 },
  { label: 'few visible', spread: 6000 },
  // A high-water allocation whose slots are mostly dead: the shape `ez-plants`
  // produces after a field re-bands. Only the tail is reclaimed, so both paths
  // still walk the whole array.
  { label: 'mostly unused', spread: 150, deadFraction: 0.8 }
];

const scene = new Scene();
const camera = new PerspectiveCamera(60, 16 / 9, 0.1, 1000);
camera.position.set(0, 30, 0);
camera.lookAt(0, 0, -100);
camera.updateMatrixWorld(true);
camera.updateProjectionMatrix();

let renderCall = 0;
const stubRenderer = {
  info: { render: { get calls() { return renderCall; } } },
  compute() {}
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
function build(count, spread, culling, deadFraction = 0) {
  const mesh = new InstancedMesh2(new BoxGeometry(1, 1, 1), new MeshBasicNodeMaterial(), { capacity: count, culling });
  let seed = 12345;
  /** @returns {number} */
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
  const random = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  mesh.addInstances(count, (instance) => {
    instance.position.set((random() - 0.5) * spread, 0, (random() - 0.5) * spread);
  });
  if (deadFraction > 0) {
    const dead = [];
    // Leave the last slot alive so the high-water mark cannot shrink.
    for (let id = 0; id < Math.floor(count * deadFraction); id++) dead.push(id);
    for (let start = 0; start < dead.length; start += 4096) {
      mesh.removeInstances(...dead.slice(start, start + 4096));
    }
  }
  mesh.updateMatrixWorld(true);
  mesh.geometry.computeBoundingSphere();
  return mesh;
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
function measure(frame) {
  for (let i = 0; i < 5; i++) frame();
  const started = performance.now();
  for (let i = 0; i < ITERS; i++) frame();
  return (performance.now() - started) / ITERS;
}

const rows = [];
for (const count of COUNTS) {
  for (const { label, spread, deadFraction = 0 } of SCENARIOS) {
    const cpuMesh = build(count, spread, 'cpu', deadFraction);
    const cpu = measure(() => {
      renderCall++;
      cpuMesh.onBeforeRender(stubRenderer, scene, camera, cpuMesh.geometry, cpuMesh.material, null);
    });
    const visible = cpuMesh.count;
    cpuMesh.dispose();

    const gpuMesh = build(count, spread, 'gpu', deadFraction);
    const gpu = measure(() => {
      renderCall++;
      gpuMesh.onBeforeRender(stubRenderer, scene, camera, gpuMesh.geometry, gpuMesh.material, null);
    });
    if (!gpuMesh.gpuCullingActive) throw new Error('the GPU path did not engage');
    gpuMesh.dispose();

    rows.push({ count, label, visible: (visible / count * 100).toFixed(1) + '%', cpu, gpu });
  }
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const pad = (value, width) => String(value).padStart(width);
console.log('instances   scenario        visible   CPU path   GPU path   speedup');
for (const row of rows) {
  console.log(
    `${pad(row.count, 9)}   ${row.label.padEnd(14)}  ${pad(row.visible, 6)}   `
    + `${pad(row.cpu.toFixed(3), 7)}ms   ${pad(row.gpu.toFixed(3), 7)}ms   ${pad((row.cpu / row.gpu).toFixed(0) + 'x', 6)}`
  );
}
