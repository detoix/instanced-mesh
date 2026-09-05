import assert from 'node:assert/strict';
import test from 'node:test';

import { BoxGeometry, Matrix4, MeshStandardMaterial, PerspectiveCamera, PlaneGeometry, Raycaster, Scene, Vector3 } from 'three/webgpu';

import * as webgpu from '../dist/build/webgpu.js';

const {
  DRAW_COMMAND_BYTES,
  DRAW_COMMAND_WORDS,
  INSTANCE_STATE_ACTIVE,
  INSTANCE_STATE_VISIBLE,
  InstancedMesh2,
  WebGPUCullingPass,
  WebGPUIndirectDrawBuffer,
  WebGPUInstanceStateBuffer,
  fitsStorageBindings
} = webgpu;

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const createMesh = (options = {}) =>
  new InstancedMesh2(new BoxGeometry(), new MeshStandardMaterial(), { capacity: 8, ...options });

// A renderer stub. `info.render.calls` is the per-pass identity the backend
// deduplicates on, and `compute` records dispatches instead of running them.
// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const createRendererStub = () => {
  const state = { calls: 0, dispatches: [] };
  return {
    state,
    info: { render: { get calls() { return state.calls; } } },
    compute(nodes) { state.dispatches.push(nodes); },
    // Three frees a storage attribute's GPU buffer only through the private
    // `_attributes.delete()`; `BufferAttribute.dispose()` just dispatches an
    // event. Recording it here is what proves the buffer was actually returned.
    _attributes: {
      released: [],
      delete(attribute) {
        this.released.push(attribute);
        return null;
      }
    }
  };
};

// Three counts an indirect attribute under `info.memory.indirectStorageAttributes`
// and every other storage attribute under `.storageAttributes`, keyed off this
// same flag (see Bindings.js). Splitting the released list the same way is what
// keeps a leak in the indirect buffers from hiding behind a passing storage count.
// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const releasedByKind = (renderer) => {
  const released = renderer._attributes.released;
  return {
    storage: released.filter((attribute) => !attribute.isIndirectStorageBufferAttribute),
    indirect: released.filter((attribute) => attribute.isIndirectStorageBufferAttribute)
  };
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const ownedAttributes = (mesh) => {
  const levels = (mesh.LODinfo?.objects ?? []).filter((object) => object !== mesh);
  return {
    storage: [
      mesh.instanceIndex.attribute,
      mesh.getInstanceIndexForPass(true).attribute,
      mesh.matricesTexture.attribute,
      mesh.colorsTexture.attribute,
      mesh._instanceState.attribute,
      // A level owns its two indexes and shares everything else with the owner.
      ...levels.flatMap((level) => [level.instanceIndex.attribute, level.getInstanceIndexForPass(true).attribute])
    ],
    indirect: mesh._cullPasses.filter(Boolean).map((pass) => pass.indirect.attribute)
  };
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const countReleases = (renderer, attribute) =>
  renderer._attributes.released.filter((released) => released === attribute).length;

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const renderOnce = (mesh, renderer, camera, scene = new Scene()) => {
  renderer.state.calls++;
  mesh.onBeforeRender(renderer, scene, camera, mesh.geometry, mesh.material, null);
};

test('the GPU path engages by default and yields to every CPU-only feature', () => {
  const mesh = createMesh();
  assert.equal(mesh.culling, 'auto');
  assert.equal(mesh.gpuCullingActive, true);

  mesh.sortObjects = true;
  assert.equal(mesh.gpuCullingActive, false, 'depth sorting has no GPU equivalent');
  mesh.sortObjects = false;
  assert.equal(mesh.gpuCullingActive, true);

  mesh.onFrustumEnter = () => true;
  assert.equal(mesh.gpuCullingActive, false, 'a per-instance JS callback cannot run in a kernel');
  mesh.onFrustumEnter = null;

  mesh.resolveLODIndex = () => 0;
  assert.equal(mesh.gpuCullingActive, false, 'a per-instance JS LOD resolver cannot run in a kernel');
  mesh.resolveLODIndex = null;
  assert.equal(mesh.gpuCullingActive, true);

  mesh.culling = 'cpu';
  assert.equal(mesh.gpuCullingActive, false, 'an explicit opt-out is honoured');

  mesh.dispose();
});

test('culling mode \'gpu\' reports the reason it had to fall back, once', () => {
  const mesh = createMesh({ culling: 'gpu' });
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (message) => warnings.push(message);

  try {
    mesh.resolveLODIndex = () => 0;
    assert.equal(mesh.gpuCullingActive, false);
    assert.equal(mesh.gpuCullingActive, false);
  } finally {
    console.warn = originalWarn;
  }

  assert.equal(warnings.length, 1, 'the fallback is reported once, not once per frame');
  assert.match(warnings[0], /resolveLODIndex/);
  assert.match(warnings[0], /setLODOverrideAt/);

  mesh.dispose();
});

test('the CPU fallback still compacts the visible prefix and uploads it', () => {
  const mesh = createMesh({ culling: 'cpu' });
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);

  mesh.addInstances(4, (instance, id) => instance.position.set(id, 0, 0));
  mesh.setVisibilityAt(1, false);
  mesh.perObjectFrustumCulled = false;

  renderOnce(mesh, renderer, camera);

  assert.equal(mesh.count, 3);
  assert.deepEqual(Array.from(mesh.instanceIndex.array.subarray(0, 3)), [0, 2, 3]);
  assert.equal(mesh.geometry.indirect, null, 'the CPU path must not leave an indirect draw behind');
  assert.equal(renderer.state.dispatches.length, 0);

  mesh.dispose();
});

test('the GPU path dispatches once per pass and leaves the visible prefix to the kernel', () => {
  const mesh = createMesh({ culling: 'gpu' });
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);

  mesh.addInstances(4, (instance, id) => instance.position.set(id, 0, 0));
  const untouched = Array.from(mesh.instanceIndex.array);

  renderOnce(mesh, renderer, camera);

  assert.equal(renderer.state.dispatches.length, 1);
  assert.equal(renderer.state.dispatches[0].length, 2, 'one reset and one cull kernel');
  assert.deepEqual(Array.from(mesh.instanceIndex.array), untouched, 'the CPU must not write the visible list');
  assert.ok(mesh.geometry.indirect, 'the draw must be indirect');
  assert.equal(mesh.geometry.indirectOffset, 0);
  // `count` is only an upper bound now: it exists so Three still issues the
  // draw, and the kernel supplies the real instance count.
  assert.equal(mesh.count, 4);

  // A second object in the same pass must reuse the first dispatch.
  mesh.onBeforeRender(renderer, new Scene(), camera, mesh.geometry, mesh.material, null);
  assert.equal(renderer.state.dispatches.length, 1);

  // A new pass dispatches again.
  renderOnce(mesh, renderer, camera);
  assert.equal(renderer.state.dispatches.length, 2);

  mesh.dispose();
});

test('active and visible flags reach the GPU state buffer as one coalesced range', () => {
  const mesh = createMesh({ culling: 'gpu' });
  mesh.addInstances(4);

  const state = mesh._instanceState;
  assert.equal(state._data[0], INSTANCE_STATE_ACTIVE | INSTANCE_STATE_VISIBLE);

  mesh.setVisibilityAt(1, false);
  assert.equal(state._data[1], INSTANCE_STATE_ACTIVE);
  assert.equal(mesh.getVisibilityAt(1), false, 'the CPU mirror stays authoritative for reads');

  mesh.removeInstances(2);
  assert.equal(state._data[2] & INSTANCE_STATE_ACTIVE, 0);

  state.attribute.clearUpdateRanges();
  mesh.setVisibilityAt(0, false);
  mesh.setVisibilityAt(3, false);
  state.flush(4);
  assert.deepEqual(state.attribute.updateRanges, [{ start: 0, count: 4 }], 'edits coalesce into one range');

  mesh.dispose();
});

test('capacity growth preserves instance state and reallocates its attribute', () => {
  const mesh = createMesh({ capacity: 4, culling: 'gpu' });
  mesh.addInstances(4);
  mesh.setVisibilityAt(2, false);

  const before = mesh._instanceState.attribute;
  mesh.addInstances(6);

  assert.notStrictEqual(mesh._instanceState.attribute, before, 'a storage attribute cannot grow in place');
  assert.equal(mesh._instanceState.capacity, mesh.capacity);
  assert.equal(mesh._instanceState._data[2], INSTANCE_STATE_ACTIVE, 'the hidden instance stayed hidden');
  assert.equal(mesh._instanceState._data[9], INSTANCE_STATE_ACTIVE | INSTANCE_STATE_VISIBLE);

  mesh.dispose();
});

test('per-instance LOD overrides are stored independently for render and shadow passes', () => {
  const mesh = createMesh({ culling: 'gpu' });
  mesh.addInstances(3);

  assert.equal(mesh.getLODOverrideAt(0), -1, 'distance selection is the default');

  mesh.setLODOverrideAt(0, 2);
  mesh.setLODOverrideAt(0, 1, true);
  assert.equal(mesh.getLODOverrideAt(0), 2);
  assert.equal(mesh.getLODOverrideAt(0, true), 1);

  // The overrides share the state word with the flags and must not disturb them.
  assert.equal(mesh._instanceState._data[0] & 3, INSTANCE_STATE_ACTIVE | INSTANCE_STATE_VISIBLE);
  mesh.setVisibilityAt(0, false);
  assert.equal(mesh.getLODOverrideAt(0), 2);
  assert.equal(mesh.getLODOverrideAt(0, true), 1);

  mesh.setLODOverrideAt(0, -1);
  assert.equal(mesh.getLODOverrideAt(0), -1);
  assert.equal(mesh.getLODOverrideAt(0, true), 1, 'clearing one pass leaves the other alone');

  mesh.dispose();
});

test('indirect commands carry the geometry words and leave instanceCount to the kernel', () => {
  const buffer = new WebGPUIndirectDrawBuffer(2);
  assert.equal(buffer.commandCount, 2);
  assert.equal(buffer.byteOffset(1), DRAW_COMMAND_BYTES);
  assert.equal(DRAW_COMMAND_WORDS, 5);

  buffer.writeCommand(0, 36, 0, true);
  buffer.writeCommand(1, 24, 12, false);

  const words = buffer.attribute.array;
  assert.deepEqual(Array.from(words.subarray(0, 5)), [36, 0, 0, 0, 0]);
  assert.deepEqual(Array.from(words.subarray(5, 10)), [24, 0, 12, 0, 0]);
  assert.equal(buffer.attribute.isIndirectStorageBufferAttribute, true);

  buffer.dispose();
});

test('a culling pass lays its commands out per level and per material group', () => {
  const mesh = createMesh({ culling: 'gpu' });
  mesh.addInstances(2);

  const pass = new WebGPUCullingPass('test');
  pass.build({
    matrixAttribute: mesh.matricesTexture.attribute,
    stateAttribute: mesh._instanceState.attribute,
    capacity: mesh.capacity,
    lodOverrideShift: 8,
    levels: [
      { visibleAttribute: mesh.instanceIndex.attribute, distanceSquared: 0, hysteresis: 0, indexed: true, ranges: [{ elementCount: 12, firstElement: 0 }, { elementCount: 24, firstElement: 12 }] },
      { visibleAttribute: mesh.getInstanceIndexForPass(true).attribute, distanceSquared: 100, hysteresis: 0, indexed: true, ranges: [{ elementCount: 6, firstElement: 0 }] }
    ]
  });

  assert.equal(pass.levelCount, 2);
  assert.deepEqual(pass.rangeCounts, [2, 1]);
  assert.equal(pass.commandByteOffset(0, 0), 0);
  assert.equal(pass.commandByteOffset(0, 1), DRAW_COMMAND_BYTES);
  assert.equal(pass.commandByteOffset(1, 0), DRAW_COMMAND_BYTES * 2);
  assert.equal(pass.indirect.commandCount, 3);

  mesh.dispose();
  pass.dispose();
});

test('a multi-material mesh dispatches an extra kernel to publish counts to every group', () => {
  const geometry = new BoxGeometry();
  geometry.clearGroups();
  geometry.addGroup(0, 18, 0);
  geometry.addGroup(18, 18, 1);
  const mesh = new InstancedMesh2(
    geometry,
    [new MeshStandardMaterial(), new MeshStandardMaterial()],
    { capacity: 4, culling: 'gpu' }
  );
  mesh.addInstances(2);

  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  const scene = new Scene();

  renderer.state.calls++;
  mesh.onBeforeRender(renderer, scene, camera, geometry, mesh.material, geometry.groups[0]);
  const firstOffset = mesh.geometry.indirectOffset;
  mesh.onBeforeRender(renderer, scene, camera, geometry, mesh.material, geometry.groups[1]);
  const secondOffset = mesh.geometry.indirectOffset;

  assert.equal(renderer.state.dispatches[0].length, 3, 'reset, cull, and the count publish');
  assert.equal(firstOffset, 0);
  assert.equal(secondOffset, DRAW_COMMAND_BYTES, 'each group draws its own command');

  const words = mesh.geometry.indirect.array;
  assert.equal(words[0], 18, 'group 0 draws its own index range');
  assert.equal(words[DRAW_COMMAND_WORDS], 18);
  assert.equal(words[DRAW_COMMAND_WORDS + 2], 18, 'group 1 starts where group 0 ends');

  mesh.dispose();
});

test('LOD level count is bounded by the device storage-buffer limit', () => {
  // Matrices, instance state and the indirect commands take three bindings.
  assert.equal(fitsStorageBindings(5, 8), true);
  assert.equal(fitsStorageBindings(6, 8), false);
  assert.equal(fitsStorageBindings(1, 4), true);
  assert.equal(fitsStorageBindings(2, 4), false);
});

test('a mesh with LODs gives every level its own indirect command slot', () => {
  const mesh = createMesh({ culling: 'gpu' });
  const farGeometry = new PlaneGeometry(1, 1);
  mesh.addLOD(farGeometry, new MeshStandardMaterial(), 10);
  mesh.addInstances(4, (instance, id) => instance.position.set(id * 4, 0, -5));

  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  const scene = new Scene();

  const [near, far] = mesh.LODinfo.render.levels;
  renderer.state.calls++;
  near.object.onBeforeRender(renderer, scene, camera, near.object.geometry, near.object.material, null);
  far.object.onBeforeRender(renderer, scene, camera, far.object.geometry, far.object.material, null);

  assert.equal(renderer.state.dispatches.length, 1, 'one dispatch classifies every level');
  assert.strictEqual(near.object.geometry.indirect, far.object.geometry.indirect);
  assert.equal(near.object.geometry.indirectOffset, 0);
  assert.equal(far.object.geometry.indirectOffset, DRAW_COMMAND_BYTES);

  mesh.dispose();
  farGeometry.dispose();
});

test('manual culling still drives the GPU path when autoUpdate is off', () => {
  const mesh = createMesh({ culling: 'gpu' });
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  mesh.addInstances(2);
  mesh.autoUpdate = false;

  renderOnce(mesh, renderer, camera);
  assert.equal(renderer.state.dispatches.length, 0, 'nothing was requested');

  mesh.performFrustumCulling(camera);
  renderOnce(mesh, renderer, camera);
  assert.equal(renderer.state.dispatches.length, 1, 'the recorded request was dispatched at draw time');

  mesh.dispose();
});

test('every mesh in a pass shares one compute submission', () => {
  // Three submits a compute group on its own command buffer, so one dispatch
  // per mesh is one GPU submission per mesh. A pooled field has dozens.
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  const scene = new Scene();
  const meshes = Array.from({ length: 12 }, () => {
    const mesh = createMesh({ culling: 'gpu' });
    mesh.addInstances(4);
    return mesh;
  });

  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
  const pass = (callback) => {
    renderer.state.calls++;
    for (const mesh of meshes) callback(mesh);
  };

  // The first pass learns the membership one mesh at a time.
  pass((mesh) => mesh.onBeforeRender(renderer, scene, camera, mesh.geometry, mesh.material, null));
  const learning = renderer.state.dispatches.length;
  assert.ok(learning <= meshes.length);

  // From the second pass on it is a single submission carrying every kernel.
  renderer.state.dispatches.length = 0;
  pass((mesh) => mesh.onBeforeRender(renderer, scene, camera, mesh.geometry, mesh.material, null));
  assert.equal(renderer.state.dispatches.length, 1, 'the pass took more than one submission');
  assert.equal(
    renderer.state.dispatches[0].length,
    meshes.length * 2,
    'the submission is missing a reset or cull kernel'
  );

  // A shadow pass batches separately, against its own camera.
  renderer.state.dispatches.length = 0;
  pass((mesh) => {
    mesh.onBeforeShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
    mesh.onAfterShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
  });
  renderer.state.dispatches.length = 0;
  pass((mesh) => {
    mesh.onBeforeShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
    mesh.onAfterShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
  });
  assert.equal(renderer.state.dispatches.length, 1, 'the shadow pass took more than one submission');

  for (const mesh of meshes) mesh.dispose();
});

test('instance edits reach the GPU even when culling is frozen', () => {
  // `autoUpdate = false` freezes *culling*, not the instance payloads. An
  // application that pins the visible set and keeps animating its instances
  // must still see its matrices, colours and flags uploaded -- the CPU path
  // has always done this unconditionally.
  for (const autoUpdate of [true, false]) {
    const mesh = createMesh({ culling: 'gpu' });
    const renderer = createRendererStub();
    const camera = new PerspectiveCamera();
    camera.updateMatrixWorld(true);
    const scene = new Scene();
    mesh.addInstances(4);
    mesh.setColorAt(0, 0xffffff);
    mesh.autoUpdate = autoUpdate;

    renderOnce(mesh, renderer, camera, scene);
    for (const buffer of [mesh.matricesTexture, mesh.colorsTexture, mesh._instanceState]) {
      buffer.attribute.clearUpdateRanges();
    }

    const matrix = new Matrix4().makeTranslation(5, 5, 5);
    mesh.setMatrixAt(2, matrix);
    mesh.setColorAt(2, 0x336699);
    mesh.setVisibilityAt(3, false);
    renderOnce(mesh, renderer, camera, scene);

    const label = `autoUpdate=${autoUpdate}`;
    assert.ok(mesh.matricesTexture.attribute.updateRanges.length > 0, `matrix edit was dropped (${label})`);
    assert.ok(mesh.colorsTexture.attribute.updateRanges.length > 0, `colour edit was dropped (${label})`);
    assert.ok(mesh._instanceState.attribute.updateRanges.length > 0, `visibility edit was dropped (${label})`);

    mesh.dispose();
  }
});

test('a steady frame rebuilds no compute graph, with or without a shadow pass', () => {
  // A rebuild regenerates the TSL graph and allocates new compute nodes, which
  // costs a pipeline compilation. Doing that per frame is the difference
  // between a working field and one frame a second, so it is pinned here for
  // both passes -- a mesh that never casts a shadow used to rebuild forever.
  for (const shadows of [false, true]) {
    const mesh = createMesh({ culling: 'gpu' });
    const renderer = createRendererStub();
    const camera = new PerspectiveCamera();
    camera.updateMatrixWorld(true);
    const scene = new Scene();
    mesh.addInstances(4);

    // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
    const frame = () => {
      renderer.state.calls++;
      mesh.onBeforeRender(renderer, scene, camera, mesh.geometry, mesh.material, null);
      if (!shadows) return;
      renderer.state.calls++;
      mesh.onBeforeShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
      mesh.onAfterShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
    };

    for (let i = 0; i < 3; i++) frame();
    const before = mesh._cullPasses.map((pass) => pass?.revision ?? 0);
    for (let i = 0; i < 20; i++) frame();
    const after = mesh._cullPasses.map((pass) => pass?.revision ?? 0);

    assert.deepEqual(after, before, `a steady frame rebuilt a compute graph (shadows: ${shadows})`);
    mesh.dispose();
  }
});

test('rebuilding a pass disposes the kernels it replaces', () => {
  // Three releases a compute pipeline, its bindings and its node state from
  // the node's own dispose event. A replaced kernel that is never disposed
  // strands all three for the renderer's lifetime.
  const mesh = createMesh({ culling: 'gpu' });
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  const scene = new Scene();
  mesh.addInstances(4);

  renderOnce(mesh, renderer, camera, scene);
  const replaced = renderer.state.dispatches[0];
  const disposed = replaced.map(() => 0);
  replaced.forEach((node, index) => {
    node.addEventListener('dispose', () => {
      disposed[index]++;
    });
  });

  // Growth replaces the storage attributes, forcing a graph rebuild.
  mesh.resizeBuffers(64);
  renderOnce(mesh, renderer, camera, scene);

  assert.deepEqual(disposed, replaced.map(() => 1), 'a replaced kernel was left undisposed');
  assert.notStrictEqual(renderer.state.dispatches[1][0], replaced[0], 'the graph was not actually rebuilt');

  mesh.dispose();
});

test('a structural change rebuilds both passes exactly once', () => {
  const mesh = createMesh({ culling: 'gpu' });
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  const scene = new Scene();
  mesh.addInstances(4);

  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
  const frame = () => {
    renderer.state.calls++;
    mesh.onBeforeRender(renderer, scene, camera, mesh.geometry, mesh.material, null);
    renderer.state.calls++;
    mesh.onBeforeShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
    mesh.onAfterShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
  };

  for (let i = 0; i < 3; i++) frame();
  const before = mesh._cullPasses.map((pass) => pass.revision);

  // Growth replaces the storage attributes, so the graphs must be rebuilt.
  mesh.resizeBuffers(64);
  for (let i = 0; i < 5; i++) frame();
  const after = mesh._cullPasses.map((pass) => pass.revision);

  assert.deepEqual(
    after.map((value, index) => value - before[index]),
    [1, 1],
    'growth did not rebuild each pass exactly once'
  );

  mesh.dispose();
});

test('a disposed mesh drops out of its renderer batch', () => {
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  const scene = new Scene();
  const kept = createMesh({ culling: 'gpu' });
  const removed = createMesh({ culling: 'gpu' });
  for (const mesh of [kept, removed]) mesh.addInstances(2);

  for (let frame = 0; frame < 2; frame++) {
    renderer.state.calls++;
    for (const mesh of [kept, removed]) {
      mesh.onBeforeRender(renderer, scene, camera, mesh.geometry, mesh.material, null);
    }
  }

  removed.dispose();
  renderer.state.dispatches.length = 0;
  renderer.state.calls++;
  kept.onBeforeRender(renderer, scene, camera, kept.geometry, kept.material, null);

  assert.equal(renderer.state.dispatches.length, 1);
  assert.equal(
    renderer.state.dispatches[0].length,
    2,
    'the disposed mesh was still culled'
  );

  kept.dispose();
});

test('raycastOnlyFrustum still hits live instances on the GPU path', () => {
  // `raycastOnlyFrustum` walks `instanceIndex` for `count` entries. The GPU
  // path never compacts that list and `count` is only an upper bound, so
  // following it would test ids that are not the live ones.
  const mesh = new InstancedMesh2(new BoxGeometry(1, 1, 1), new MeshStandardMaterial(), {
    capacity: 16,
    culling: 'gpu'
  });
  mesh.addInstances(10, (instance, id) => instance.position.set(id * 2, 0, 0));
  // Leaves ids 5..9 live at x = 10, 12, 14, 16, 18, with a hole at the front.
  mesh.removeInstances(0, 1, 2, 3, 4);
  mesh.raycastOnlyFrustum = true;
  mesh.updateMatrixWorld(true);
  mesh.computeBoundingSphere();

  assert.equal(mesh.usesCPUVisibleList(), false, 'the GPU path has no CPU visible list to follow');
  assert.equal(mesh.instancesCount, 5);

  const raycaster = new Raycaster(new Vector3(14, 10, 0), new Vector3(0, -1, 0));
  const hits = [];
  mesh.raycast(raycaster, hits);

  assert.ok(hits.length > 0, 'a live instance under the ray was missed');
  assert.equal(hits[0].instanceId, 7, 'the wrong instance was reported');

  mesh.dispose();
});

test('shadow LOD levels are chosen by distance from the viewer, not the light', () => {
  // The WebGL path culls with the shadow camera but levels with the render
  // camera. Three r185 hands `onBeforeShadow` the shadow camera in both
  // slots, so the render camera has to be remembered for parity.
  const mesh = createMesh({ culling: 'gpu' });
  const farGeometry = new PlaneGeometry(1, 1);
  mesh.addLOD(farGeometry, new MeshStandardMaterial(), 10);
  mesh.addInstances(4);

  const renderer = createRendererStub();
  const scene = new Scene();
  const camera = new PerspectiveCamera();
  camera.position.set(0, 0, 50);
  camera.updateMatrixWorld(true);
  const shadowCamera = new PerspectiveCamera();
  shadowCamera.position.set(0, 50, 0);
  shadowCamera.updateMatrixWorld(true);

  renderer.state.calls++;
  mesh.onBeforeRender(renderer, scene, camera, mesh.geometry, mesh.material, null);
  renderer.state.calls++;
  mesh.onBeforeShadow(renderer, scene, shadowCamera, shadowCamera, mesh.geometry, mesh.material, null);
  mesh.onAfterShadow(renderer, scene, shadowCamera, shadowCamera, mesh.geometry, mesh.material, null);

  const shadowLODCamera = mesh._cullPasses[1]._cameraPosUniform.value;
  assert.deepEqual(
    [shadowLODCamera.x, shadowLODCamera.y, shadowLODCamera.z],
    [0, 0, 50],
    'the shadow pass levelled by distance from the light'
  );

  mesh.dispose();
  farGeometry.dispose();
});

test('an emptied mesh stops pointing at indirect commands nobody refreshed', () => {
  const mesh = createMesh({ culling: 'gpu' });
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld(true);
  const scene = new Scene();
  mesh.addInstances(4);

  renderOnce(mesh, renderer, camera, scene);
  assert.ok(mesh.geometry.indirect, 'the populated mesh should draw indirectly');

  mesh.clearInstances();
  renderOnce(mesh, renderer, camera, scene);

  assert.equal(mesh.count, 0, 'an emptied mesh must not be drawn');
  assert.equal(mesh.geometry.indirect, null, 'stale indirect commands were left bound');

  mesh.dispose();
});

test('an instance state buffer rejects out-of-range writes', () => {
  const state = new WebGPUInstanceStateBuffer(4);
  assert.throws(() => state.enqueueUpdate(4), RangeError);
  assert.throws(() => state.flush(5), RangeError);
  state.dispose();
});

test('disposing a GPU-culled mesh releases every storage AND indirect attribute it owns, once', () => {
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  const scene = new Scene();
  const mesh = createMesh();

  mesh.addInstances(4, (instance, id) => instance.position.set(id, 0, 0));
  mesh.setColorAt(0, 0x336699);
  mesh.addLOD(new BoxGeometry(0.5, 0.5, 0.5), new MeshStandardMaterial(), 12);

  // Both passes, so both culling passes and both indirect buffers exist.
  renderOnce(mesh, renderer, camera, scene);
  renderer.state.calls++;
  mesh.onBeforeShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);
  mesh.onAfterShadow(renderer, scene, camera, camera, mesh.geometry, mesh.material, null);

  const owned = ownedAttributes(mesh);
  assert.equal(owned.storage.length, 7, 'the owner\'s five buffers plus the level\'s two indexes');
  assert.equal(owned.indirect.length, 2, 'a render and a shadow culling pass were built');
  const all = [...owned.storage, ...owned.indirect];
  assert.equal(new Set(all).size, all.length, 'every owned buffer has its own attribute');
  assert.equal(
    releasedByKind(renderer).storage.length,
    0,
    'rendering never drops an instance buffer; only the culling rebuild reallocates, and only its indirect commands'
  );
  // Building the graph for two levels grew each pass's command buffer once, so
  // the indirect baseline is whatever those rebuilds already returned.
  const indirectBeforeDispose = releasedByKind(renderer).indirect.length;

  mesh.dispose();

  const released = releasedByKind(renderer);
  for (const attribute of owned.storage) {
    assert.equal(countReleases(renderer, attribute), 1, `${attribute.name} is released exactly once`);
  }
  for (const attribute of owned.indirect) {
    assert.equal(countReleases(renderer, attribute), 1, `indirect ${attribute.name} is released exactly once`);
  }
  assert.equal(released.storage.length, owned.storage.length, 'the storage count returns to baseline');
  assert.equal(
    released.indirect.length,
    indirectBeforeDispose + owned.indirect.length,
    'the indirect count returns to baseline'
  );

  const afterDispose = renderer._attributes.released.length;
  mesh.dispose();
  assert.equal(renderer._attributes.released.length, afterDispose, 'a second dispose() releases nothing again');
});

test('a grow cycle releases the attributes it reallocates away from, storage and indirect alike', () => {
  const renderer = createRendererStub();
  const camera = new PerspectiveCamera();
  const scene = new Scene();
  const mesh = createMesh({ capacity: 2 });

  mesh.addInstances(2, (instance, id) => instance.position.set(id, 0, 0));
  mesh.setColorAt(0, 0x336699);
  renderOnce(mesh, renderer, camera, scene);

  const before = ownedAttributes(mesh);
  assert.equal(before.indirect.length, 1);

  // Growing reallocates every per-instance buffer; adding a level rebuilds the
  // culling graph, which reallocates the indirect commands.
  mesh.resizeBuffers(16);
  mesh.addLOD(new BoxGeometry(0.5, 0.5, 0.5), new MeshStandardMaterial(), 12);
  renderer.state.calls++;
  renderOnce(mesh, renderer, camera, scene);

  const after = ownedAttributes(mesh);
  for (const attribute of before.storage) {
    assert.equal(countReleases(renderer, attribute), 1, `the replaced ${attribute.name} is released once`);
    assert.equal(after.storage.includes(attribute), false, 'the mesh moved to a fresh attribute');
  }
  assert.equal(
    releasedByKind(renderer).indirect.length,
    1,
    'the reallocated indirect commands are released, so the indirect count does not climb'
  );
  assert.equal(before.indirect.includes(after.indirect[0]), false);

  mesh.dispose();

  const released = releasedByKind(renderer);
  assert.equal(
    released.storage.length,
    before.storage.length + after.storage.length,
    'every storage generation is released exactly once'
  );
  assert.equal(
    released.indirect.length,
    before.indirect.length + after.indirect.length,
    'every indirect generation is released exactly once'
  );
});

test('a GPU-culled mesh disposed before it was ever rendered has no renderer to release through', () => {
  const mesh = createMesh();
  mesh.addInstances(1);
  mesh.setColorAt(0, 0x336699);

  assert.doesNotThrow(() => mesh.dispose());
});
