import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BoxGeometry,
  BufferGeometry,
  Float32BufferAttribute,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  ShaderChunk,
  ShaderMaterial,
  Skeleton
} from 'three';
import { vec3, vec4 } from 'three/tsl';

import * as rootPackage from '../dist/build/index.js';
import * as webgpuPackage from '../dist/build/webgpu.js';

const RootInstancedMesh2 = rootPackage.InstancedMesh2;
const WebGPUInstancedMesh2 = webgpuPackage.InstancedMesh2;
const setWebGPUInstancePositionNode = webgpuPackage.setWebGPUInstancePositionNode;

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const createMesh = (capacity = 8, material = new MeshStandardMaterial()) =>
  new WebGPUInstancedMesh2(new BoxGeometry(), material, { capacity });

const disposeCounter = (target) => {
  let count = 0;
  target.addEventListener('dispose', () => count++);
  return () => count;
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const assertSingleUpdateRange = (attribute, start, count) => {
  assert.deepEqual(attribute.updateRanges, [{ start, count }]);
};

test('root and independently built WebGPU entry points expose distinct backends with their feature APIs', () => {
  assert.notStrictEqual(WebGPUInstancedMesh2, RootInstancedMesh2);
  assert.equal(typeof rootPackage.createInstancedMesh2From, 'function');
  assert.equal(Object.hasOwn(webgpuPackage, 'createInstancedMesh2From'), false);
  assert.equal(Object.hasOwn(rootPackage, 'setWebGPUInstancePositionNode'), false);
  assert.equal(typeof setWebGPUInstancePositionNode, 'function');

  for (const method of [
    'addInstances',
    'removeInstances',
    'performFrustumCulling',
    'addLOD',
    'setMatrixAt',
    'setColorAt',
    'setVisibilityAt'
  ]) {
    assert.equal(typeof RootInstancedMesh2.prototype[method], 'function', `${method} is available on the root backend`);
    assert.equal(typeof WebGPUInstancedMesh2.prototype[method], 'function', `${method} is available on the WebGPU backend`);
  }

  assert.match(ShaderChunk.batching_pars_vertex, /#include <instanced_pars_vertex>/);
  assert.match(ShaderChunk.project_vertex, /USE_INSTANCING_INDIRECT/);
});

test('instance-aware source position nodes survive cloning, growth, render LOD and shadow LOD rebuilds', () => {
  const sourceMaterial = new MeshStandardMaterial({ color: 0x668844 });
  const basePositionNode = vec3(1, 2, 3);
  const baseNormalNode = vec3(0, 1, 0);
  sourceMaterial.positionNode = basePositionNode;
  sourceMaterial.normalNode = baseNormalNode;

  const sourceContexts = [];
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- public JavaScript API fixture
  const sourceFactory = (context) => {
    sourceContexts.push(context);
    return context.positionNode.add(vec3(0.125, 0, 0));
  };
  assert.strictEqual(
    setWebGPUInstancePositionNode(sourceMaterial, sourceFactory),
    sourceMaterial
  );

  const geometry = new BoxGeometry();
  const mesh = new WebGPUInstancedMesh2(geometry, sourceMaterial, { capacity: 1 });

  assert.notStrictEqual(mesh.material, sourceMaterial);
  assert.strictEqual(sourceMaterial.positionNode, basePositionNode);
  assert.strictEqual(sourceMaterial.normalNode, baseNormalNode);
  assert.strictEqual(mesh.material.normalNode, baseNormalNode);
  assert.equal(mesh.material.castShadowPositionNode?.isNode, true);
  assert.notStrictEqual(mesh.material.castShadowPositionNode, mesh.material.positionNode);
  assert.equal(sourceContexts.length, 2);
  assert.strictEqual(sourceContexts[0].positionNode, basePositionNode);
  assert.strictEqual(sourceContexts[0].geometry, geometry);
  for (const key of ['normalNode', 'instanceMatrix', 'instanceId', 'drawInstanceIndex']) {
    assert.equal(sourceContexts[0][key]?.isNode, true, `${key} is supplied as a TSL node`);
  }

  const initialPositionNode = mesh.material.positionNode;
  const initialMatrixNode = sourceContexts[0].instanceMatrix;
  const initialProgramCacheKey = mesh.material.customProgramCacheKey();
  mesh.addInstances(2);
  assert.ok(mesh.capacity > 1);
  assert.notStrictEqual(mesh.material.positionNode, initialPositionNode);
  assert.notEqual(
    mesh.material.customProgramCacheKey(),
    initialProgramCacheKey,
    'storage growth invalidates Three WebGPU render-object bindings'
  );
  assert.ok(sourceContexts.length >= 2);
  assert.notStrictEqual(sourceContexts.at(-1).instanceMatrix, initialMatrixNode);
  assert.strictEqual(sourceContexts.at(-1).positionNode, basePositionNode);

  const reusedGeometry = new BoxGeometry(0.75, 0.75, 0.75);
  const reusedMesh = new WebGPUInstancedMesh2(
    reusedGeometry,
    mesh.material,
    { capacity: 1 }
  );
  assert.strictEqual(sourceContexts.at(-1).positionNode, basePositionNode);
  assert.strictEqual(reusedMesh.material.normalNode, baseNormalNode);
  assert.notStrictEqual(reusedMesh.material.positionNode, mesh.material.positionNode);

  const lodGeometry = new BoxGeometry(0.5, 0.5, 0.5);
  const lodMaterial = new MeshStandardMaterial({ color: 0x224466 });
  const lodNormalNode = vec3(0, 0, 1);
  lodMaterial.normalNode = lodNormalNode;
  const lodContexts = [];
  setWebGPUInstancePositionNode(lodMaterial, (context) => {
    lodContexts.push(context);
    return context.positionNode.add(vec3(0, 0.25, 0));
  });
  mesh.addLOD(lodGeometry, lodMaterial, 12);
  const lodObject = mesh.LODinfo.render.levels[1].object;
  assert.ok(lodObject instanceof WebGPUInstancedMesh2);
  assert.equal(lodContexts.length, 2);
  assert.strictEqual(lodContexts[0].geometry, lodGeometry);
  assert.strictEqual(lodObject.material.normalNode, lodNormalNode);

  const shadowGeometry = new BoxGeometry(0.25, 0.25, 0.25);
  const contextsBeforeShadowLOD = sourceContexts.length;
  mesh.addShadowLOD(shadowGeometry, 20);
  const shadowObject = mesh.LODinfo.shadowRender.levels[0].object;
  assert.ok(shadowObject instanceof WebGPUInstancedMesh2);
  assert.notStrictEqual(shadowObject, mesh);
  assert.ok(sourceContexts.length > contextsBeforeShadowLOD);
  assert.strictEqual(sourceContexts.at(-1).geometry, shadowGeometry);
  assert.strictEqual(shadowObject.material.normalNode, baseNormalNode);
  assert.equal(shadowObject.material.positionNode?.isNode, true);

  const callsBeforeSecondGrowth = sourceContexts.length;
  const lodCallsBeforeSecondGrowth = lodContexts.length;
  mesh.addInstances(mesh.capacity + 1);
  assert.ok(sourceContexts.length > callsBeforeSecondGrowth);
  assert.ok(lodContexts.length > lodCallsBeforeSecondGrowth);

  mesh.dispose();
  reusedMesh.dispose();
  geometry.dispose();
  reusedGeometry.dispose();
  lodGeometry.dispose();
  shadowGeometry.dispose();
  sourceMaterial.dispose();
  lodMaterial.dispose();
});

test('instance-aware source position node API validates inputs and supports removal', () => {
  const material = new MeshStandardMaterial();
  let calls = 0;
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- public JavaScript API fixture
  const factory = ({ positionNode }) => {
    calls++;
    return positionNode;
  };

  assert.throws(
    () => setWebGPUInstancePositionNode({}, factory),
    /requires a Three\.js material/
  );
  assert.throws(
    () => setWebGPUInstancePositionNode(material, {}),
    /must be a function or null/
  );
  setWebGPUInstancePositionNode(material, factory);
  setWebGPUInstancePositionNode(material, null);

  const mesh = createMesh(1, material);
  assert.equal(calls, 0);
  mesh.dispose();
  material.dispose();
});

test('WebGPU storage accepts a 100001 capacity and a visible prefix above 1000', () => {
  const mesh = createMesh(100_001);

  assert.equal(mesh.capacity, 100_001);
  assert.equal(mesh.instanceIndex.array.length, 100_001);
  assert.equal(mesh.instanceIndex.attribute.count, 100_001);
  assert.equal(mesh.matricesTexture._data.length, 100_001 * 16);
  assert.equal(mesh.matricesTexture.attribute.count, 100_001);

  mesh.addInstances(1_001);
  mesh.perObjectFrustumCulled = false;
  mesh.performFrustumCulling(new PerspectiveCamera());

  assert.equal(mesh.instancesCount, 1_001);
  assert.equal(mesh.count, 1_001);
  assert.deepEqual(Array.from(mesh.instanceIndex.array.subarray(997, 1_001)), [997, 998, 999, 1_000]);
  assert.equal(mesh.instanceIndex.flush(mesh.count), true);
  assertSingleUpdateRange(mesh.instanceIndex.attribute, 0, 1_001);

  mesh.dispose();
});

test('WebGPU keeps per-object instancing cache identity without enabling Three native transforms', () => {
  const mesh = createMesh();
  mesh.addInstances(1);

  assert.equal(mesh.isInstancedMesh2, true);
  assert.equal(mesh.isInstancedMesh, true);
  assert.equal(mesh.instanceMatrix.isBufferAttribute, true);
  assert.equal(mesh.instanceMatrix.isInstancedBufferAttribute, undefined);
  assert.equal(mesh.instanceMatrix.count, 0);
  assert.equal(mesh.instanceColor, null);

  let serialized;
  assert.doesNotThrow(() => {
    serialized = mesh.toJSON();
  });
  assert.equal(serialized.object.type, 'InstancedMesh');
  assert.equal(serialized.object.instanceMatrix.itemSize, 16);
  assert.deepEqual(serialized.object.instanceMatrix.array, []);

  mesh.dispose();
});

test('add/remove reuses free ids and visibility produces a compact visible-index prefix', () => {
  const mesh = createMesh();
  const camera = new PerspectiveCamera();

  mesh.addInstances(5, (instance, id) => instance.position.set(id, 0, 0));
  mesh.removeInstances(1, 3);

  const firstReuse = [];
  mesh.addInstances(1, (_instance, id) => firstReuse.push(id));
  assert.deepEqual(firstReuse, [3], 'the most recently freed id is reused first');

  mesh.setVisibilityAt(2, false);
  mesh.perObjectFrustumCulled = false;
  mesh.performFrustumCulling(camera);

  assert.equal(mesh.instancesCount, 4);
  assert.equal(mesh.count, 3);
  assert.deepEqual(Array.from(mesh.instanceIndex.array.subarray(0, mesh.count)), [0, 3, 4]);

  const secondReuse = [];
  mesh.addInstances(1, (_instance, id) => secondReuse.push(id));
  assert.deepEqual(secondReuse, [1]);

  mesh.performFrustumCulling(camera);
  assert.equal(mesh.count, 4);
  assert.deepEqual(Array.from(mesh.instanceIndex.array.subarray(0, mesh.count)), [0, 1, 3, 4]);

  mesh.dispose();
});

test('matrix and color storage keep CPU backing arrays and coalesce dirty uploads', () => {
  const mesh = createMesh();
  mesh.addInstances(8);

  const matrixStorage = mesh.matricesTexture;
  assert.ok(matrixStorage._data instanceof Float32Array);
  assert.strictEqual(matrixStorage.image.data, matrixStorage._data);
  assert.equal(matrixStorage.flush(8), true);
  matrixStorage.attribute.clearUpdateRanges();

  const firstMatrix = new Matrix4().makeTranslation(2, 3, 4);
  const secondMatrix = new Matrix4().makeScale(5, 6, 7);
  mesh.setMatrixAt(1, firstMatrix);
  mesh.setMatrixAt(6, secondMatrix);

  assert.deepEqual(Array.from(matrixStorage._data.subarray(16, 32)), firstMatrix.toArray());
  assert.deepEqual(Array.from(matrixStorage._data.subarray(96, 112)), secondMatrix.toArray());
  assert.equal(matrixStorage.flush(8), true);
  assertSingleUpdateRange(matrixStorage.attribute, 16, 96);
  assert.equal(matrixStorage.flush(8), false, 'a flushed dirty range is not re-enqueued');

  matrixStorage.attribute.clearUpdateRanges();
  mesh.setMatrixAt(6, firstMatrix);
  assert.equal(matrixStorage.flush(4), false, 'a dirty tail remains pending when outside the active prefix');
  assert.equal(matrixStorage.flush(8), true);
  assertSingleUpdateRange(matrixStorage.attribute, 96, 16);

  mesh.setColorAt(0, 0xffffff);
  const colorStorage = mesh.colorsTexture;
  assert.ok(colorStorage._data instanceof Float32Array);
  assert.strictEqual(colorStorage.image.data, colorStorage._data);
  assert.equal(colorStorage.flush(8), true);
  colorStorage.attribute.clearUpdateRanges();

  mesh.setColorAt(1, 0xff0000);
  mesh.setColorAt(6, 0x00ff00);

  assert.deepEqual(Array.from(colorStorage._data.subarray(4, 8)), [1, 0, 0, 1]);
  assert.deepEqual(Array.from(colorStorage._data.subarray(24, 28)), [0, 1, 0, 1]);
  assert.equal(colorStorage.flush(8), true);
  assertSingleUpdateRange(colorStorage.attribute, 4, 24);
  assert.equal(colorStorage.flush(8), false);

  mesh.dispose();
});

test('automatic growth preserves payloads, replaces GPU attributes, and rebuilds material nodes', () => {
  const mesh = createMesh(2);
  const firstMatrix = new Matrix4().makeTranslation(11, 12, 13);
  const secondMatrix = new Matrix4().makeTranslation(21, 22, 23);

  mesh.addInstances(2);
  mesh.setMatrixAt(0, firstMatrix);
  mesh.setMatrixAt(1, secondMatrix);
  mesh.setColorAt(0, 0x336699);
  mesh.setColorAt(1, 0xcc8844);
  const firstColorData = Array.from(mesh.colorsTexture._data.subarray(0, 4));
  const secondColorData = Array.from(mesh.colorsTexture._data.subarray(4, 8));

  const oldIndexAttribute = mesh.instanceIndex.attribute;
  const oldShadowIndex = mesh.getInstanceIndexForPass(true);
  const oldShadowIndexAttribute = oldShadowIndex.attribute;
  const oldMatrixAttribute = mesh.matricesTexture.attribute;
  const oldColorAttribute = mesh.colorsTexture.attribute;
  const oldPositionNode = mesh.material.positionNode;
  const oldColorNode = mesh.material.colorNode;
  const indexDisposeCount = disposeCounter(oldIndexAttribute);
  const shadowIndexDisposeCount = disposeCounter(oldShadowIndexAttribute);
  const matrixDisposeCount = disposeCounter(oldMatrixAttribute);
  const colorDisposeCount = disposeCounter(oldColorAttribute);

  mesh.addInstances(1, (instance) => instance.position.set(31, 32, 33));

  assert.ok(mesh.capacity > 2);
  assert.equal(mesh.instancesCount, 3);
  assert.ok(mesh.getMatrixAt(0, new Matrix4()).equals(firstMatrix));
  assert.ok(mesh.getMatrixAt(1, new Matrix4()).equals(secondMatrix));
  assert.deepEqual(Array.from(mesh.colorsTexture._data.subarray(0, 4)), firstColorData);
  assert.deepEqual(Array.from(mesh.colorsTexture._data.subarray(4, 8)), secondColorData);

  assert.notStrictEqual(mesh.instanceIndex.attribute, oldIndexAttribute);
  assert.notStrictEqual(mesh.getInstanceIndexForPass(true), mesh.instanceIndex);
  assert.notStrictEqual(mesh.getInstanceIndexForPass(true).attribute, oldShadowIndexAttribute);
  assert.notStrictEqual(mesh.matricesTexture.attribute, oldMatrixAttribute);
  assert.notStrictEqual(mesh.colorsTexture.attribute, oldColorAttribute);
  assert.equal(indexDisposeCount(), 1);
  assert.equal(shadowIndexDisposeCount(), 1);
  assert.equal(matrixDisposeCount(), 1);
  assert.equal(colorDisposeCount(), 1);

  assert.notStrictEqual(mesh.material.positionNode, oldPositionNode);
  assert.notStrictEqual(mesh.material.colorNode, oldColorNode);
  assert.strictEqual(mesh.instanceIndex.attribute.array, mesh.instanceIndex.array);
  assert.strictEqual(mesh.matricesTexture.attribute.array, mesh.matricesTexture._data);
  assert.strictEqual(mesh.colorsTexture.attribute.array, mesh.colorsTexture._data);
  assert.equal(mesh.instanceIndex.attribute.count, mesh.capacity);
  assert.equal(mesh.getInstanceIndexForPass(true).attribute.count, mesh.capacity);
  assert.equal(mesh.matricesTexture.attribute.count, mesh.capacity);
  assert.equal(mesh.colorsTexture.attribute.count, mesh.capacity);
  assert.equal(
    mesh.instanceIndex.array.subarray(2).every((id, offset) => id === offset + 2),
    true,
    'the newly allocated visible-index tail is initialized to stable identity ids'
  );

  mesh.dispose();
});

test('geometry replacement rebuilds the material position graph across normal layouts', () => {
  const mesh = createMesh();
  const withNormalsPositionNode = mesh.material.positionNode;

  const withoutNormals = new BufferGeometry();
  withoutNormals.setAttribute('position', new Float32BufferAttribute([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0
  ], 3));
  mesh.geometry = withoutNormals;
  const withoutNormalsPositionNode = mesh.material.positionNode;

  assert.strictEqual(mesh.geometry, withoutNormals);
  assert.notStrictEqual(withoutNormalsPositionNode, withNormalsPositionNode);

  const withNormalsAgain = new BoxGeometry(2, 2, 2);
  mesh.geometry = withNormalsAgain;
  assert.strictEqual(mesh.geometry, withNormalsAgain);
  assert.notStrictEqual(mesh.material.positionNode, withoutNormalsPositionNode);

  mesh.dispose();
  withoutNormals.dispose();
  withNormalsAgain.dispose();
});

test('meshes sharing a source material own separate material graphs and buffers', () => {
  const sourceMaterial = new MeshStandardMaterial({ color: 0x78a341 });
  const sourcePositionNode = sourceMaterial.positionNode;
  const sourceColorNode = sourceMaterial.colorNode;
  const first = createMesh(4, sourceMaterial);
  const second = createMesh(4, sourceMaterial);

  assert.notStrictEqual(first.material, sourceMaterial);
  assert.notStrictEqual(second.material, sourceMaterial);
  assert.notStrictEqual(first.material, second.material);
  assert.notStrictEqual(first.material.positionNode, second.material.positionNode);
  assert.strictEqual(sourceMaterial.positionNode, sourcePositionNode);
  assert.strictEqual(sourceMaterial.colorNode, sourceColorNode);

  assert.notStrictEqual(first.instanceIndex.attribute, second.instanceIndex.attribute);
  assert.notStrictEqual(first.matricesTexture.attribute, second.matricesTexture.attribute);

  first.setColorAt(0, 0xff0000);
  second.setColorAt(0, 0x00ff00);
  assert.notStrictEqual(first.colorsTexture.attribute, second.colorsTexture.attribute);
  assert.notStrictEqual(first.material.colorNode, second.material.colorNode);
  assert.notDeepEqual(
    Array.from(first.colorsTexture._data.subarray(0, 4)),
    Array.from(second.colorsTexture._data.subarray(0, 4))
  );

  first.dispose();
  second.dispose();
  sourceMaterial.dispose();
});

test('direct material replacement rebuilds owned TSL graphs without mutating caller materials', () => {
  const initialSource = new MeshStandardMaterial({ color: 0x78a341 });
  const mesh = createMesh(4, initialSource);
  mesh.setColorAt(0, 0xffffff);

  const previousOwnedMaterial = mesh.material;
  const previousPositionNode = previousOwnedMaterial.positionNode;
  const previousColorNode = previousOwnedMaterial.colorNode;
  const previousDisposeCount = disposeCounter(previousOwnedMaterial);
  const initialSourceDisposeCount = disposeCounter(initialSource);

  const replacementSource = new MeshStandardMaterial({ color: 0x335577 });
  const replacementBasePositionNode = vec3(4, 5, 6);
  const replacementBaseColorNode = vec4(0.25, 0.5, 0.75, 1);
  replacementSource.positionNode = replacementBasePositionNode;
  replacementSource.colorNode = replacementBaseColorNode;
  const replacementSourceDisposeCount = disposeCounter(replacementSource);

  mesh.material = replacementSource;

  assert.equal(previousDisposeCount(), 1);
  assert.notStrictEqual(mesh.material, previousOwnedMaterial);
  assert.notStrictEqual(mesh.material, replacementSource);
  assert.notStrictEqual(mesh.material.positionNode, previousPositionNode);
  assert.notStrictEqual(mesh.material.colorNode, previousColorNode);
  assert.notStrictEqual(mesh.material.positionNode, replacementBasePositionNode);
  assert.notStrictEqual(mesh.material.colorNode, replacementBaseColorNode);
  assert.strictEqual(replacementSource.positionNode, replacementBasePositionNode);
  assert.strictEqual(replacementSource.colorNode, replacementBaseColorNode);
  assert.equal(initialSource.positionNode, undefined);
  assert.equal(initialSource.colorNode, undefined);
  assert.equal(initialSourceDisposeCount(), 0);
  assert.equal(replacementSourceDisposeCount(), 0);

  const replacementOwnedDisposeCount = disposeCounter(mesh.material);
  mesh.dispose();
  assert.equal(previousDisposeCount(), 1);
  assert.equal(replacementOwnedDisposeCount(), 1);
  assert.equal(initialSourceDisposeCount(), 0);
  assert.equal(replacementSourceDisposeCount(), 0);

  initialSource.dispose();
  replacementSource.dispose();
});

test('invalid material-array replacement is atomic and creates no partial owned clones', () => {
  const initialSource = new MeshStandardMaterial({ color: 0x446688 });
  const mesh = createMesh(4, initialSource);
  mesh.setColorAt(0, 0xffffff);

  const previousOwnedMaterial = mesh.material;
  const previousPositionNode = previousOwnedMaterial.positionNode;
  const previousColorNode = previousOwnedMaterial.colorNode;
  const previousDisposeCount = disposeCounter(previousOwnedMaterial);

  const validSource = new MeshStandardMaterial({ color: 0x88aa44 });
  const validSourceClone = validSource.clone.bind(validSource);
  let partialCloneCount = 0;
  let partialCloneDisposeCount = 0;
  validSource.clone = () => {
    partialCloneCount++;
    const clone = validSourceClone();
    clone.addEventListener('dispose', () => partialCloneDisposeCount++);
    return clone;
  };

  const invalidSource = new MeshStandardMaterial();
  invalidSource.onBeforeCompile = () => {};

  assert.throws(
    () => {
      mesh.material = [validSource, invalidSource];
    },
    /cannot run a GLSL onBeforeCompile customization/
  );

  assert.strictEqual(mesh.material, previousOwnedMaterial);
  assert.strictEqual(mesh.material.positionNode, previousPositionNode);
  assert.strictEqual(mesh.material.colorNode, previousColorNode);
  assert.equal(previousDisposeCount(), 0);
  assert.equal(partialCloneCount, 0, 'all sources are validated before any owned clone is created');
  assert.equal(partialCloneDisposeCount, 0);
  assert.doesNotThrow(() => mesh.setColorAt(1, 0xff00ff));
  assert.deepEqual(Array.from(mesh.colorsTexture._data.subarray(4, 8)), [1, 0, 1, 1]);

  mesh.dispose();
  assert.equal(previousDisposeCount(), 1);
  initialSource.dispose();
  validSource.dispose();
  invalidSource.dispose();
});

test('LOD children use the WebGPU backend with private indexes and shared payload storage', () => {
  const parent = createMesh(6);
  parent.addInstances(2);
  parent.setColorAt(0, 0xabcdef);

  const lodMaterial = new MeshStandardMaterial({ color: 0x224466 });
  parent.addLOD(new BoxGeometry(0.5, 0.5, 0.5), lodMaterial, 12);

  assert.equal(parent.LODinfo.render.levels.length, 2);
  const child = parent.LODinfo.render.levels[1].object;

  assert.ok(child instanceof WebGPUInstancedMesh2);
  assert.notStrictEqual(child.instanceIndex, parent.instanceIndex);
  assert.notStrictEqual(child.instanceIndex.attribute, parent.instanceIndex.attribute);
  assert.strictEqual(child.matricesTexture, parent.matricesTexture);
  assert.strictEqual(child.colorsTexture, parent.colorsTexture);
  assert.strictEqual(child.availabilityArray, parent.availabilityArray);
  assert.notStrictEqual(child.material, lodMaterial);
  assert.equal(child.capacity, parent.capacity);

  parent.dispose();
  lodMaterial.dispose();
});

test('removeLOD disposes and detaches WebGPU child resources exactly once', () => {
  const parent = createMesh(6);
  const childGeometry = new BoxGeometry(0.5, 0.5, 0.5);
  const sourceMaterial = new MeshStandardMaterial({ color: 0x224466 });
  parent.addLOD(childGeometry, sourceMaterial, 12);

  const child = parent.LODinfo.render.levels[1].object;
  const ownedMaterial = child.material;
  const childDisposeCount = disposeCounter(child);
  const indexDisposeCount = disposeCounter(child.instanceIndex.attribute);
  const materialDisposeCount = disposeCounter(ownedMaterial);
  const geometryDisposeCount = disposeCounter(childGeometry);
  const sourceMaterialDisposeCount = disposeCounter(sourceMaterial);

  parent.removeLOD(1, true);

  assert.equal(parent.LODinfo.render, null);
  assert.deepEqual(parent.LODinfo.objects, [parent]);
  assert.equal(parent.children.includes(child), false);
  assert.equal(child.parent, null);
  assert.equal(childDisposeCount(), 1);
  assert.equal(indexDisposeCount(), 1);
  assert.equal(materialDisposeCount(), 1);
  assert.equal(geometryDisposeCount(), 1);
  assert.equal(sourceMaterialDisposeCount(), 0);

  parent.dispose();
  child.dispose();
  assert.equal(childDisposeCount(), 1);
  assert.equal(indexDisposeCount(), 1);
  assert.equal(materialDisposeCount(), 1);
  assert.equal(geometryDisposeCount(), 1);

  sourceMaterial.dispose();
});

test('unsupported WebGPU APIs fail explicitly instead of using WebGL storage paths', () => {
  assert.throws(
    () => new WebGPUInstancedMesh2(new BoxGeometry(), new ShaderMaterial()),
    /does not support ShaderMaterial/
  );

  const mesh = createMesh();
  assert.throws(() => mesh.initUniformsPerInstance({}), /does not support per-instance uniforms yet/);
  assert.throws(() => mesh.initSkeleton(new Skeleton()), /does not support instanced skinning yet/);
  assert.throws(() => mesh.setMorphAt(0, new Mesh()), /does not support per-instance morph targets yet/);
  assert.throws(() => mesh.clone(), /does not support clone\/copy yet/);
  assert.throws(() => mesh.copy(mesh), /does not support clone\/copy yet/);

  mesh.dispose();
});

test('classic GLSL onBeforeCompile customization reports the required TSL migration', () => {
  const material = new MeshStandardMaterial();
  material.onBeforeCompile = () => {};

  assert.throws(
    () => new WebGPUInstancedMesh2(new BoxGeometry(), material),
    (error) => {
      assert.match(error.message, /cannot run a GLSL onBeforeCompile customization/);
      assert.match(error.message, /Port the material customization to NodeMaterial\/TSL/);
      return true;
    }
  );

  material.dispose();
});

test('dispose releases owned storage attributes and materials exactly once', () => {
  const sourceMaterial = new MeshStandardMaterial();
  const mesh = createMesh(4, sourceMaterial);
  mesh.addInstances(1);
  mesh.setColorAt(0, 0x123456);

  const ownedMaterial = mesh.material;
  const indexAttribute = mesh.instanceIndex.attribute;
  const matrixAttribute = mesh.matricesTexture.attribute;
  const colorAttribute = mesh.colorsTexture.attribute;

  const meshDisposeCount = disposeCounter(mesh);
  const sourceMaterialDisposeCount = disposeCounter(sourceMaterial);
  const ownedMaterialDisposeCount = disposeCounter(ownedMaterial);
  const indexDisposeCount = disposeCounter(indexAttribute);
  const matrixDisposeCount = disposeCounter(matrixAttribute);
  const colorDisposeCount = disposeCounter(colorAttribute);

  mesh.dispose();
  mesh.dispose();

  assert.equal(meshDisposeCount(), 1);
  assert.equal(sourceMaterialDisposeCount(), 0, 'the caller-owned source material is not disposed');
  assert.equal(ownedMaterialDisposeCount(), 1);
  assert.equal(indexDisposeCount(), 1);
  assert.equal(matrixDisposeCount(), 1);
  assert.equal(colorDisposeCount(), 1);
  assert.deepEqual(indexAttribute.updateRanges, []);
  assert.deepEqual(matrixAttribute.updateRanges, []);
  assert.deepEqual(colorAttribute.updateRanges, []);

  sourceMaterial.dispose();
});
