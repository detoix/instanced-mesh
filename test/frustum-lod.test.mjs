import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BoxGeometry,
  MeshStandardMaterial,
  PerspectiveCamera,
  WebGPUCoordinateSystem
} from 'three/webgpu';

import { InstancedMesh2 } from '../dist/build/webgpu.js';

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const webGPUCamera = ({ near = 0.1, far = 200 } = {}) => {
  const camera = new PerspectiveCamera(60, 1, near, far);
  camera.coordinateSystem = WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const createLODMesh = (geometry = new BoxGeometry(0.25, 0.25, 0.25)) => {
  const sourceMaterials = [
    new MeshStandardMaterial({ color: 0x446622 }),
    new MeshStandardMaterial({ color: 0x224466 })
  ];
  const lodGeometry = geometry.clone();
  const mesh = new InstancedMesh2(geometry, sourceMaterials[0], {
    capacity: 8
  });
  mesh.addLOD(lodGeometry, sourceMaterials[1], 5);
  return { mesh, lodGeometry, sourceMaterials };
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const disposeFixture = ({ mesh, lodGeometry, sourceMaterials }) => {
  mesh.dispose();
  mesh.geometry.dispose();
  lodGeometry.dispose();
  for (const material of sourceMaterials) material.dispose();
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
const addShadowLevels = (fixture) => {
  const shadowGeometry = fixture.mesh.geometry.clone();
  fixture.mesh.addShadowLOD(fixture.mesh.geometry);
  fixture.mesh.addShadowLOD(shadowGeometry, 5);
  return shadowGeometry;
};

test('LOD classification honors disabled per-object frustum culling', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const camera = webGPUCamera();

  try {
    mesh.addInstances(3, (instance, index) => {
      if (index === 0) instance.position.set(0, 0, -2);
      if (index === 1) instance.position.set(100, 0, -2);
      if (index === 2) instance.position.set(0, 0, -8);
    });
    mesh.setVisibilityAt(2, false);
    mesh.perObjectFrustumCulled = false;

    mesh.performFrustumCulling(camera);

    const [near, far] = mesh.LODinfo.render.levels;
    assert.equal(near.object.count, 1);
    assert.equal(near.object.instanceIndex.array[0], 0);
    assert.equal(far.object.count, 1);
    assert.equal(far.object.instanceIndex.array[0], 1);

    // Re-enabling the feature proves that the off-frustum far instance was
    // retained because of the flag, not because its bounds were accidentally
    // considered visible.
    mesh.perObjectFrustumCulled = true;
    mesh.performFrustumCulling(camera);
    assert.equal(near.object.count, 1);
    assert.equal(far.object.count, 0);
  } finally {
    disposeFixture(fixture);
  }
});

test('LOD frustum extraction uses the WebGPU camera coordinate system', () => {
  const fixture = createLODMesh(new BoxGeometry(0.001, 0.001, 0.001));
  const { mesh } = fixture;
  const camera = webGPUCamera({ near: 0.1, far: 20 });

  try {
    mesh.addInstances(2, (instance, index) => {
      instance.position.set(0, 0, index === 0 ? -0.05 : -1);
    });
    mesh.perObjectFrustumCulled = true;

    mesh.performFrustumCulling(camera);

    const [near, far] = mesh.LODinfo.render.levels;
    assert.equal(near.object.count, 1);
    assert.equal(near.object.instanceIndex.array[0], 1);
    assert.equal(far.object.count, 0);
  } finally {
    disposeFixture(fixture);
  }
});

test('LOD hysteresis is applied before squaring the distance threshold', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;

  try {
    mesh.updateLOD(1, 10, 0.2);
    const levels = mesh.LODinfo.render.levels;

    // Ten metres with 20% hysteresis switches at eight metres. Both the
    // supplied distance and stored threshold are squared internally.
    assert.equal(mesh.getObjectLODIndexForDistance(levels, 7.999 ** 2), 0);
    assert.equal(mesh.getObjectLODIndexForDistance(levels, 8 ** 2), 1);
  } finally {
    disposeFixture(fixture);
  }
});

test('a resolver explicitly selects linear main and shadow LODs', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const shadowGeometry = addShadowLevels(fixture);
  const mainCamera = webGPUCamera();
  const shadowCamera = webGPUCamera();
  const resolutions = [];
  const enteredLevels = [];

  try {
    mesh.addInstances(1, (instance) => instance.position.set(0, 0, -2));
    mesh.perObjectFrustumCulled = false;
    mesh.resolveLODIndex = (
      index,
      camera,
      cameraLOD,
      computedIndex,
      isShadowPass
    ) => {
      resolutions.push({
        index,
        camera,
        cameraLOD,
        computedIndex,
        isShadowPass
      });
      return 1;
    };
    mesh.onFrustumEnter = (_index, _camera, _cameraLOD, levelIndex) => {
      enteredLevels.push(levelIndex);
      return true;
    };

    mesh.performFrustumCulling(mainCamera);
    const [mainNear, mainFar] = mesh.LODinfo.render.levels;
    assert.equal(mainNear.object.count, 0);
    assert.equal(mainFar.object.count, 1);
    assert.equal(mainFar.object.instanceIndex.array[0], 0);

    mesh.performFrustumCulling(shadowCamera, mainCamera);
    const [shadowNear, shadowFar] = mesh.LODinfo.shadowRender.levels;
    assert.equal(shadowNear.object.count, 0);
    assert.equal(shadowFar.object.count, 1);
    assert.equal(shadowFar.object.instanceIndex.array[0], 0);

    assert.equal(resolutions.length, 2);
    assert.deepEqual(
      resolutions.map(({ index, computedIndex }) => ({ index, computedIndex })),
      [
        { index: 0, computedIndex: 0 },
        { index: 0, computedIndex: 0 }
      ]
    );
    assert.strictEqual(resolutions[0].camera, mainCamera);
    assert.strictEqual(resolutions[0].cameraLOD, mainCamera);
    assert.strictEqual(resolutions[1].camera, shadowCamera);
    assert.strictEqual(resolutions[1].cameraLOD, mainCamera);
    assert.equal(resolutions[0].isShadowPass, false);
    assert.equal(resolutions[1].isShadowPass, true);
    assert.deepEqual(enteredLevels, [1, 1]);
  } finally {
    shadowGeometry.dispose();
    disposeFixture(fixture);
  }
});

test('WebGPU shadow callbacks explicitly select shadow LODs when both camera arguments match', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const shadowGeometry = addShadowLevels(fixture);
  const shadowCamera = webGPUCamera();
  const resolutions = [];

  try {
    mesh.addInstances(1, (instance) => instance.position.set(0, 0, -2));
    mesh.perObjectFrustumCulled = false;
    mesh.resolveLODIndex = (
      _index,
      camera,
      cameraLOD,
      _computedIndex,
      isShadowPass
    ) => {
      resolutions.push({ camera, cameraLOD, isShadowPass });
      return 1;
    };

    const renderer = { info: { calls: 41 } };
    mesh.onBeforeShadow(
      renderer,
      mesh,
      shadowCamera,
      shadowCamera,
      mesh.geometry,
      mesh.material,
      null
    );
    mesh.onAfterShadow(
      renderer,
      mesh,
      shadowCamera,
      shadowCamera,
      mesh.geometry,
      mesh.material,
      null
    );

    const [mainNear, mainFar] = mesh.LODinfo.render.levels;
    const [shadowNear, shadowFar] = mesh.LODinfo.shadowRender.levels;
    assert.equal(mainNear.object.count, 0);
    assert.equal(mainFar.object.count, 0);
    assert.equal(shadowNear.object.count, 0);
    assert.equal(shadowFar.object.count, 1);
    assert.equal(shadowFar.object.instanceIndex.array[0], 0);
    assert.deepEqual(resolutions, [{
      camera: shadowCamera,
      cameraLOD: shadowCamera,
      isShadowPass: true
    }]);
  } finally {
    shadowGeometry.dispose();
    disposeFixture(fixture);
  }
});

test('WebGPU keeps main and shadow LOD index bindings independent', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const shadowGeometry = addShadowLevels(fixture);
  const camera = webGPUCamera();

  try {
    mesh.addInstances(2, (instance, index) => {
      instance.position.set(0, 0, -2 - index);
    });
    mesh.perObjectFrustumCulled = false;
    mesh.resolveLODIndex = (index, _camera, _cameraLOD, _computed, shadow) => (
      shadow ? 0 : index
    );

    mesh.performFrustumCulling(camera);
    const [mainNear, mainFar] = mesh.LODinfo.render.levels;
    const mainNearIndex = mainNear.object.getInstanceIndexForPass(false);
    const mainFarIndex = mainFar.object.getInstanceIndexForPass(false);
    assert.deepEqual(Array.from(mainNearIndex.array.subarray(0, 1)), [0]);
    assert.deepEqual(Array.from(mainFarIndex.array.subarray(0, 1)), [1]);

    mesh.performFrustumCulling(camera, camera, true);
    const [shadowNear] = mesh.LODinfo.shadowRender.levels;
    const shadowNearIndex = shadowNear.object.getInstanceIndexForPass(true);

    assert.notStrictEqual(shadowNearIndex, shadowNear.object.getInstanceIndexForPass(false));
    assert.equal(shadowNear.object.count, 2);
    assert.deepEqual(Array.from(shadowNearIndex.array.subarray(0, 2)), [0, 1]);
    assert.deepEqual(
      Array.from(mainFarIndex.array.subarray(0, 1)),
      [1],
      'shadow classification does not overwrite the color-pass visible list'
    );
  } finally {
    shadowGeometry.dispose();
    disposeFixture(fixture);
  }
});

test('a main LOD child restores color counts when it renders before the parent after shadows', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const shadowGeometry = addShadowLevels(fixture);
  const camera = webGPUCamera();
  const renderer = { info: { calls: 77 } };

  try {
    mesh.addInstances(1, (instance) => instance.position.set(0, 0, -2));
    mesh.perObjectFrustumCulled = false;
    mesh.resolveLODIndex = (
      _index,
      _camera,
      _cameraLOD,
      _computedIndex,
      isShadowPass
    ) => isShadowPass ? 0 : 1;

    mesh.onBeforeShadow(
      renderer,
      mesh,
      camera,
      camera,
      mesh.geometry,
      mesh.material,
      null
    );
    mesh.onAfterShadow(
      renderer,
      mesh,
      camera,
      camera,
      mesh.geometry,
      mesh.material,
      null
    );

    const [mainNear, mainFar] = mesh.LODinfo.render.levels;
    assert.deepEqual(
      mesh.LODinfo.render.levels.map(({ object }) => object.count),
      [1, 0],
      'shadow classification remains in the shared CPU counts'
    );

    mainFar.object.onBeforeRender(
      renderer,
      mesh,
      camera,
      mainFar.object.geometry,
      mainFar.object.material,
      null
    );

    assert.equal(mainNear.object.count, 0);
    assert.equal(mainFar.object.count, 1);
    assert.equal(mainFar.object.instanceIndex.array[0], 0);
  } finally {
    shadowGeometry.dispose();
    disposeFixture(fixture);
  }
});

test('a resolver applies to sorted BVH main and unsorted BVH shadow classification', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const shadowGeometry = addShadowLevels(fixture);
  const mainCamera = webGPUCamera();
  const shadowCamera = webGPUCamera();
  const passes = [];

  try {
    mesh.addInstances(1, (instance) => instance.position.set(0, 0, -2));
    mesh.computeBVH();
    mesh.perObjectFrustumCulled = true;
    mesh.sortObjects = true;
    mesh.resolveLODIndex = (
      _index,
      _camera,
      _cameraLOD,
      _computedIndex,
      isShadowPass
    ) => {
      passes.push(isShadowPass ? 'shadow' : 'main');
      return 1;
    };

    mesh.performFrustumCulling(mainCamera);
    assert.equal(mesh.LODinfo.render.levels[0].object.count, 0);
    assert.equal(mesh.LODinfo.render.levels[1].object.count, 1);

    mesh.performFrustumCulling(shadowCamera, mainCamera);
    assert.equal(mesh.LODinfo.shadowRender.levels[0].object.count, 0);
    assert.equal(mesh.LODinfo.shadowRender.levels[1].object.count, 1);
    assert.deepEqual(passes, ['main', 'shadow']);
  } finally {
    shadowGeometry.dispose();
    disposeFixture(fixture);
  }
});

test('invalid resolver results fall back to distance classification', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const camera = webGPUCamera();

  try {
    mesh.addInstances(1, (instance) => instance.position.set(0, 0, -8));
    mesh.perObjectFrustumCulled = false;

    for (const invalidResult of [-1, 2, 0.5, Number.NaN]) {
      mesh.resolveLODIndex = () => invalidResult;
      mesh.performFrustumCulling(camera);
      const [near, far] = mesh.LODinfo.render.levels;
      assert.equal(near.object.count, 0, `accepted ${String(invalidResult)}`);
      assert.equal(
        far.object.count,
        1,
        `did not fall back for ${String(invalidResult)}`
      );
    }
  } finally {
    disposeFixture(fixture);
  }
});

test('shadow and main pass cache keys remain distinct within one render', () => {
  const fixture = createLODMesh();
  const { mesh } = fixture;
  const mainCamera = webGPUCamera();
  const shadowCamera = webGPUCamera();
  const renderCall = 27;

  try {
    assert.equal(
      mesh.frustumCullingAlreadyPerformed(renderCall, mainCamera, null),
      false
    );
    assert.equal(
      mesh.frustumCullingAlreadyPerformed(renderCall, mainCamera, null),
      true
    );
    assert.equal(
      mesh.frustumCullingAlreadyPerformed(renderCall, mainCamera, shadowCamera),
      false
    );
    assert.equal(
      mesh.frustumCullingAlreadyPerformed(renderCall, mainCamera, shadowCamera),
      true
    );
    assert.equal(
      mesh.frustumCullingAlreadyPerformed(renderCall, mainCamera, null),
      false
    );
  } finally {
    disposeFixture(fixture);
  }
});
