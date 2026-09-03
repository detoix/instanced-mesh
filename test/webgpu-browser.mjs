import {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  MeshBasicMaterial,
  MeshStandardMaterial,
  OrthographicCamera,
  PerspectiveCamera,
  PlaneGeometry,
  Group,
  RenderTarget,
  Scene,
  Vector3,
  WebGPURenderer
} from 'three/webgpu';
import { float, vec3 } from 'three/tsl';
import {
  InstancedMesh2,
  setWebGPUInstancePositionNode
} from '../dist/build/webgpu.js';

const result = {
  backend: null,
  count: 0,
  drawCalls: 0,
  error: null,
  validationError: null,
  singleInstanceBindings: null,
  shadowDeformation: null,
  gpuDriven: null
};
window.__webgpuResult = result;

try {
  const renderer = new WebGPURenderer({ antialias: true });
  window.__renderer = renderer;
  renderer.setPixelRatio(1);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  document.body.append(renderer.domElement);
  await renderer.init();
  renderer.info.autoReset = false;

  result.backend = renderer.backend.isWebGPUBackend ? 'webgpu' : 'fallback';
  const device = renderer.backend.device;
  device?.addEventListener('uncapturederror', (event) => {
    result.validationError = event.error?.message ?? String(event.error);
  });

  const scene = new Scene();
  scene.background = new Color(0x16202a);

  const camera = new PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 300);
  camera.position.set(24, 20, 30);
  camera.lookAt(0, 0, 0);

  scene.add(new AmbientLight(0xffffff, 1.2));
  const sun = new DirectionalLight(0xffffff, 3);
  sun.position.set(12, 20, 8);
  sun.castShadow = true;
  scene.add(sun);

  const material = new MeshStandardMaterial({ color: 0xffffff, roughness: 0.72 });
  const mesh = new InstancedMesh2(new BoxGeometry(0.34, 1, 0.34), material, { capacity: 4096 });
  mesh.perObjectFrustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.addInstances(4096, (instance, index) => {
    const x = index % 64;
    const z = Math.floor(index / 64);
    instance.position.set((x - 31.5) * 0.55, 0.5, (z - 31.5) * 0.55);
    instance.scale.y = 0.5 + ((index * 17) % 100) / 100;
    mesh.setColorAt(index, new Color().setHSL(0.24 + (index % 7) * 0.008, 0.7, 0.28 + (index % 5) * 0.025));
  });
  scene.add(mesh);

  // Regression: count === 1 used to let two structurally identical materials
  // share the first mesh's storage-buffer bindings. Render both meshes into a
  // target and verify that their independently stored transforms/colors land
  // on opposite sides.
  const bindingScene = new Scene();
  bindingScene.background = new Color(0x000000);
  const bindingCamera = new OrthographicCamera(-2, 2, 1, -1, 0.1, 10);
  bindingCamera.position.z = 5;
  const bindingGeometry = new PlaneGeometry(0.9, 0.9);
  const bindingMaterial = new MeshBasicMaterial({ color: 0xffffff });
  setWebGPUInstancePositionNode(
    bindingMaterial,
    ({ positionNode, instanceId }) => positionNode.add(
      vec3(0, float(instanceId).mul(0.6), 0)
    )
  );
  const leftMesh = new InstancedMesh2(bindingGeometry, bindingMaterial, { capacity: 1 });
  const rightMesh = new InstancedMesh2(bindingGeometry, bindingMaterial, { capacity: 2 });
  for (const singleMesh of [leftMesh, rightMesh]) singleMesh.perObjectFrustumCulled = false;
  leftMesh.addInstances(1, (instance) => instance.position.set(-1, -0.3, 0));
  rightMesh.addInstances(2, (instance) => instance.position.set(1, -0.3, 0));
  rightMesh.removeInstances(0);
  leftMesh.setColorAt(0, 0xff0000);
  rightMesh.setColorAt(1, 0x0000ff);
  bindingScene.add(leftMesh, rightMesh);

  const bindingTarget = new RenderTarget(128, 64, { depthBuffer: false });
  renderer.setRenderTarget(bindingTarget);
  for (let frame = 0; frame < 3; frame++) {
    renderer.render(bindingScene, bindingCamera);
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }
  const bindingPixels = await renderer.readRenderTargetPixelsAsync(bindingTarget, 0, 0, 128, 64);
  renderer.setRenderTarget(null);
  let redCount = 0;
  let blueCount = 0;
  let redX = 0;
  let blueX = 0;
  let redY = 0;
  let blueY = 0;
  for (let offset = 0; offset < bindingPixels.length; offset += 4) {
    const red = bindingPixels[offset];
    const green = bindingPixels[offset + 1];
    const blue = bindingPixels[offset + 2];
    const x = (offset / 4) % 128;
    const y = Math.floor((offset / 4) / 128);
    if (red > 50 && red > green * 1.5 && red > blue * 1.5) {
      redCount++;
      redX += x;
      redY += y;
    }
    if (blue > 50 && blue > green * 1.5 && blue > red * 1.5) {
      blueCount++;
      blueX += x;
      blueY += y;
    }
  }
  const redCentroid = redCount > 0 ? redX / redCount : null;
  const blueCentroid = blueCount > 0 ? blueX / blueCount : null;
  const redCentroidY = redCount > 0 ? redY / redCount : null;
  const blueCentroidY = blueCount > 0 ? blueY / blueCount : null;
  result.singleInstanceBindings = {
    redCount,
    blueCount,
    redCentroid,
    blueCentroid,
    redCentroidY,
    blueCentroidY,
    distinct:
      redCount > 20
      && blueCount > 20
      && redCentroid < 64
      && blueCentroid > 64
      && Math.abs(redCentroidY - blueCentroidY) > 12
  };
  if (!result.singleInstanceBindings.distinct) {
    throw new Error(`single-instance storage bindings collided: ${JSON.stringify(result.singleInstanceBindings)}`);
  }
  leftMesh.dispose();
  rightMesh.dispose();
  bindingTarget.dispose();
  bindingGeometry.dispose();
  bindingMaterial.dispose();

  // WebGPURenderer derives its shadow/depth override from the source
  // material's positionNode. Render two hidden one-instance casters from
  // directly overhead: only their shadows remain in the color target. The
  // right caster is displaced upward in the local-position factory, so an
  // angled sun must move its shadow along the ground relative to the left.
  const shadowScene = new Scene();
  shadowScene.background = new Color(0xffffff);
  const shadowCamera = new OrthographicCamera(-4, 4, 4, -4, 0.1, 20);
  shadowCamera.position.set(0, 8, 0);
  shadowCamera.up.set(0, 0, -1);
  shadowCamera.lookAt(0, 0, 0);
  shadowScene.add(new AmbientLight(0xffffff, 0.35));
  const shadowSun = new DirectionalLight(0xffffff, 3);
  shadowSun.position.set(0, 7, 5);
  shadowSun.castShadow = true;
  shadowSun.shadow.mapSize.set(256, 256);
  shadowSun.shadow.camera.left = -5;
  shadowSun.shadow.camera.right = 5;
  shadowSun.shadow.camera.top = 5;
  shadowSun.shadow.camera.bottom = -5;
  shadowSun.shadow.camera.near = 0.1;
  shadowSun.shadow.camera.far = 20;
  shadowScene.add(shadowSun, shadowSun.target);

  const shadowGroundGeometry = new PlaneGeometry(8, 8);
  const shadowGroundMaterial = new MeshStandardMaterial({ color: 0xffffff, roughness: 1 });
  const shadowGround = new (await import('three/webgpu')).Mesh(
    shadowGroundGeometry,
    shadowGroundMaterial
  );
  shadowGround.rotation.x = -Math.PI / 2;
  shadowGround.receiveShadow = true;
  shadowScene.add(shadowGround);

  const baselineGeometry = new BoxGeometry(0.8, 1, 0.8);
  const deformedGeometry = baselineGeometry.clone();
  const baselineMaterial = new MeshStandardMaterial({ color: 0xffffff, colorWrite: false });
  const deformedMaterial = new MeshStandardMaterial({ color: 0xffffff, colorWrite: false });
  setWebGPUInstancePositionNode(
    deformedMaterial,
    ({ positionNode }) => positionNode.add(vec3(0, 1, 0))
  );
  const baselineCaster = new InstancedMesh2(baselineGeometry, baselineMaterial, { capacity: 1 });
  const deformedCaster = new InstancedMesh2(deformedGeometry, deformedMaterial, { capacity: 1 });
  for (const caster of [baselineCaster, deformedCaster]) {
    caster.perObjectFrustumCulled = false;
    caster.castShadow = true;
  }
  baselineCaster.addInstances(1, (instance) => instance.position.set(-1.4, 0.5, 0));
  deformedCaster.addInstances(1, (instance) => instance.position.set(1.4, 0.5, 0));
  shadowScene.add(baselineCaster, deformedCaster);

  const shadowTarget = new RenderTarget(192, 192, { depthBuffer: true });
  renderer.setRenderTarget(shadowTarget);
  for (let frame = 0; frame < 4; frame++) {
    renderer.render(shadowScene, shadowCamera);
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }
  const shadowPixels = await renderer.readRenderTargetPixelsAsync(
    shadowTarget,
    0,
    0,
    192,
    192
  );
  renderer.setRenderTarget(null);
  const shadowStats = [
    { count: 0, y: 0 },
    { count: 0, y: 0 }
  ];
  for (let offset = 0; offset < shadowPixels.length; offset += 4) {
    const pixel = offset / 4;
    const x = pixel % 192;
    const y = Math.floor(pixel / 192);
    const luminance
      = shadowPixels[offset] * 0.2126
        + shadowPixels[offset + 1] * 0.7152
        + shadowPixels[offset + 2] * 0.0722;
    if (luminance >= 185) continue;
    const side = x < 96 ? 0 : 1;
    shadowStats[side].count++;
    shadowStats[side].y += y;
  }
  const baselineShadowY = shadowStats[0].count
    ? shadowStats[0].y / shadowStats[0].count
    : null;
  const deformedShadowY = shadowStats[1].count
    ? shadowStats[1].y / shadowStats[1].count
    : null;
  result.shadowDeformation = {
    baselinePixels: shadowStats[0].count,
    deformedPixels: shadowStats[1].count,
    baselineShadowY,
    deformedShadowY,
    distinct:
      shadowStats[0].count > 20
      && shadowStats[1].count > 20
      && Math.abs(baselineShadowY - deformedShadowY) > 8
  };
  if (!result.shadowDeformation.distinct) {
    throw new Error(
      `instance positionNode did not move the WebGPU shadow pass: ${JSON.stringify(result.shadowDeformation)}`
    );
  }

  baselineCaster.dispose();
  deformedCaster.dispose();
  baselineGeometry.dispose();
  deformedGeometry.dispose();
  baselineMaterial.dispose();
  deformedMaterial.dispose();
  shadowGroundGeometry.dispose();
  shadowGroundMaterial.dispose();
  shadowTarget.dispose();

  // GPU-driven culling. Every figure below is read back out of the indirect
  // draw commands the compute pass wrote, so it is evidence that the kernel
  // ran and that the draw consumed its result -- not that the CPU agreed with
  // itself. See docs/webgpu-architecture.md.
  const gpuScene = new Scene();
  gpuScene.background = new Color(0x101418);
  const gpuCamera = new PerspectiveCamera(60, 1, 0.1, 100);
  gpuCamera.position.set(0, 0, 12);
  gpuCamera.lookAt(0, 0, 0);
  gpuCamera.updateMatrixWorld(true);
  gpuScene.add(new AmbientLight(0xffffff, 1.5));

  const gpuGeometry = new BoxGeometry(0.4, 0.4, 0.4);
  const gpuMaterial = new MeshStandardMaterial({ color: 0xffcc44 });
  const gpuMesh = new InstancedMesh2(gpuGeometry, gpuMaterial, { capacity: 64, culling: 'gpu' });
  // Object-level culling would skip the draw entirely and leave the previous
  // frame's counts in the indirect buffer, which is not what is under test.
  gpuMesh.frustumCulled = false;
  // A 8x8 sheet at z = 0, comfortably inside the frustum.
  gpuMesh.addInstances(64, (instance, index) => {
    instance.position.set((index % 8) - 3.5, Math.floor(index / 8) - 3.5, 0);
  });
  gpuScene.add(gpuMesh);

  const gpuTarget = new RenderTarget(96, 96, { depthBuffer: true });
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
  const renderGpuScene = async (frames = 2) => {
    renderer.setRenderTarget(gpuTarget);
    for (let frame = 0; frame < frames; frame++) {
      renderer.render(gpuScene, gpuCamera);
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    }
    renderer.setRenderTarget(null);
  };

  const gpuDriven = { active: gpuMesh.gpuCullingActive };

  await renderGpuScene();
  gpuDriven.allVisible = (await gpuMesh.getVisibleCountsAsync())?.[0] ?? null;

  // Hidden instances must not survive compaction.
  for (let index = 0; index < 64; index += 2) gpuMesh.setVisibilityAt(index, false);
  await renderGpuScene();
  gpuDriven.halfHidden = (await gpuMesh.getVisibleCountsAsync())?.[0] ?? null;
  for (let index = 0; index < 64; index += 2) gpuMesh.setVisibilityAt(index, true);

  // Removed instances must not survive either, and their slots must come back.
  gpuMesh.removeInstances(0, 1, 2, 3);
  await renderGpuScene();
  gpuDriven.afterRemove = (await gpuMesh.getVisibleCountsAsync())?.[0] ?? null;
  gpuMesh.addInstances(4, (instance, id) => {
    instance.position.set((id % 8) - 3.5, Math.floor(id / 8) - 3.5, 0);
  });
  await renderGpuScene();
  gpuDriven.afterReadd = (await gpuMesh.getVisibleCountsAsync())?.[0] ?? null;

  // Turning the camera away must empty the visible list entirely.
  gpuCamera.position.set(0, 0, 12);
  gpuCamera.lookAt(0, 0, 40);
  gpuCamera.updateMatrixWorld(true);
  await renderGpuScene();
  gpuDriven.noneVisible = (await gpuMesh.getVisibleCountsAsync())?.[0] ?? null;

  // A camera that keeps only the right-hand columns must keep a strict subset.
  gpuCamera.position.set(3.6, 0, 3);
  gpuCamera.lookAt(3.6, 0, 0);
  gpuCamera.updateMatrixWorld(true);
  await renderGpuScene();
  gpuDriven.partiallyVisible = (await gpuMesh.getVisibleCountsAsync())?.[0] ?? null;

  gpuCamera.position.set(0, 0, 12);
  gpuCamera.lookAt(0, 0, 0);
  gpuCamera.updateMatrixWorld(true);
  await renderGpuScene();
  const gpuPixels = await (async () => {
    renderer.setRenderTarget(gpuTarget);
    renderer.render(gpuScene, gpuCamera);
    const pixels = await renderer.readRenderTargetPixelsAsync(gpuTarget, 0, 0, 96, 96);
    renderer.setRenderTarget(null);
    return pixels;
  })();
  let litPixels = 0;
  for (let offset = 0; offset < gpuPixels.length; offset += 4) {
    if (gpuPixels[offset] > 90 && gpuPixels[offset + 1] > 60) litPixels++;
  }
  gpuDriven.litPixels = litPixels;

  gpuMesh.dispose();
  gpuGeometry.dispose();
  gpuMaterial.dispose();

  // A material array draws one indirect command per geometry group. Without
  // the count-publish kernel only the first group would receive a non-zero
  // instance count, so the second material's color would never appear.
  const multiGeometry = new BoxGeometry(3, 3, 3);
  multiGeometry.clearGroups();
  multiGeometry.addGroup(0, 18, 0);
  multiGeometry.addGroup(18, 18, 1);
  const multiMaterials = [
    new MeshBasicMaterial({ color: 0xff0000 }),
    new MeshBasicMaterial({ color: 0x0000ff })
  ];
  const multiMesh = new InstancedMesh2(multiGeometry, multiMaterials, { capacity: 2, culling: 'gpu' });
  multiMesh.frustumCulled = false;
  // BoxGeometry's first group owns +X/-X/+Y and its second owns -Y/+Z/-Z. Yaw
  // the boxes so one face of each group is turned towards the camera.
  multiMesh.addInstances(2, (instance, index) => {
    instance.position.set(index * 4 - 2, 0, 0);
    instance.quaternion.setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 5);
  });
  gpuScene.add(multiMesh);
  await renderGpuScene(3);

  renderer.setRenderTarget(gpuTarget);
  renderer.render(gpuScene, gpuCamera);
  const multiPixels = await renderer.readRenderTargetPixelsAsync(gpuTarget, 0, 0, 96, 96);
  renderer.setRenderTarget(null);
  let redOnly = 0;
  let blueOnly = 0;
  for (let offset = 0; offset < multiPixels.length; offset += 4) {
    const red = multiPixels[offset];
    const green = multiPixels[offset + 1];
    const blue = multiPixels[offset + 2];
    if (red > 100 && green < 60 && blue < 60) redOnly++;
    if (blue > 100 && green < 60 && red < 60) blueOnly++;
  }
  gpuDriven.multiMaterial = { redOnly, blueOnly };
  gpuScene.remove(multiMesh);
  multiMesh.dispose();
  multiGeometry.dispose();
  for (const material of multiMaterials) material.dispose();

  // A transformed parent. The kernel tests object-space planes against the
  // instance matrices, so a mesh whose matrixWorld is not identity -- the
  // ordinary case for a mesh inside a scene graph -- must still cull exactly.
  const worldScene = new Scene();
  worldScene.background = new Color(0x101418);
  worldScene.add(new AmbientLight(0xffffff, 1.5));
  const worldGroup = new Group();
  worldGroup.position.set(120, -40, 60);
  worldGroup.rotation.set(0.3, Math.PI / 3, -0.2);
  worldGroup.scale.setScalar(2.5);
  worldScene.add(worldGroup);

  const worldGeometry = new BoxGeometry(0.3, 0.3, 0.3);
  const worldMaterial = new MeshBasicMaterial({ color: 0x66ff99 });
  const worldMesh = new InstancedMesh2(worldGeometry, worldMaterial, { capacity: 16, culling: 'gpu' });
  worldMesh.frustumCulled = false;
  // A line of instances along the group's local X axis.
  worldMesh.addInstances(16, (instance, index) => instance.position.set(index - 7.5, 0, 0));
  worldGroup.add(worldMesh);
  worldScene.updateMatrixWorld(true);

  const worldCamera = new PerspectiveCamera(60, 1, 0.1, 400);
  const worldTarget = new RenderTarget(64, 64, { depthBuffer: true });
  const localPoint = new Vector3();
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
  const aimWorldCamera = (localX, backOff) => {
    localPoint.set(localX, 0, 0).applyMatrix4(worldGroup.matrixWorld);
    worldCamera.position.copy(localPoint).add(new Vector3(0, 0, backOff));
    worldCamera.lookAt(localPoint);
    worldCamera.updateMatrixWorld(true);
  };
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
  const renderWorldScene = async () => {
    renderer.setRenderTarget(worldTarget);
    for (let frame = 0; frame < 2; frame++) {
      renderer.render(worldScene, worldCamera);
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    }
    renderer.setRenderTarget(null);
  };

  aimWorldCamera(0, 90);
  await renderWorldScene();
  gpuDriven.transformedAll = (await worldMesh.getVisibleCountsAsync())?.[0] ?? null;

  aimWorldCamera(0, 4);
  await renderWorldScene();
  gpuDriven.transformedNear = (await worldMesh.getVisibleCountsAsync())?.[0] ?? null;

  worldCamera.position.copy(localPoint).add(new Vector3(0, 0, 60));
  worldCamera.lookAt(localPoint.clone().add(new Vector3(0, 0, 400)));
  worldCamera.updateMatrixWorld(true);
  await renderWorldScene();
  gpuDriven.transformedNone = (await worldMesh.getVisibleCountsAsync())?.[0] ?? null;

  worldMesh.dispose();
  worldGeometry.dispose();
  worldMaterial.dispose();
  worldTarget.dispose();

  // LOD classification and an independent shadow pass, both GPU-side.
  const lodScene = new Scene();
  lodScene.background = new Color(0x101418);
  lodScene.add(new AmbientLight(0xffffff, 1.2));
  const lodSun = new DirectionalLight(0xffffff, 2);
  lodSun.position.set(0, 12, 0);
  lodSun.castShadow = true;
  lodSun.shadow.mapSize.set(256, 256);
  // Deliberately narrow: the far row of instances falls outside it, so the
  // shadow pass has to reach a different visible set from the main camera.
  lodSun.shadow.camera.left = -20;
  lodSun.shadow.camera.right = 20;
  lodSun.shadow.camera.top = 20;
  lodSun.shadow.camera.bottom = -20;
  lodSun.shadow.camera.far = 40;
  lodScene.add(lodSun, lodSun.target);

  const nearGeometry = new BoxGeometry(0.5, 0.5, 0.5);
  const farGeometry = new BoxGeometry(0.5, 0.5, 0.5);
  const lodMesh = new InstancedMesh2(
    nearGeometry,
    new MeshStandardMaterial({ color: 0x88ccff }),
    { capacity: 16, culling: 'gpu' }
  );
  lodMesh.addLOD(farGeometry, new MeshStandardMaterial({ color: 0xff8844 }), 10);
  lodMesh.castShadow = true;
  lodMesh.frustumCulled = false;
  // Eight instances within 10 units of the camera, eight well beyond it.
  lodMesh.addInstances(16, (instance, index) => {
    instance.position.set((index % 4) - 1.5, 0, index < 8 ? -2 : -30);
  });
  lodScene.add(lodMesh);

  // Three only renders a shadow map when something actually receives it.
  const lodGroundGeometry = new PlaneGeometry(120, 120);
  const lodGroundMaterial = new MeshStandardMaterial({ color: 0x334455, roughness: 1 });
  const lodGround = new (await import('three/webgpu')).Mesh(lodGroundGeometry, lodGroundMaterial);
  lodGround.rotation.x = -Math.PI / 2;
  lodGround.position.y = -1;
  lodGround.receiveShadow = true;
  lodScene.add(lodGround);

  const lodCamera = new PerspectiveCamera(75, 1, 0.1, 200);
  lodCamera.position.set(0, 0, 0);
  lodCamera.lookAt(0, 0, -1);
  lodCamera.updateMatrixWorld(true);

  const lodTarget = new RenderTarget(96, 96, { depthBuffer: true });
  renderer.setRenderTarget(lodTarget);
  for (let frame = 0; frame < 3; frame++) {
    renderer.render(lodScene, lodCamera);
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }
  renderer.setRenderTarget(null);

  gpuDriven.lodLevels = await lodMesh.getVisibleCountsAsync(false);
  gpuDriven.shadowLevels = await lodMesh.getVisibleCountsAsync(true);

  lodMesh.dispose();
  nearGeometry.dispose();
  farGeometry.dispose();
  lodGroundGeometry.dispose();
  lodGroundMaterial.dispose();
  lodTarget.dispose();
  gpuTarget.dispose();

  result.gpuDriven = gpuDriven;

  const ground = new (await import('three/webgpu')).Mesh(
    new PlaneGeometry(70, 70),
    new MeshStandardMaterial({ color: 0x243322, roughness: 1 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  for (let frame = 0; frame < 30; frame++) {
    renderer.render(scene, camera);
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }
  result.count = mesh.count;
  result.drawCalls = renderer.info.render.drawCalls;
  result.ready = true;
} catch (error) {
  result.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  result.ready = true;
} finally {
  document.documentElement.dataset.webgpuResult = JSON.stringify(result);
}
