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
  RenderTarget,
  Scene,
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
  shadowDeformation: null
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
