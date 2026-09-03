/**
 * Drives test/webgpu-browser.html in a real WebGPU browser and asserts the
 * results it publishes.
 *
 * The page is served from memory through Playwright request interception, so
 * this needs no dev server and still gets a secure origin -- `navigator.gpu`
 * is undefined on insecure ones.
 *
 * Run `vite build --config vite.test.config.js` first. Playwright is not a
 * dependency of this package: point NODE_PATH at an installation, or set
 * PLAYWRIGHT_MODULE to its entry point.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const PAGE_ORIGIN = 'https://instanced-mesh.test';
const DIST = fileURLToPath(new URL('../dist/test-browser/', import.meta.url));

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.map': 'application/json'
};

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
async function loadPlaywright() {
  const specifier = process.env.PLAYWRIGHT_MODULE ?? 'playwright';
  try {
    return await import(specifier);
  } catch {
    return null;
  }
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type -- JavaScript test helper
function launchArgs() {
  return [
    '--ozone-platform=wayland',
    '--use-gl=angle',
    '--use-angle=gl',
    '--ignore-gpu-blocklist',
    '--enable-unsafe-webgpu',
    ...(process.env.WEBGPU_ADAPTER ? [`--use-webgpu-adapter=${process.env.WEBGPU_ADAPTER}`] : []),
    ...(process.env.EXTRA_CHROME_ARGS ? process.env.EXTRA_CHROME_ARGS.split(' ').filter(Boolean) : [])
  ];
}

const playwright = await loadPlaywright();
if (!playwright) {
  console.log('SKIP: playwright is not installed; set NODE_PATH or PLAYWRIGHT_MODULE.');
  process.exit(0);
}

const browser = await playwright.chromium.launch({
  channel: process.env.BROWSER_CHANNEL ?? undefined,
  args: launchArgs()
});

try {
  const page = await browser.newPage();
  page.on('pageerror', (error) => console.error('[page error]', error.message));
  if (process.env.WEBGPU_VERBOSE) {
    page.on('console', (message) => console.log(`[page ${message.type()}]`, message.text()));
  }

  await page.route(`${PAGE_ORIGIN}/**`, async (route) => {
    const path = new URL(route.request().url()).pathname;
    const file = join(DIST, normalize(path === '/' ? '/webgpu-browser.html' : path));
    try {
      await route.fulfill({
        status: 200,
        contentType: MIME[extname(file)] ?? 'application/octet-stream',
        body: readFileSync(file)
      });
    } catch {
      await route.fulfill({ status: 404, body: 'not found' });
    }
  });

  await page.goto(`${PAGE_ORIGIN}/webgpu-browser.html`);
  await page.waitForFunction(() => document.documentElement.dataset.webgpuResult !== undefined, null, {
    timeout: 180_000
  });
  const result = JSON.parse(await page.evaluate(() => document.documentElement.dataset.webgpuResult));

  console.log(JSON.stringify(result, null, 2));

  assert.equal(result.error, null, 'the page reported an error');
  assert.equal(result.validationError, null, 'the device reported a validation error');
  assert.equal(result.backend, 'webgpu', 'the renderer did not take the WebGPU backend');

  const gpu = result.gpuDriven;
  assert.ok(gpu, 'the GPU-driven section did not run');
  assert.equal(gpu.active, true, 'the GPU culling path was not active');
  assert.equal(gpu.allVisible, 64, 'every instance should survive when all are in frustum');
  assert.equal(gpu.halfHidden, 32, 'hidden instances must not reach the visible list');
  assert.equal(gpu.afterRemove, 60, 'removed instances must not reach the visible list');
  assert.equal(gpu.afterReadd, 64, 'reused slots must reach the visible list again');
  assert.equal(gpu.noneVisible, 0, 'a camera facing away must produce an empty visible list');
  assert.ok(
    gpu.partiallyVisible > 0 && gpu.partiallyVisible < 64,
    `partial visibility should be a strict subset, got ${gpu.partiallyVisible}`
  );
  assert.ok(gpu.litPixels > 200, `the indirect draw produced no visible geometry (${gpu.litPixels} px)`);

  assert.ok(
    gpu.multiMaterial.redOnly > 100 && gpu.multiMaterial.blueOnly > 100,
    `both material groups must draw, got ${JSON.stringify(gpu.multiMaterial)}`
  );

  // A mesh under a translated, rotated and scaled parent.
  assert.equal(gpu.transformedAll, 16, 'a transformed mesh lost instances that were in view');
  assert.ok(
    gpu.transformedNear > 0 && gpu.transformedNear < 16,
    `a close camera should keep a strict subset, got ${gpu.transformedNear}`
  );
  assert.equal(gpu.transformedNone, 0, 'a transformed mesh kept instances that were behind the camera');

  assert.deepEqual(gpu.lodLevels, [8, 8], 'LOD classification did not split by distance');
  // The shadow pass culls against the light's frustum -- which excludes the
  // far row entirely -- but picks LOD levels by distance from the viewer, so
  // the survivors land on the near level. Both halves matter: visibility is
  // the light's, level selection is the camera's, as on the WebGL path.
  assert.deepEqual(
    gpu.shadowLevels,
    [8, 0],
    'the shadow pass did not cull against the light or did not level by the viewer'
  );

  console.log('\nOK: GPU-driven culling, LOD classification, indirect draws and shadows verified.');
} finally {
  await browser.close();
}
