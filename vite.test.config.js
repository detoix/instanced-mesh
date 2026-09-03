import { resolve } from 'path';
import { defineConfig } from 'vite';

/**
 * Bundles the WebGPU validation page so it can be opened from a hardware
 * WebGPU browser without a dev server. See test/run-webgpu-browser.mjs.
 */
export default defineConfig({
  root: resolve(__dirname, 'test'),
  publicDir: false,
  base: './',
  build: {
    outDir: resolve(__dirname, 'dist/test-browser'),
    emptyOutDir: true,
    target: 'esnext',
    rollupOptions: {
      input: resolve(__dirname, 'test/webgpu-browser.html')
    }
  }
});
