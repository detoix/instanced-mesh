import { resolve } from 'path';
import { defineConfig } from 'vite';
import { externalizeDeps } from 'vite-plugin-externalize-deps';

export default defineConfig({
  publicDir: false,
  build: {
    emptyOutDir: false,
    sourcemap: true,
    lib: {
      entry: resolve(__dirname, 'src/index.webgpu.ts'),
      fileName: 'build/webgpu',
      formats: ['es']
    }
  },
  plugins: [externalizeDeps()]
});
