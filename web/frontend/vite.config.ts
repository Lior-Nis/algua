import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import { defineConfig } from 'vitest/config'

// Only the real `vite build` for the PRODUCTION (non-demo) bundle swaps `./transport` for the
// fixture-free stub — never `vite build` for the demo bundle (VITE_ALGUA_DEMO=1, which needs
// the real fixtures), never `vite dev`, and never a Vitest run (which imports `./transport`
// directly from transport.test.ts and must get the real module). See src/transport.prod.ts
// for why tree-shaking alone isn't enough to keep fixtures out of the production bundle.
function prodTransportAlias(command: string) {
  if (command !== 'build' || process.env.VITEST || process.env.VITE_ALGUA_DEMO === '1') return []
  return [
    {
      find: './transport',
      replacement: fileURLToPath(new URL('./src/transport.prod.ts', import.meta.url)),
    },
  ]
}

export default defineConfig(({ command }) => ({
  resolve: {
    alias: prodTransportAlias(command),
  },
  plugins: [
    react(),
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.ts',
      registerType: 'autoUpdate',
      injectRegister: false,
      manifest: {
        name: 'algua monitor',
        short_name: 'algua',
        description: 'read-only fleet monitor',
        display: 'standalone',
        start_url: '/',
        scope: '/',
        theme_color: '#000000',
        background_color: '#000000',
        icons: [
          { src: '/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icon-512.png', sizes: '512x512', type: 'image/png' },
          {
            src: '/icon-maskable-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
        ],
      },
    }),
  ],
  server: {
    proxy: {
      // Default matches the deployed monitor's port (systemd `algua-web.service`
      // binds 8787). Override with ALGUA_WEB_DEV_PROXY_TARGET to point the dev
      // server at a scratch backend without editing this file.
      '/api': process.env.ALGUA_WEB_DEV_PROXY_TARGET ?? 'http://127.0.0.1:8787',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
}))
