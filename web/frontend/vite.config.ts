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

// Spec §6 guard 2, layer 1 (the airtight one — see FIX 9 in the 2026-08-27 fix wave). The
// `verify-demo-build.mjs` sentinel grep is a second, coarser layer: it inspects the FINISHED
// bundle for one constant string, so a module that imports a named fixture export (e.g. `SEEDS`
// for a future empty-state placeholder) without ever touching that constant could ship fixture
// data while the grep still prints `ok`. This plugin instead fails the build the moment ANYTHING
// resolves a module under `src/fixtures/` — at resolve time, before Rollup ever gets a chance to
// tree-shake it away. Only wired into the real, non-demo, non-test build (same gate as the
// transport alias above): the demo build and every Vitest run need the real fixtures module.
function banFixturesInProdBuild(command: string) {
  if (command !== 'build' || process.env.VITEST || process.env.VITE_ALGUA_DEMO === '1') return []
  const FIXTURES_SPECIFIER = /(^|\/)fixtures(\/|$)/
  return [
    {
      name: 'ban-fixtures-in-prod-build',
      // `enforce: 'pre'` is load-bearing: Vite's own core resolver plugin runs BEFORE a
      // normal-priority plugin's `resolveId`, and Rollup's resolveId chain stops at the first
      // hook that returns a resolved id — so without this, Vite would already have turned
      // `../fixtures/steady-state` into an absolute filesystem path before this hook ever saw
      // the raw specifier, and the plugin would silently never fire. Confirmed empirically: it
      // is the difference between this plugin catching a fixture import and doing nothing.
      enforce: 'pre',
      resolveId(source: string) {
        if (FIXTURES_SPECIFIER.test(source)) {
          throw new Error(
            `production build must never resolve a fixtures module (attempted "${source}"). ` +
              'Fixture data is demo-only (VITE_ALGUA_DEMO=1) — see src/transport.ts, ' +
              'src/transport.prod.ts, and scripts/verify-demo-build.mjs.',
          )
        }
        return null
      },
    },
  ]
}

export default defineConfig(({ command }) => ({
  resolve: {
    alias: prodTransportAlias(command),
  },
  plugins: [
    ...banFixturesInProdBuild(command),
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
