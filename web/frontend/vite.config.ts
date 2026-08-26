import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import { defineConfig } from 'vitest/config'

export default defineConfig({
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
})
