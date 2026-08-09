/// <reference lib="webworker" />
// Minimal service worker: precache the app shell, serve it for navigations.
// NO runtime caching of /api/* — a monitor must never serve SW-cached API data.

import { clientsClaim } from 'workbox-core'
import { createHandlerBoundToURL, precacheAndRoute } from 'workbox-precaching'
import { NavigationRoute, registerRoute } from 'workbox-routing'

declare const self: ServiceWorkerGlobalScope

precacheAndRoute(self.__WB_MANIFEST)

// Deep links (/s/x, /funnel, ...) get the precached shell; API/health never.
registerRoute(
  new NavigationRoute(createHandlerBoundToURL('index.html'), {
    denylist: [/^\/api(?:\/|$)/, /^\/healthz$/],
  }),
)

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    void self.skipWaiting()
  }
})

clientsClaim()
