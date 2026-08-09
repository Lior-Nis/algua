/// <reference lib="webworker" />
// Minimal service worker: precache the app shell, serve it for navigations.
// NO runtime caching of /api/* — a monitor must never serve SW-cached API data.

import { clientsClaim } from 'workbox-core'
import { createHandlerBoundToURL, precacheAndRoute } from 'workbox-precaching'
import { NavigationRoute, registerRoute } from 'workbox-routing'

declare const self: ServiceWorkerGlobalScope

precacheAndRoute(self.__WB_MANIFEST)

// Deep links (/s/x, /funnel, ...) get the precached shell; API/health never.
// Workbox matches pathname+search, so the namespace rule must also stop at ? and #
// (a bare /api?x=1 navigation is still API namespace).
registerRoute(
  new NavigationRoute(createHandlerBoundToURL('index.html'), {
    denylist: [/^\/api(?:[/?#]|$)/, /^\/healthz(?:[?#]|$)/],
  }),
)

// autoUpdate + injectManifest + injectRegister:false: nothing sends SKIP_WAITING,
// so the new worker must activate itself — otherwise an update waits forever and
// the registration module's activated-reload never fires.
void self.skipWaiting()
clientsClaim()

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    void self.skipWaiting()
  }
})
