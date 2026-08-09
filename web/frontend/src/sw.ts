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

// Push payloads come from backend/poller.py's diff half. Safari REQUIRES the
// visible-notification work inside waitUntil — a silent push gets the push
// permission revoked — so showNotification runs on EVERY push, defaults and all.
interface PushPayload {
  title?: string
  body?: string
  tag?: string
  url?: string
  renotify?: boolean
}

self.addEventListener('push', (event) => {
  let p: PushPayload = {}
  try {
    p = (event.data?.json() as PushPayload | null) ?? {}
  } catch {
    // malformed payload — still show a notification with the defaults
  }
  const title = p.title ?? 'algua monitor'
  // renotify is real in Chrome/FF but missing from TS's NotificationOptions.
  const options: NotificationOptions & { renotify?: boolean } = {
    body: p.body ?? '',
    tag: p.tag,
    renotify: p.renotify ?? false,
    data: { url: p.url ?? '/' },
  }
  event.waitUntil(self.registration.showNotification(title, options))
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const url = (event.notification.data as { url?: string } | undefined)?.url ?? '/'
  event.waitUntil(
    (async () => {
      const wins = await self.clients.matchAll({ type: 'window', includeUncontrolled: true })
      const existing = wins[0]
      if (existing) {
        await existing.focus()
        await existing.navigate(url)
      } else {
        await self.clients.openWindow(url)
      }
    })(),
  )
})
