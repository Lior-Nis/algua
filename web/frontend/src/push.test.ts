import { afterEach, expect, it, vi } from 'vitest'
import { alertsState, urlBase64ToUint8Array } from './push'

afterEach(() => {
  vi.unstubAllGlobals()
})

function toBase64Url(bytes: Uint8Array): string {
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

it('urlBase64ToUint8Array round-trips arbitrary bytes (unpadded base64url)', () => {
  for (const len of [1, 2, 3, 65]) {
    const bytes = Uint8Array.from({ length: len }, (_, i) => (i * 37 + 4) % 256)
    expect(Array.from(urlBase64ToUint8Array(toBase64Url(bytes)))).toEqual(Array.from(bytes))
  }
  // The base64url alphabet actually maps - and _
  expect(Array.from(urlBase64ToUint8Array('-_8'))).toEqual([251, 255])
})

function stubPushEnv(opts: {
  userAgent?: string
  serviceWorker?: unknown
  pushManager?: boolean
  permission?: NotificationPermission
}): void {
  vi.stubGlobal('navigator', {
    userAgent: opts.userAgent ?? 'Mozilla/5.0 (X11; Linux x86_64)',
    ...(opts.serviceWorker !== undefined ? { serviceWorker: opts.serviceWorker } : {}),
  })
  if (opts.pushManager) vi.stubGlobal('PushManager', class {})
  if (opts.permission) vi.stubGlobal('Notification', { permission: opts.permission })
}

function fakeServiceWorker(subscription: unknown) {
  return {
    getRegistration: async () => ({
      pushManager: { getSubscription: async () => subscription },
    }),
  }
}

it("alertsState is 'needs-install' on iOS Safari outside standalone", async () => {
  // test-setup's matchMedia stub always reports matches: false (not standalone).
  stubPushEnv({ userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)' })
  expect(await alertsState()).toBe('needs-install')
})

it("alertsState is 'unsupported' without serviceWorker/PushManager", async () => {
  stubPushEnv({})
  expect(await alertsState()).toBe('unsupported')
})

it("alertsState is 'denied' when notification permission is denied", async () => {
  stubPushEnv({ serviceWorker: fakeServiceWorker(null), pushManager: true, permission: 'denied' })
  expect(await alertsState()).toBe('denied')
})

it("alertsState is 'on' with an existing subscription", async () => {
  stubPushEnv({
    serviceWorker: fakeServiceWorker({ endpoint: 'https://push.example.com/v1/x' }),
    pushManager: true,
    permission: 'granted',
  })
  expect(await alertsState()).toBe('on')
})

it("alertsState is 'off' without a subscription", async () => {
  stubPushEnv({ serviceWorker: fakeServiceWorker(null), pushManager: true, permission: 'default' })
  expect(await alertsState()).toBe('off')
})
