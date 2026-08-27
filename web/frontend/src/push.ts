// Web-push client plumbing (slice E): capability detection + enable/disable
// against the backend's /api/push endpoints.
import { DEMO } from './transport'

export type AlertsState = 'unsupported' | 'needs-install' | 'denied' | 'on' | 'off'

/** Decode a base64url VAPID application-server key to the bytes subscribe() wants. */
export function urlBase64ToUint8Array(base64url: string): Uint8Array<ArrayBuffer> {
  const padding = '='.repeat((4 - (base64url.length % 4)) % 4)
  const base64 = (base64url + padding).replace(/-/g, '+').replace(/_/g, '/')
  const raw = atob(base64)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i)
  return out
}

/** iOS Safari only exposes web push to INSTALLED (home-screen) apps. */
function needsInstall(): boolean {
  const ios = /iphone|ipad/i.test(navigator.userAgent)
  return ios && !window.matchMedia('(display-mode: standalone)').matches
}

function supported(): boolean {
  return 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window
}

export async function alertsState(): Promise<AlertsState> {
  // Spec §6 guard 2: the demo build must never reach the network. Short-circuit BEFORE any
  // other check — the ENABLE ALERTS control this feeds is real UI wired to real fetch() calls
  // (Notification.requestPermission + /api/push/key + /api/push/subscribe), and the fixture
  // happily serves /api/push/key, so nothing else here would notice a live demo hitting it.
  if (DEMO) return 'unsupported'
  // needs-install FIRST: iOS Safari outside standalone hides PushManager entirely,
  // so the unsupported check would otherwise shadow the actionable answer.
  if (needsInstall()) return 'needs-install'
  if (!supported()) return 'unsupported'
  if (Notification.permission === 'denied') return 'denied'
  // getRegistration (not .ready): in dev no SW is registered and .ready never settles.
  const reg = await navigator.serviceWorker.getRegistration()
  const sub = reg ? await reg.pushManager.getSubscription() : null
  return sub ? 'on' : 'off'
}

export async function enableAlerts(): Promise<AlertsState> {
  // Belt-and-suspenders with the alertsState() short-circuit above: alertsState() already
  // hides the ENABLE ALERTS control in demo mode, but this guards the subscribe path directly
  // in case anything ever calls enableAlerts() without going through that gate first.
  if (DEMO) return 'unsupported'
  const permission = await Notification.requestPermission()
  if (permission !== 'granted') return permission === 'denied' ? 'denied' : 'off'
  const keyRes = await fetch('/api/push/key', { headers: { accept: 'application/json' } })
  if (!keyRes.ok) throw new Error(`push key fetch failed: HTTP ${keyRes.status}`)
  const { key } = (await keyRes.json()) as { key: string }
  const reg = await navigator.serviceWorker.ready
  const sub = await reg.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(key),
  })
  const res = await fetch('/api/push/subscribe', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(sub.toJSON()),
  })
  if (!res.ok) {
    // Don't leave a browser subscription the backend never learned about.
    await sub.unsubscribe()
    throw new Error(`subscription registration failed: HTTP ${res.status}`)
  }
  return 'on'
}

export async function disableAlerts(): Promise<AlertsState> {
  // Same belt-and-suspenders as enableAlerts() above.
  if (DEMO) return 'off'
  const reg = await navigator.serviceWorker.getRegistration()
  const sub = reg ? await reg.pushManager.getSubscription() : null
  if (sub) {
    const endpoint = sub.endpoint
    await sub.unsubscribe()
    await fetch(`/api/push/subscribe?endpoint=${encodeURIComponent(endpoint)}`, {
      method: 'DELETE',
    })
  }
  return 'off'
}
