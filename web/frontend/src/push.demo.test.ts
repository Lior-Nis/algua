import { afterEach, expect, it, vi } from 'vitest'

// Spec §6 guard 2: the demo build must never reach the network. push.ts calls fetch() directly
// (bypassing getJSON/demoJSON), so this is the ONLY thing that would catch a regression here —
// mock DEMO=true (the way vite.config.ts's real build-time constant folding would leave it) and
// prove nothing in push.ts's public surface ever touches fetch.
vi.mock('./transport', () => ({
  DEMO: true,
  demoJSON: async () => {
    throw new Error('unused in this test')
  },
}))

afterEach(() => {
  vi.unstubAllGlobals()
})

it('alertsState never reaches the network when DEMO is true', async () => {
  const fetchSpy = vi.fn()
  vi.stubGlobal('fetch', fetchSpy)
  const { alertsState } = await import('./push')
  expect(await alertsState()).toBe('unsupported')
  expect(fetchSpy).not.toHaveBeenCalled()
})

it('enableAlerts never reaches the network when DEMO is true', async () => {
  const fetchSpy = vi.fn()
  vi.stubGlobal('fetch', fetchSpy)
  const { enableAlerts } = await import('./push')
  expect(await enableAlerts()).toBe('unsupported')
  expect(fetchSpy).not.toHaveBeenCalled()
})

it('disableAlerts never reaches the network when DEMO is true', async () => {
  const fetchSpy = vi.fn()
  vi.stubGlobal('fetch', fetchSpy)
  const { disableAlerts } = await import('./push')
  expect(await disableAlerts()).toBe('off')
  expect(fetchSpy).not.toHaveBeenCalled()
})
