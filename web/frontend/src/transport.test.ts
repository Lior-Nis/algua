import { afterEach, describe, expect, it, vi } from 'vitest'
import { demoJSON } from './transport'

afterEach(() => vi.restoreAllMocks())

describe('demoJSON', () => {
  it('serves a fixture WITHOUT touching the network', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const payload = await demoJSON<{ ok: boolean }>('/api/fleet')
    expect(payload.ok).toBe(true)
    // Spec §6 guard 2: the demo build must never reach the network.
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('THROWS on an unfixtured URL rather than inventing an empty payload', async () => {
    // A silent empty payload would look exactly like a real empty state, which is the
    // failure mode this whole redesign exists to remove.
    await expect(demoJSON('/api/not-fixtured')).rejects.toThrow(/no fixture/i)
  })
})
