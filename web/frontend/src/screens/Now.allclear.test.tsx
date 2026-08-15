/** Own file: api.ts's fetch cache is module-level and keyed by URL, and vitest isolates the
 * module graph per FILE — a second /api/triage fixture needs its own file. */
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { TriagePayload } from '../types'
import Now from './Now'

const clear: TriagePayload = {
  ok: true,
  items: [],
  sources: { fleet: true, ops: true, book: true },
  headline: { fleet_ok: 7, fleet_total: 10, book_allocated: 7, book_capacity: 64,
              loops_alerting: 0 },
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('collapses to a single quiet line when nothing needs you', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () =>
        url.startsWith('/api/activity')
          ? { ok: true, fetched_at: clear.fetched_at, stale: false, data: { data: [] } }
          : clear,
    })) as unknown as typeof fetch,
  )

  render(
    <MemoryRouter>
      <Now />
    </MemoryRouter>,
  )

  // The all-clear case is the COMMON case: one line, no panel, no celebration.
  expect(await screen.findByText('nothing needs you')).toBeTruthy()
  // ...and no partial-read warning, because every source loaded.
  expect(screen.queryByText(/unavailable — this list may be incomplete/)).toBeNull()
})
