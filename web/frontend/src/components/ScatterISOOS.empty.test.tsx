/** Own file: api.ts's fetch cache is module-level and keyed by URL, and vitest isolates the
 * module graph per FILE — a second /api/runs fixture needs its own file. */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunsListPayload } from '../types'
import ScatterISOOS from './ScatterISOOS'

const empty: ApiEnvelope<RunsListPayload> = {
  ok: true,
  fetched_at: '2026-08-26T00:00:00Z',
  stale: false,
  data: { count: 0, runs: [] },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders the honest empty state when the ledger has no gate runs — no svg drawn around nothing', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => empty })) as unknown as typeof fetch,
  )
  const { container } = render(<ScatterISOOS />)
  expect(await screen.findByText(/no gate runs recorded yet/i)).toBeTruthy()
  expect(container.querySelector('svg')).toBeNull()
})
