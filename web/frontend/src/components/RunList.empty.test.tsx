/** Own file: api.ts's fetch cache is module-level and keyed by URL, and vitest isolates the
 * module graph per FILE — a second /api/runs fixture needs its own file (ScatterISOOS.empty.test.tsx
 * precedent). */
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunsListPayload } from '../types'
import RunList from './RunList'

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

it('renders the honest empty state and NO table chrome when the ledger has no runs', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => empty })) as unknown as typeof fetch,
  )
  const { container } = render(
    <MemoryRouter>
      <RunList />
    </MemoryRouter>,
  )
  expect(
    await screen.findByText(/no runs recorded yet — the ledger fills when the operator loop runs/i),
  ).toBeTruthy()
  // No table/list chrome: no sort chips, no run rows, nothing to sort or tap.
  expect(container.querySelector('.chip-row')).toBeNull()
  expect(container.querySelectorAll('[data-testid="run-row"]').length).toBe(0)
  expect(container.querySelector('a')).toBeNull()
  // No sparkline caption either — it describes a mark that isn't shown in this state.
  expect(screen.queryByText(/worst walk-forward window/i)).toBeNull()
})
