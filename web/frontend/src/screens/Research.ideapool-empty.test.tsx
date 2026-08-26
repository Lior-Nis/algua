/** Own file: api.ts's fetch cache is module-level and keyed by URL, and vitest isolates the
 * module graph per FILE — an `/api/ideas` fixture with a DIFFERENT row count than
 * Research.test.tsx's needs its own file (ScatterISOOS.empty.test.tsx precedent). */
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import Research from './Research'

const strategies = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: { data: [] },
}

const runs = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: { count: 0, runs: [] },
}

const ops = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: { ok: true, checked_at: '2026-08-15T17:04:00Z', alerting: [], loops: {} },
}

const emptyIdeas = {
  ok: true,
  ideas: [],
  stats: { window_days: 90, counts: {} },
  stats_window_days: 90,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('collapses the idea pool to a bare count when it has zero rows — no tiles, no chips, no list', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const body = url.startsWith('/api/strategies')
        ? strategies
        : url.startsWith('/api/runs')
          ? runs
          : url.startsWith('/api/ops')
            ? ops
            : emptyIdeas
      return { ok: true, status: 200, json: async () => body }
    }) as unknown as typeof fetch,
  )
  const { container } = render(
    <MemoryRouter>
      <Research />
    </MemoryRouter>,
  )
  expect(await screen.findByText('idea pool')).toBeTruthy()
  expect(screen.getByText('0 ideas')).toBeTruthy()
  expect(container.querySelector('.tile-row')).toBeNull()
  expect(container.querySelector('.filter-chip')).toBeNull()
  expect(container.querySelector('.alert-list')).toBeNull()
})
