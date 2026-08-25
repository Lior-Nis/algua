import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import Research from './Research'

const strategies = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: {
    data: [
      { id: 1, name: 'live_one', stage: 'paper', family: 'momentum', tags: [], author: 'agent',
        hypothesis_status: 'untested', derived_from: null, description: null },
      { id: 2, name: 'benched', stage: 'retired', family: null, tags: [], author: 'agent',
        hypothesis_status: 'refuted', derived_from: null, description: null },
    ],
  },
}

const runs = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: { count: 0, runs: [] },
}

const ops = {
  ok: false,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: {
    ok: false,
    checked_at: '2026-08-15T17:04:00Z',
    alerting: ['research'],
    loops: {
      research: { health: 'rate_limited', detail: 'provider usage limit reached',
                  last_ok_at: '2026-08-14T14:00:01+00:00', consecutive_failures: 13 },
      mergeback: { health: 'stale', queue_depth: 23 },
    },
  },
}

const ideas = {
  ok: true,
  ideas: [
    { id: 1, title: 'skip-month persistence', hypothesis: 'h', family: null, tags: [],
      source_type: 'paper', source_ref: null, source_date: null, status: 'open',
      created_at: '2026-08-10T00:00:00+00:00' },
  ],
  stats: { window_days: 90, counts: { open: 1, authored: 0, total: 1 } },
  stats_window_days: 90,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function renderResearch() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const body = url.startsWith('/api/strategies')
        ? strategies
        : url.startsWith('/api/runs')
          ? runs
          : url.startsWith('/api/ops')
            ? ops
            : ideas
      return { ok: true, status: 200, json: async () => body }
    }) as unknown as typeof fetch,
  )
  render(
    <MemoryRouter>
      <Research />
    </MemoryRouter>,
  )
}

it('surfaces a stopped research loop — the difference between "no ideas" and "none ever again"', async () => {
  renderResearch()
  expect(await screen.findByText('research loop')).toBeTruthy()
  expect(screen.getByText('provider usage limit reached')).toBeTruthy()
  expect(screen.getByText(/13 consecutive failed runs/)).toBeTruthy()
})

it('reports a wedged merge-back queue', async () => {
  renderResearch()
  expect(await screen.findByText(/merge-back queue 23 — wedged/)).toBeTruthy()
})

it('summarizes the funnel as a one-line count strip, not per-strategy detail', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const body = url.startsWith('/api/strategies')
        ? strategies
        : url.startsWith('/api/runs')
          ? runs
          : url.startsWith('/api/ops')
            ? ops
            : ideas
      return { ok: true, status: 200, json: async () => body }
    }) as unknown as typeof fetch,
  )
  const { container } = render(
    <MemoryRouter>
      <Research />
    </MemoryRouter>,
  )
  expect(await screen.findByText('paper (1)')).toBeTruthy()
  expect(screen.getByText('retired (1)')).toBeTruthy()
  // Per-strategy names/links used to render here (`<details>` per stage) — that is now Fleet's
  // job. The funnel section is chips only: no strategy name, no link, no per-row health badge.
  expect(screen.queryByText('live_one')).toBeNull()
  expect(screen.queryByText('benched')).toBeNull()
  expect(container.querySelector('.funnel-row')).toBeNull()
  expect(container.querySelector('details')).toBeNull()
})

it('wires in the IS-vs-OOS scatter and the ranked run list as the run-ledger surface', async () => {
  renderResearch()
  expect(await screen.findByText('is vs oos')).toBeTruthy()
  expect(screen.getByText('ranked runs')).toBeTruthy()
  expect(
    screen.getByText(/no runs recorded yet — the ledger fills when the operator loop runs/i),
  ).toBeTruthy()
})

it('absorbs the idea pool, keeping the windowed stats on their own line', async () => {
  renderResearch()
  expect(await screen.findByText('idea pool')).toBeTruthy()
  expect(screen.getByText('skip-month persistence')).toBeTruthy()
  expect(screen.getByText(/last 90d: open 1/)).toBeTruthy()
})
