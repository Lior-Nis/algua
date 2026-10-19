import { cleanup, render, screen, within } from '@testing-library/react'
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

const board = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: {
    schema_version: 'strategy-board/v1',
    count: 2,
    rows: [
      { strategy_id: 1, name: 'live_one', family: 'momentum', stage: 'paper',
        latest_gate: { run_id: 7, created_at: '2026-08-14T00:00:00+00:00', passed: true,
                       sharpe_oos: 0.8, total_return_oos: 0.1, max_drawdown_oos: -0.05,
                       n_obs_oos: 126, mean_window_sharpe: 0.7, min_window_sharpe: 0.2,
                       pct_positive_windows: 0.75 } },
      { strategy_id: 2, name: 'benched', family: null, stage: 'retired', latest_gate: null },
    ],
  },
}

function routeBody(url: string) {
  return url.startsWith('/api/strategy-board')
    ? board
    : url.startsWith('/api/strategies')
      ? strategies
      : url.startsWith('/api/runs')
        ? runs
        : url.startsWith('/api/ops')
          ? ops
          : ideas
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
  depth: {
    open_unclaimed: 10, claimed: 2, needs_data: 1, refill_at: 72, ceiling: 252,
    below_refill: false,
    inputs: { runs_per_day: 12, hypotheses_per_run: 3, floor_days: 2, ceiling_days: 7 },
  },
  scorecard: {
    days: 90, min_n_for_rates: 5,
    by_venue: {
      arxiv: { n: 12, outcomes: {}, stages: {}, survivals: 3,
                integrity_yield: 0.6, walkforward_yield: 0.5, survival_yield: 0.25 },
      forums: { n: 2, outcomes: {}, stages: {}, survivals: 0,
                integrity_yield: null, walkforward_yield: null, survival_yield: null },
    },
    by_category: {}, by_obscurity: {}, by_inspiration: {},
  },
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
    vi.fn(async (url: string) => ({
      ok: true, status: 200, json: async () => routeBody(url),
    })) as unknown as typeof fetch,
  )
  return render(
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

it('keeps the funnel count strip separate from the board filters', async () => {
  const { container } = renderResearch()
  await screen.findByText('strategy board')
  expect(screen.getByText('paper (1)')).toBeTruthy()
  expect(screen.getByText('retired (1)')).toBeTruthy()
  expect(screen.getByRole('button', { name: /^paper 1$/ })).toBeTruthy()
  expect(screen.getByRole('button', { name: /^retired 1$/ })).toBeTruthy()
  expect(screen.getByText(/^funnel$/i)).toBeTruthy()
  expect(container.querySelector('.funnel-row')).toBeNull()
  expect(container.querySelector('details')).toBeNull()
})

it('wires in the IS-vs-OOS scatter and the strategy board, not the ranked run list', async () => {
  renderResearch()
  expect(await screen.findByText('is vs oos')).toBeTruthy()
  expect(await screen.findByText('strategy board')).toBeTruthy()
  expect(screen.queryByText('ranked runs')).toBeNull()
  // One row per registered strategy — including the one with no gate evidence.
  const boardRows = screen.getAllByTestId('board-row')
  expect(boardRows).toHaveLength(2)
  expect(within(boardRows[1]).getByText(/^no usable gate evidence$/i)).toBeTruthy()
})

it('absorbs the idea pool, keeping the windowed stats on their own line', async () => {
  renderResearch()
  expect(await screen.findByText('idea pool')).toBeTruthy()
  expect(screen.getByText('skip-month persistence')).toBeTruthy()
  expect(screen.getByText(/last 90d: open 1/)).toBeTruthy()
})

it('shows pool depth above the tiles and yield-by-venue after the list', async () => {
  renderResearch()
  expect(await screen.findByText(/10 open.*refill at 72/)).toBeTruthy()
  expect(screen.getByRole('table')).toBeTruthy()
  expect(screen.getByText('arxiv')).toBeTruthy()
  expect(screen.getByText('forums')).toBeTruthy()
  expect(screen.getByText('60.0%')).toBeTruthy()  // arxiv integrity_yield
  const dashCells = screen.getAllByText('--')
  expect(dashCells.length).toBeGreaterThan(0)  // forums' null rates render as --
})
