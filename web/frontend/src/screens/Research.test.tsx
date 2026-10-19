import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import { clearFetchCache } from '../api'
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
    { id: 1, title: 'skip-month persistence', hypothesis: 'Returns persist after the skip month.', family: null, tags: [],
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
  clearFetchCache()
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

function renderResearchWithOps(opsResponse: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => url.startsWith('/api/ops') ? opsResponse : routeBody(url),
    })) as unknown as typeof fetch,
  )
  return render(
    <MemoryRouter>
      <Research />
    </MemoryRouter>,
  )
}

it('shows loading rather than inventing a blocked decision before required reads finish', () => {
  vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})) as unknown as typeof fetch)
  const { container } = render(
    <MemoryRouter>
      <Research />
    </MemoryRouter>,
  )
  expect(container.querySelector('.decision-state')?.textContent).toBe('loading')
  expect(screen.getByText(/no current decision yet/i)).toBeTruthy()
  const inventory = screen.getByRole('region', { name: /lifecycle inventory/i })
  expect(within(inventory).getByTestId('evidence-status').getAttribute('data-state')).toBe('loading')
  expect(within(inventory).queryByText(/no strategies registered/i)).toBeNull()
})

it('labels an empty cached registry response as stale rather than authoritative empty inventory', async () => {
  const staleEmptyStrategies = {
    ...strategies,
    stale: true,
    data: { data: [] },
  }
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => url.startsWith('/api/strategies') ? staleEmptyStrategies : routeBody(url),
    })) as unknown as typeof fetch,
  )
  render(<MemoryRouter><Research /></MemoryRouter>)
  const inventory = await screen.findByRole('region', { name: /lifecycle inventory/i })
  const status = within(inventory).getByTestId('evidence-status')
  expect(status.getAttribute('data-state')).toBe('stale')
  expect(status.textContent).toMatch(/cached.*no strategies/i)
})

it('encodes lifecycle counts proportionally without inflating small stages to a false minimum', async () => {
  const skewed = {
    ...strategies,
    data: {
      data: [
        { ...strategies.data.data[0], id: 1, stage: 'paper' },
        ...Array.from({ length: 99 }, (_, index) => ({
          ...strategies.data.data[1], id: index + 2, name: `retired_${index}`, stage: 'retired',
        })),
      ],
    },
  }
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => url.startsWith('/api/strategies') ? skewed : routeBody(url),
    })) as unknown as typeof fetch,
  )
  const view = render(<MemoryRouter><Research /></MemoryRouter>)
  await screen.findByRole('region', { name: /lifecycle inventory/i })
  const widths = [...view.container.querySelectorAll<HTMLElement>('.funnel-track span')]
    .map((node) => node.style.width)

  expect(widths).toEqual(['1.0101010101010102%', '100%'])
})

it('surfaces a stopped research loop — the difference between "no ideas" and "none ever again"', async () => {
  renderResearch()
  expect(await screen.findByRole('heading', { name: /research decision/i })).toBeTruthy()
  expect(screen.getByText(/^blocked$/i)).toBeTruthy()
  expect(screen.getByText(/new evidence may not arrive/i)).toBeTruthy()
  expect(screen.getByText(/next clean research loop run/i)).toBeTruthy()
  expect(await screen.findByText('research loop')).toBeTruthy()
  expect(screen.getByText(/provider usage limit reached/)).toBeTruthy()
  expect(screen.getByText(/13 consecutive failed runs/)).toBeTruthy()
})

it('reports a wedged merge-back queue', async () => {
  renderResearch()
  expect(await screen.findByText(/merge-back queue 23 is wedged/i)).toBeTruthy()
})

it.each([
  ['absent', undefined],
  ['unknown', { health: 'mystery', queue_depth: 0 }],
])('blocks when merge-back state is %s', async (_label, mergeback) => {
  renderResearchWithOps({
    ...ops,
    data: {
      ...ops.data,
      alerting: [],
      loops: {
        research: { health: 'ok', detail: 'idle', last_ok_at: '2026-08-15T17:00:00Z', consecutive_failures: 0 },
        ...(mergeback === undefined ? {} : { mergeback }),
      },
    },
  })
  expect(await screen.findByText(/^blocked$/i)).toBeTruthy()
  expect(screen.getByText(/merge-back.*(?:absent|mystery)/i)).toBeTruthy()
})

it('qualifies a blocked loop conclusion when operations came from stale cache', async () => {
  renderResearchWithOps({ ...ops, stale: true })
  expect(await screen.findByText(/^blocked$/i)).toBeTruthy()
  expect(screen.getByText(/operations evidence is stale.*cached research loop/i)).toBeTruthy()
})

it('keeps the funnel count strip separate from the board filters', async () => {
  const { container } = renderResearch()
  await screen.findByRole('heading', { name: 'strategy board' })
  expect(screen.getByLabelText('paper: 1 strategies').closest('button')).toBeNull()
  expect(screen.getByLabelText('retired: 1 strategies').closest('button')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: /board controls/i }))
  expect(screen.getByRole('button', { name: /^paper 1$/ })).toBeTruthy()
  expect(screen.getByRole('button', { name: /^retired 1$/ })).toBeTruthy()
  expect(screen.getByRole('heading', { name: /lifecycle inventory/i })).toBeTruthy()
  expect(container.querySelector('[aria-label="strategy inventory"]')).toBeTruthy()
  expect(container.querySelectorAll('.funnel-track')).toHaveLength(2)
})

it('wires in the IS-vs-OOS scatter and the strategy board, not the ranked run list', async () => {
  renderResearch()
  expect(await screen.findByText('is vs oos')).toBeTruthy()
  expect(await screen.findByRole('heading', { name: 'strategy board' })).toBeTruthy()
  expect(screen.queryByText('ranked runs')).toBeNull()
  // One row per registered strategy — including the one with no gate evidence.
  const boardRows = screen.getAllByTestId('board-row')
  expect(boardRows).toHaveLength(2)
  expect(within(boardRows[1]).getByText(/^no usable gate evidence$/i)).toBeTruthy()
})

it('absorbs the idea pool, keeping the windowed stats on their own line', async () => {
  renderResearch()
  expect(await screen.findByText('idea pool')).toBeTruthy()
  const idea = screen.getByRole('button', { name: /skip-month persistence/i })
  expect(idea.getAttribute('aria-expanded')).toBe('false')
  expect(screen.queryByText('Returns persist after the skip month.')).toBeNull()
  fireEvent.click(idea)
  expect(idea.getAttribute('aria-expanded')).toBe('true')
  expect(screen.getByText('Returns persist after the skip month.')).toBeTruthy()
  expect(screen.getByText(/last 90d: open 1/)).toBeTruthy()
})

it('shows pool depth above the tiles and yield-by-venue after the list', async () => {
  renderResearch()
  expect(await screen.findByText(/10 open.*refill at 72/)).toBeTruthy()
  expect(screen.queryByRole('table')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: /yield by venue/i }))
  expect(screen.getByRole('table')).toBeTruthy()
  expect(screen.getByText('arxiv')).toBeTruthy()
  expect(screen.getByText('forums')).toBeTruthy()
  expect(screen.getByText('60.0%')).toBeTruthy()  // arxiv integrity_yield
  const dashCells = screen.getAllByText('--')
  expect(dashCells.length).toBeGreaterThan(0)  // forums' null rates render as --
})
