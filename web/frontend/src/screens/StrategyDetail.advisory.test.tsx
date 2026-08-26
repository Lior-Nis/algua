/** Own file: api.ts's fetch cache is module-level and keyed by URL, and vitest isolates
 * the module graph per FILE — a second /api/strategy/* fixture needs its own file. */
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type {
  ApiEnvelope,
  RunDetail,
  RunSeriesPayload,
  RunsListPayload,
  SeriesPayload,
  StrategyDetailResponse,
} from '../types'
import StrategyDetail from './StrategyDetail'

/** The NORMAL shape under the factory soft gate: the gate PASSES on its binding floors
 * while advisory statistical checks fail — `passed` ignores advisory checks entirely
 * (algua/research/gates.py). The token is minted but not yet spent. */
const detail: StrategyDetailResponse = {
  ok: true,
  strategy: 'mom_breakout',
  registry: {
    id: 1,
    name: 'mom_breakout',
    stage: 'candidate',
    family: 'momentum',
    tags: [],
    author: 'agent',
    hypothesis_status: 'untested',
    derived_from: null,
    description: null,
    transitions: [],
  },
  paper: null,
  gates: {
    strategy: 'mom_breakout',
    gate_evaluations: [
      {
        id: 9,
        passed: 1,
        consumed: 0,
        actor: 'agent',
        created_at: '2026-08-02T15:00:00+00:00',
        decision: {
          passed: true,
          checks: [
            { name: 'pit_universe', op: '==', threshold: null, value: null, passed: true },
            {
              name: 'holdout_sharpe',
              op: '>=',
              threshold: 0.62,
              value: 0.31,
              passed: false,
              advisory: true,
            },
          ],
        },
      },
    ],
    forward_gate_evaluations: [],
  },
  gates_limit: 50,
  part_errors: { paper: 'cli_failed' },
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
}

const seriesEnvelope: ApiEnvelope<SeriesPayload> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    strategy: 'mom_breakout',
    lane_filter: null,
    series: { paper: [], live: null },
    truncated: { paper: false, live: null },
    n_legacy_excluded: 0,
    n_unparseable: 0,
    n_invalid_lane: 0,
  },
}

// The gate bullet card (task 5) sources checks from `/api/runs/{id}`, a SEPARATE waterfall from
// `detail.gates` above (`/api/runs?...&kind=gate&limit=1` -> `/api/runs/{id}`). Same checks as
// `detail.gates.gate_evaluations[0].decision` so this exercises the same advisory-pass scenario.
const runsListEnvelope: ApiEnvelope<RunsListPayload> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    count: 1,
    runs: [
      {
        id: 90,
        kind: 'gate',
        strategy_name: 'mom_breakout',
        strategy_id: 1,
        created_at: '2026-08-02T15:00:00+00:00',
        passed: 1,
        mean_window_sharpe: null,
        sharpe_oos: null,
      },
    ],
  },
}

const runDetailEnvelope: ApiEnvelope<RunDetail> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    id: 90,
    kind: 'gate',
    strategy_name: 'mom_breakout',
    strategy_id: 1,
    created_at: '2026-08-02T15:00:00+00:00',
    passed: 1,
    mean_window_sharpe: null,
    sharpe_oos: null,
    extra_metrics: {},
    gate_decision: {
      passed: true,
      checks: [
        { name: 'pit_universe', op: '==', threshold: null, value: null, passed: true },
        {
          name: 'holdout_sharpe',
          op: '>=',
          threshold: 0.62,
          value: 0.31,
          passed: false,
          advisory: true,
        },
      ],
    },
  },
}

// ReturnOverlay (task 7) also queries `/api/runs?...&kind=backtest` and batches its ids through
// `/api/runs/series` — no backtest/holdout series is fixtured here, so this stays an empty
// envelope; the overlay just renders its own honest empty state.
const runSeriesEnvelope: ApiEnvelope<RunSeriesPayload> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: { series: {} },
}

function renderDetail() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => {
        // Registered before the `/api/runs?`/`/api/runs/{id}` checks below — see the sibling
        // note in StrategyDetail.test.tsx.
        if (url.startsWith('/api/runs/series')) return runSeriesEnvelope
        if (url.endsWith('/series')) return seriesEnvelope
        if (/^\/api\/runs\/\d+$/.test(url)) return runDetailEnvelope
        if (url.startsWith('/api/runs?')) return runsListEnvelope
        return detail
      },
    })) as unknown as typeof fetch,
  )
  render(
    <MemoryRouter initialEntries={['/s/mom_breakout']}>
      <Routes>
        <Route path="/s/:name" element={<StrategyDetail />} />
      </Routes>
    </MemoryRouter>,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('marks a failed ADVISORY check as non-vetoing instead of rendering it like a breach', async () => {
  renderDetail()

  // The gate bullet card (task 5): advisory checks render in their OWN group, captioned as
  // non-vetoing, and the failed one carries an explicit "advisory" text label — never colour
  // alone.
  expect(await screen.findByText(/advisory — recorded, never vetoes/)).toBeTruthy()
  const advisoryGroup = screen.getByTestId('bullet-group-advisory')
  const advisoryRow = within(advisoryGroup).getByText('holdout_sharpe').closest(
    '[data-testid="gate-check-row"]',
  ) as HTMLElement
  expect(within(advisoryRow).getByText('advisory')).toBeTruthy()
  // The advisory failure must NOT wear the red of a breached binding floor.
  const marks = screen.getAllByText('fail')
  expect(marks).toHaveLength(1)
  expect(marks[0].getAttribute('style')).toContain('var(--amber)')
})

it('names the ledger by the lifecycle edge it gates', async () => {
  renderDetail()
  expect(await screen.findByText(/research gate — backtested → candidate/)).toBeTruthy()
})

it('shows whether the passing gate token has been spent', async () => {
  renderDetail()
  expect(await screen.findByText('token unspent')).toBeTruthy()
})
