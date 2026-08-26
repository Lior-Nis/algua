import { cleanup, render, screen } from '@testing-library/react'
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

const detail: StrategyDetailResponse = {
  ok: true,
  strategy: 'mom_breakout',
  registry: {
    id: 1,
    name: 'mom_breakout',
    stage: 'backtested',
    family: 'momentum',
    tags: ['trend'],
    author: 'agent',
    hypothesis_status: 'untested',
    derived_from: null,
    description: 'breakout over rolling high',
    transitions: [
      {
        id: 1,
        from_stage: 'idea',
        to_stage: 'backtested',
        actor: 'agent',
        reason: 'walk-forward complete',
        created_at: '2026-08-01T12:00:00+00:00',
      },
    ],
  },
  // A null part = REAL backend failure — the page must degrade, not crash.
  paper: null,
  gates: {
    strategy: 'mom_breakout',
    gate_evaluations: [
      {
        id: 7,
        passed: 0,
        actor: 'agent',
        created_at: '2026-08-02T15:00:00+00:00',
        fdr_p_value: 0.04,
        fdr_alpha_level: 0.00764,
        decision: {
          passed: false,
          dsr_confidence: 0.91,
          checks: [
            { name: 'holdout_sharpe', op: '>=', threshold: 0.62, value: 0.31, passed: false },
          ],
        },
        decision_dropped_keys: [],
      },
    ],
    forward_gate_evaluations: [],
  },
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

// The gate bullet card (task 5) is fed `checks` directly from `detail.gates.gate_evaluations[0]
// .decision` (fix round 2 — no fetch of its own; see `GateBulletCard.tsx`'s docstring). Only
// `TrialDistribution` (task 6) still does the `/api/runs?...&kind=gate&limit=1` ->
// `/api/runs/{id}` two-step waterfall, for the deflation strip and the trial cloud's own marker
// — these two fixtures exist for THAT fetch. Same `holdout_sharpe` check value as
// `detail.gates.gate_evaluations[0].decision` so the two views agree, per both docstrings.
const runsListEnvelope: ApiEnvelope<RunsListPayload> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    count: 1,
    runs: [
      {
        id: 70,
        kind: 'gate',
        strategy_name: 'mom_breakout',
        strategy_id: 1,
        created_at: '2026-08-02T15:00:00+00:00',
        passed: 0,
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
    id: 70,
    kind: 'gate',
    strategy_name: 'mom_breakout',
    strategy_id: 1,
    created_at: '2026-08-02T15:00:00+00:00',
    passed: 0,
    mean_window_sharpe: null,
    sharpe_oos: null,
    extra_metrics: {},
    gate_decision: {
      passed: false,
      dsr_confidence: 0.91,
      checks: [
        { name: 'holdout_sharpe', op: '>=', threshold: 0.62, value: 0.31, passed: false },
      ],
    },
  },
}

// ReturnOverlay (task 7) also queries `/api/runs?...&kind=backtest` and batches its ids through
// `/api/runs/series` — no backtest/holdout series is fixtured here, so this stays an empty
// envelope; the overlay just renders its own honest empty state, same as the equity chart above.
const runSeriesEnvelope: ApiEnvelope<RunSeriesPayload> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: { series: {} },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders the degraded paper banner and the rest of the page when paper is null', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => {
        // Registered before the `/api/runs?` and `/api/runs/{id}` checks below — its own path
        // segment ("series") would otherwise fall through to neither and hit the `detail`
        // catch-all, handing ReturnOverlay a StrategyDetailResponse where it expects
        // `{series: {...}}` (mirrors web/backend/main.py's own route-registration-order note).
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

  // degraded-part banner carries the part_errors code
  expect(await screen.findByText(/paper state unavailable: cli_failed/)).toBeTruthy()
  // ...while the rest of the page still renders
  expect(screen.getByText('mom_breakout')).toBeTruthy()
  expect(screen.getByText('breakout over rolling high')).toBeTruthy()
  expect(screen.getByText('idea → backtested')).toBeTruthy()
  // The gate bullet card's checks come straight from the already-fetched composite response
  // (fix round 2 — no fetch of its own), so this is available as soon as the page renders.
  expect(await screen.findByText('holdout_sharpe')).toBeTruthy()
  expect(screen.getAllByText('fail').length).toBeGreaterThanOrEqual(2) // verdict + check row
  // empty series -> chart placeholder, no crash
  expect(screen.getByText(/awaiting tick history/i)).toBeTruthy()
})

// The runs-ledger-empty regression (fix round 2) lives in its own file
// (`StrategyDetail.emptyruns.test.tsx`): the funnel-wide `/api/runs?kind=sweep_trial...` URL is
// never `strategy=`-scoped, and this file's first test above already populates that URL's
// module-level fetch cache with a non-empty response — a second, empty-runs fixture for the SAME
// URL in this file would silently read back the FIRST test's cached response (the precedent
// `TrialDistribution.render.test.tsx`'s header comment documents).
