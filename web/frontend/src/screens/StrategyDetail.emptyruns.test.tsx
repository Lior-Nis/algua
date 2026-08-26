/** Own file: the funnel-wide `/api/runs?kind=sweep_trial...` URL (never `strategy=`-scoped) is
 * identical across every test that renders `StrategyDetail`, and `api.ts`'s fetch cache is
 * module-level and keyed by URL while vitest isolates the module graph per FILE — a second,
 * distinct `runs`-ledger fixture for that URL needs its own file (see
 * `TrialDistribution.render.test.tsx`'s header comment for the same precedent).
 *
 * Fix round 2 regression: on the operator's real database, `gate_evaluations` has real rows
 * while the `runs` ledger (v44) has ZERO — spec Q8 has no backfill. Before the fix,
 * `GateBulletCard` independently fetched `/api/runs?...&kind=gate&limit=1`, found nothing on a
 * runs-empty database, and rendered "no gate checks recorded yet" OVER a real, passing/failing
 * gate header — the per-check breakdown for every historical evaluation was unreachable. Feeding
 * the card `row.decision.checks` directly makes it immune to the `runs` ledger being empty. */
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type {
  ApiEnvelope,
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
    transitions: [],
  },
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
  part_errors: {},
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

// The `runs` ledger is EMPTY across the board — TrialDistribution's own fetch chain (kind=
// sweep_trial, kind=gate) and ReturnOverlay's (kind=backtest, kind=gate) all find nothing, but
// GateBulletCard must not depend on the `runs` ledger at all.
const emptyRunsEnvelope: ApiEnvelope<RunsListPayload> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: { count: 0, runs: [] },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders the gate per-check breakdown even when the runs ledger has zero rows (gate_evaluations non-empty)', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => {
        if (url.endsWith('/series') && !url.startsWith('/api/runs/series')) return seriesEnvelope
        if (url.startsWith('/api/runs')) return emptyRunsEnvelope
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

  // The per-check breakdown renders from `row.decision.checks` — no waterfall required.
  expect(await screen.findByText('holdout_sharpe')).toBeTruthy()
  expect(screen.queryByText(/no gate checks recorded yet/i)).toBeNull()
  // TrialDistribution degrades honestly instead (its own fetch chain really did find nothing).
  expect(await screen.findByText(/no sweep trials recorded yet/i)).toBeTruthy()
})
