import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, SeriesPayload, StrategyDetailResponse } from '../types'
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
      json: async () => (url.endsWith('/series') ? seriesEnvelope : detail),
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
  expect(screen.getByText('holdout_sharpe')).toBeTruthy()
  expect(screen.getAllByText('fail').length).toBeGreaterThanOrEqual(2) // verdict + check row
  // empty series -> chart placeholder, no crash
  expect(screen.getByText(/awaiting tick history/i)).toBeTruthy()
})
