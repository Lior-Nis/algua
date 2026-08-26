/** Own file — see `TrialDistribution.render.test.tsx`'s header comment: the funnel-wide trial
 * list URL is never `strategy=`-scoped, so it must be isolated per fixture across files. */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, GateCheck, RunDetail, RunRow, RunsListPayload } from '../types'
import TrialDistribution from './TrialDistribution'

function trial(overrides: Partial<RunRow> & { id: number; strategy_name: string }): RunRow {
  return {
    kind: 'sweep_trial',
    strategy_id: 1,
    created_at: '2026-08-25T00:00:00+00:00',
    passed: null,
    mean_window_sharpe: null,
    sharpe_oos: null,
    ...overrides,
  }
}

function listEnvelope(runs: RunRow[]): ApiEnvelope<RunsListPayload> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: { count: runs.length, runs } }
}

function gateRow(id: number, strategy: string): RunRow {
  return {
    id,
    kind: 'gate',
    strategy_name: strategy,
    strategy_id: 1,
    created_at: '2026-08-25T00:00:00+00:00',
    passed: 0,
    mean_window_sharpe: null,
    sharpe_oos: null,
  }
}

function gateDetail(id: number, strategy: string, checks: GateCheck[], effectiveFloor: number): RunDetail {
  return {
    ...gateRow(id, strategy),
    extra_metrics: {},
    gate_decision: { passed: false, checks, effective_min_holdout_sharpe: effectiveFloor },
  }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it("the strip's aria-label spells out the advisory qualifier in plain words — a screen-reader " +
  'user gets no shape/colour/position cue at all, so the text is the whole encoding', async () => {
  const trials = Array.from({ length: 5 }, (_, i) =>
    trial({ id: i + 1, strategy_name: `s${i}`, mean_window_sharpe: 0.1 * i }),
  )
  const checks: GateCheck[] = [
    { name: 'holdout_sharpe', op: '>=', threshold: 2.677, value: 0.025, passed: false, advisory: true },
  ]
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return { ok: true, status: 200, json: async () => ({
          ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false,
          data: gateDetail(60, 'aria_probe', checks, 2.677),
        }) }
      }
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([gateRow(60, 'aria_probe')]) }
    }) as unknown as typeof fetch,
  )

  render(<TrialDistribution strategy="aria_probe" />)
  await screen.findByTestId('own-marker')
  const svg = screen.getByTestId('trial-strip-svg')
  const ariaLabel = svg.getAttribute('aria-label') ?? ''
  expect(ariaLabel).toMatch(/advisory check, does not veto the gate/i)
  expect(ariaLabel).toMatch(/below the deflated bar/i)
})
