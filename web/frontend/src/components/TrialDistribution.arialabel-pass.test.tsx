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

it("a PASSING advisory marker's aria-label still names the deflated bar clearly (not just the " +
  'fail case)', async () => {
  const trials = [trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 })]
  const checks: GateCheck[] = [
    { name: 'holdout_sharpe', op: '>=', threshold: 0.5, value: 1.4, passed: true, advisory: true },
  ]
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return { ok: true, status: 200, json: async () => ({
          ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false,
          data: gateDetail(61, 'aria_probe_pass', checks, 0.5),
        }) }
      }
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([gateRow(61, 'aria_probe_pass')]) }
    }) as unknown as typeof fetch,
  )

  render(<TrialDistribution strategy="aria_probe_pass" />)
  await screen.findByTestId('own-marker')
  const svg = document.querySelector('svg.trial-dist-svg') as SVGElement
  const ariaLabel = svg.getAttribute('aria-label') ?? ''
  expect(ariaLabel).toMatch(/clears the deflated bar/i)
  expect(ariaLabel).toMatch(/advisory check, does not veto the gate/i)
})
