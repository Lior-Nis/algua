/** Own file: `/api/runs?kind=sweep_trial...` is NEVER `strategy=`-scoped (funnel-wide by
 * design), so every test in this component's suite that stubs a trial list uses the SAME url —
 * `api.ts`'s fetch cache is module-level and keyed by URL, and vitest isolates the module graph
 * per FILE, so each distinct trial-list fixture needs its own file (see
 * `TrialDistribution.test.tsx`'s comment, and the `ScatterISOOS.empty.test.tsx` precedent). */
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

it('renders the threshold rule, direct-labelled, and the strategy holdout marker, ' +
  'direct-labelled and distinguishable from trial points', async () => {
  const trials = Array.from({ length: 10 }, (_, i) =>
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
          data: gateDetail(55, 'mom_breakout', checks, 2.677),
        }) }
      }
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      // kind=gate lookup
      return { ok: true, status: 200, json: async () => listEnvelope([gateRow(55, 'mom_breakout')]) }
    }) as unknown as typeof fetch,
  )

  render(<TrialDistribution strategy="mom_breakout" />)

  const rule = await screen.findByTestId('threshold-line')
  expect(rule).toBeTruthy()
  expect(screen.getByTestId('threshold-label').textContent).toMatch(/2\.68|2\.677/)

  const marker = screen.getByTestId('own-marker')
  expect(marker).toBeTruthy()
  expect(marker.tagName.toLowerCase()).not.toBe('circle') // trial points are circles; marker is not
  expect(screen.getByTestId('own-marker-label').textContent).toMatch(/0\.03|0\.025/)
  // The marker's own advisory check never vetoes the OVERALL gate (types.ts: an advisory fail
  // "must never render like a failed binding floor") — the fail-red diamond must carry the word
  // "advisory" in its direct label, since colour/shape alone would read exactly like one.
  expect(screen.getByTestId('own-marker-label').textContent).toMatch(/advisory/i)

  const points = screen.getAllByTestId('trial-point')
  expect(points.length).toBe(10)
  points.forEach((p) => expect(p.tagName.toLowerCase()).toBe('circle'))

  // The marker fails (0.025 < 2.677): rendered with the fail status token, not the neutral
  // trial-point fill.
  expect(marker.getAttribute('class')).toContain('fail')
})
