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

function gateRow(id: number, strategy: string, ownMeanWindowSharpe: number | null): RunRow {
  return {
    id,
    kind: 'gate',
    strategy_name: strategy,
    strategy_id: 1,
    created_at: '2026-08-25T00:00:00+00:00',
    passed: 0,
    mean_window_sharpe: ownMeanWindowSharpe,
    sharpe_oos: null,
  }
}

function gateDetail(row: RunRow, checks: GateCheck[], effectiveFloor: number): RunDetail {
  return {
    ...row,
    extra_metrics: {},
    gate_decision: { passed: false, checks, effective_min_holdout_sharpe: effectiveFloor },
  }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders both independently-scaled marks: the trial cloud (own mean_window_sharpe marked) ' +
  'and the deflation strip (bar vs holdout result), each distinguishable from a plain trial dot', async () => {
  const trials = Array.from({ length: 10 }, (_, i) =>
    trial({ id: i + 1, strategy_name: `s${i}`, mean_window_sharpe: 0.1 * i }),
  )
  const checks: GateCheck[] = [
    { name: 'holdout_sharpe', op: '>=', threshold: 2.677, value: 0.025, passed: false, advisory: true },
  ]
  const row = gateRow(55, 'mom_breakout', 1.4)
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return { ok: true, status: 200, json: async () => ({
          ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false,
          data: gateDetail(row, checks, 2.677),
        }) }
      }
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      // kind=gate lookup
      return { ok: true, status: 200, json: async () => listEnvelope([row]) }
    }) as unknown as typeof fetch,
  )

  render(<TrialDistribution strategy="mom_breakout" />)

  // Mark 1 — the trial cloud: N plain trial dots (circles) plus this strategy's own
  // mean_window_sharpe as a distinct (non-circle) marker.
  const points = await screen.findAllByTestId('trial-point')
  expect(points.length).toBe(10)
  points.forEach((p) => expect(p.tagName.toLowerCase()).toBe('circle'))

  const ownCloudMarker = await screen.findByTestId('own-cloud-marker')
  expect(ownCloudMarker.tagName.toLowerCase()).not.toBe('circle')
  expect(screen.getByTestId('own-cloud-marker-label').textContent).toMatch(/1\.4/)

  // Mark 2 — the deflation strip: the bar, direct-labelled, and the holdout-class marker,
  // direct-labelled and distinguishable from the cloud's own marker.
  const bar = await screen.findByTestId('bar-line')
  expect(bar).toBeTruthy()
  expect(screen.getByTestId('bar-label').textContent).toMatch(/2\.68|2\.677/)

  const marker = screen.getByTestId('own-marker')
  expect(marker).toBeTruthy()
  expect(marker.tagName.toLowerCase()).not.toBe('circle')
  expect(screen.getByTestId('own-marker-label').textContent).toMatch(/0\.03|0\.025/)
  // The marker's own advisory check never vetoes the OVERALL gate (types.ts: an advisory fail
  // "must never render like a failed binding floor") — the fail diamond must carry the word
  // "advisory" in its direct label, since colour/shape alone would read exactly like one.
  expect(screen.getByTestId('own-marker-label').textContent).toMatch(/advisory/i)
  // Fix round 3: colour follows the ENTITY. The diamond itself carries this strategy's identity
  // class — the SAME one its marker in the trial cloud carries — never a verdict class, so
  // Electric cannot mean "this strategy" in one mark and something else in the other.
  expect(marker.getAttribute('class')).toBe('trial-dist-own-strip-marker')
  expect(screen.getByTestId('own-cloud-marker').getAttribute('class')).toBe(
    'trial-dist-own-cloud-marker',
  )
  // The advisory verdict (0.025 < 2.677) survives on the LABEL, in words plus a status tint.
  expect(screen.getByTestId('own-marker-label').getAttribute('class')).toContain('fail')

  // The two marks are genuinely SEPARATE SVGs (independent coordinate spaces), not one shared
  // axis split by CSS.
  expect(screen.getByTestId('trial-cloud-svg')).not.toBe(screen.getByTestId('trial-strip-svg'))

  // The tying caption states the causal story.
  const summary = screen.getByTestId('trial-dist-summary').textContent ?? ''
  expect(summary).toMatch(/10 trials of search/i)
  expect(summary).toMatch(/2\.68|2\.677/)
  expect(summary).toMatch(/0\.03|0\.025/)
})
