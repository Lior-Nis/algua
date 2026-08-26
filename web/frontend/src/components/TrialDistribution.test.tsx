import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ApiEnvelope, GateCheck, RunDetail, RunRow, RunsListPayload } from '../types'
import TrialDistribution, { buildTrialDistributionGeometry } from './TrialDistribution'

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

describe('buildTrialDistributionGeometry (pure layout)', () => {
  it('excludes a trial with no mean_window_sharpe and counts it — never plots it at 0', () => {
    const trials = [
      trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 }),
      trial({ id: 2, strategy_name: 'b', mean_window_sharpe: null }),
    ]
    const geo = buildTrialDistributionGeometry(trials, null, null)
    expect(geo.points.map((p) => p.id)).toEqual([1])
    expect(geo.excludedCount).toBe(1)
    expect(geo.points.some((p) => p.value === 0)).toBe(false)
  })

  it('an empty trial list produces zero points and zero exclusions', () => {
    const geo = buildTrialDistributionGeometry([], null, null)
    expect(geo.points).toEqual([])
    expect(geo.excludedCount).toBe(0)
    expect(geo.threshold).toBeNull()
    expect(geo.marker).toBeNull()
  })

  it('never fabricates a threshold when effective_min_holdout_sharpe is missing/non-finite', () => {
    const trials = [trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 })]
    expect(buildTrialDistributionGeometry(trials, null, null).threshold).toBeNull()
    expect(buildTrialDistributionGeometry(trials, undefined, null).threshold).toBeNull()
    expect(buildTrialDistributionGeometry(trials, Number.POSITIVE_INFINITY, null).threshold).toBeNull()
  })

  it('never fabricates a marker when the own holdout value is missing/non-finite', () => {
    const trials = [trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 })]
    expect(buildTrialDistributionGeometry(trials, 0.5, null).marker).toBeNull()
    expect(buildTrialDistributionGeometry(trials, 0.5, { value: null, passed: null }).marker).toBeNull()
    expect(
      buildTrialDistributionGeometry(trials, 0.5, { value: Number.NaN, passed: false }).marker,
    ).toBeNull()
  })

  it('extends the domain to include a threshold/marker far outside the trial cluster — the ' +
    'real case this chart exists to show (holdout 0.025 vs a bar of 2.677)', () => {
    const trials = [
      trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.1 }),
      trial({ id: 2, strategy_name: 'b', mean_window_sharpe: 0.3 }),
    ]
    const geo = buildTrialDistributionGeometry(trials, 2.677, { value: 0.025, passed: false })
    expect(geo.domain.max).toBeGreaterThanOrEqual(2.677)
    expect(geo.domain.min).toBeLessThanOrEqual(0.025)
    expect(geo.threshold?.value).toBe(2.677)
    expect(geo.marker?.value).toBe(0.025)
    expect(geo.marker?.passed).toBe(false)
  })

  it('jitters a trial point deterministically — same id always lands at the same y', () => {
    const trials = [trial({ id: 7, strategy_name: 'a', mean_window_sharpe: 0.4 })]
    const geo1 = buildTrialDistributionGeometry(trials, null, null)
    const geo2 = buildTrialDistributionGeometry(trials, null, null)
    expect(geo1.points[0].cy).toBe(geo2.points[0].cy)
  })

  it("places the own marker outside the trial swarm's vertical band — a distinct row, not " +
    'blended into the jittered cloud', () => {
    const trials = Array.from({ length: 20 }, (_, i) =>
      trial({ id: i + 1, strategy_name: `s${i}`, mean_window_sharpe: 0.1 * i }),
    )
    const geo = buildTrialDistributionGeometry(trials, 1.5, { value: 1.4, passed: true })
    const swarmYs = geo.points.map((p) => p.cy)
    const swarmMax = Math.max(...swarmYs)
    const swarmMin = Math.min(...swarmYs)
    expect(geo.marker).not.toBeNull()
    const markerY = geo.marker!.cy
    expect(markerY < swarmMin || markerY > swarmMax).toBe(true)
  })

  it('points are monotonic in x with their value (higher sharpe -> further right)', () => {
    const trials = [
      trial({ id: 1, strategy_name: 'lo', mean_window_sharpe: -0.5 }),
      trial({ id: 2, strategy_name: 'hi', mean_window_sharpe: 1.2 }),
    ]
    const geo = buildTrialDistributionGeometry(trials, null, null)
    const lo = geo.points.find((p) => p.id === 1)!
    const hi = geo.points.find((p) => p.id === 2)!
    expect(hi.cx).toBeGreaterThan(lo.cx)
  })
})

// -------------------------------------------------------------------------------------------
// Component-level tests

function listEnvelope(runs: RunRow[]): ApiEnvelope<RunsListPayload> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: { count: runs.length, runs } }
}

function detailEnvelope(detail: RunDetail): ApiEnvelope<RunDetail> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: detail }
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

/** Stubs the three-call fan-out: funnel-wide `/api/runs?...kind=sweep_trial...` (NEVER filtered
 * to `strategy`), the two-call gate waterfall (`kind=gate&limit=1` then `/api/runs/{id}`) for
 * THIS strategy's own holdout result. */
function stubFetch(opts: {
  strategy: string
  trials: RunRow[]
  gateRunId: number | null
  checks?: GateCheck[]
  effectiveFloor?: number
}): { calls: string[] } {
  const calls: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      calls.push(url)
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return {
          ok: true,
          status: 200,
          json: async () =>
            detailEnvelope(
              gateDetail(opts.gateRunId ?? 0, opts.strategy, opts.checks ?? [], opts.effectiveFloor ?? 0),
            ),
        }
      }
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(opts.trials) }
      }
      if (url.includes('kind=gate')) {
        const runs = opts.gateRunId !== null ? [gateRow(opts.gateRunId, opts.strategy)] : []
        return { ok: true, status: 200, json: async () => listEnvelope(runs) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([]) }
    }) as unknown as typeof fetch,
  )
  return { calls }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('fetches the FUNNEL-WIDE trial list — never scoped to this strategy', async () => {
  const trials = [
    trial({ id: 1, strategy_name: 'alpha', mean_window_sharpe: 0.2 }),
    trial({ id: 2, strategy_name: 'beta', mean_window_sharpe: 0.5 }),
  ]
  const { calls } = stubFetch({ strategy: 'alpha', trials, gateRunId: null })
  render(<TrialDistribution strategy="alpha" />)
  await screen.findAllByTestId('trial-point')

  const trialListCall = calls.find((u) => u.includes('kind=sweep_trial'))
  expect(trialListCall).toBeDefined()
  expect(trialListCall).not.toContain('strategy=')
})

// The remaining component-level scenarios (threshold+marker rendering, the empty state, NULL
// exclusion) each live in their OWN file: the funnel-wide trial list URL is IDENTICAL across
// every strategy (never `strategy=`-scoped, by design), and `api.ts`'s fetch cache is
// module-level and keyed by URL while vitest isolates the module graph per FILE — a second
// `/api/runs?kind=sweep_trial...` fixture in this same file would silently read back the FIRST
// test's cached response instead of its own stub (see `ScatterISOOS.empty.test.tsx` for the same
// precedent on the `kind=gate` list).
