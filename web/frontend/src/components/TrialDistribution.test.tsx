import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
import TrialDistribution, {
  buildHoldoutStripGeometry,
  buildTrialCloudGeometry,
} from './TrialDistribution'

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

describe('buildTrialCloudGeometry (pure layout, mark 1 — mean_window_sharpe only)', () => {
  it('excludes a trial with no mean_window_sharpe and counts it — never plots it at 0', () => {
    const trials = [
      trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 }),
      trial({ id: 2, strategy_name: 'b', mean_window_sharpe: null }),
    ]
    const geo = buildTrialCloudGeometry(trials, null)
    expect(geo.points.map((p) => p.id)).toEqual([1])
    expect(geo.excludedCount).toBe(1)
    expect(geo.points.some((p) => p.value === 0)).toBe(false)
  })

  it('an empty trial list produces zero points, zero exclusions, and no own marker', () => {
    const geo = buildTrialCloudGeometry([], null)
    expect(geo.points).toEqual([])
    expect(geo.excludedCount).toBe(0)
    expect(geo.own).toBeNull()
  })

  it('never fabricates an own marker when the own mean_window_sharpe is missing/non-finite', () => {
    const trials = [trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 })]
    expect(buildTrialCloudGeometry(trials, null).own).toBeNull()
    expect(buildTrialCloudGeometry(trials, undefined).own).toBeNull()
    expect(buildTrialCloudGeometry(trials, Number.NaN).own).toBeNull()
  })

  it('extends the domain to include an own marker far outside the trial cluster', () => {
    const trials = [
      trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.1 }),
      trial({ id: 2, strategy_name: 'b', mean_window_sharpe: 0.3 }),
    ]
    const geo = buildTrialCloudGeometry(trials, 2.5)
    expect(geo.domain.max).toBeGreaterThanOrEqual(2.5)
    expect(geo.own?.value).toBe(2.5)
  })

  it('jitters a trial point deterministically — same id always lands at the same y', () => {
    const trials = [trial({ id: 7, strategy_name: 'a', mean_window_sharpe: 0.4 })]
    const geo1 = buildTrialCloudGeometry(trials, null)
    const geo2 = buildTrialCloudGeometry(trials, null)
    expect(geo1.points[0].cy).toBe(geo2.points[0].cy)
  })

  it('points are monotonic in x with their value (higher sharpe -> further right)', () => {
    const trials = [
      trial({ id: 1, strategy_name: 'lo', mean_window_sharpe: -0.5 }),
      trial({ id: 2, strategy_name: 'hi', mean_window_sharpe: 1.2 }),
    ]
    const geo = buildTrialCloudGeometry(trials, null)
    const lo = geo.points.find((p) => p.id === 1)!
    const hi = geo.points.find((p) => p.id === 2)!
    expect(hi.cx).toBeGreaterThan(lo.cx)
  })
})

describe('buildHoldoutStripGeometry (pure layout, mark 2 — holdout-class only)', () => {
  it('produces nothing when neither the bar nor the own result is available', () => {
    const geo = buildHoldoutStripGeometry(null, null)
    expect(geo.bar).toBeNull()
    expect(geo.own).toBeNull()
    expect(geo.domain).toBeNull()
  })

  it('never fabricates the bar when effective_min_holdout_sharpe is missing/non-finite', () => {
    expect(buildHoldoutStripGeometry(null, { value: 0.5, passed: true }).bar).toBeNull()
    expect(
      buildHoldoutStripGeometry(Number.POSITIVE_INFINITY, { value: 0.5, passed: true }).bar,
    ).toBeNull()
  })

  it('never fabricates the own point when its value is missing/non-finite', () => {
    expect(buildHoldoutStripGeometry(0.5, { value: null, passed: null }).own).toBeNull()
    expect(buildHoldoutStripGeometry(0.5, { value: Number.NaN, passed: false }).own).toBeNull()
    expect(buildHoldoutStripGeometry(0.5, null).own).toBeNull()
  })

  it('scales to its OWN pair regardless of how far apart the bar and result sit — the real ' +
    'case this chart exists to show (holdout 0.025 vs a bar of 2.677)', () => {
    const geo = buildHoldoutStripGeometry(2.677, { value: 0.025, passed: false })
    expect(geo.domain?.max).toBeGreaterThanOrEqual(2.677)
    expect(geo.domain?.min).toBeLessThanOrEqual(0.025)
    expect(geo.bar?.value).toBe(2.677)
    expect(geo.own?.value).toBe(0.025)
    expect(geo.own?.passed).toBe(false)
    expect(geo.bar!.cx).toBeGreaterThan(geo.own!.cx)
  })
})

// -------------------------------------------------------------------------------------------
// Component-level tests

function listEnvelope(runs: RunRow[]): ApiEnvelope<RunsListPayload> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: { count: runs.length, runs } }
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
  const calls: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      calls.push(url)
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([]) }
    }) as unknown as typeof fetch,
  )
  render(<TrialDistribution strategy="alpha" />)
  await screen.findAllByTestId('trial-point')

  const trialListCall = calls.find((u) => u.includes('kind=sweep_trial'))
  expect(trialListCall).toBeDefined()
  expect(trialListCall).not.toContain('strategy=')
})

// The remaining component-level scenarios (bar+marker rendering, the empty state, NULL
// exclusion, the tying caption, the truncation notice, aria-labels) each live in their OWN file:
// the funnel-wide trial list URL is IDENTICAL across every strategy (never `strategy=`-scoped,
// by design), and `api.ts`'s fetch cache is module-level and keyed by URL while vitest isolates
// the module graph per FILE — a second `/api/runs?kind=sweep_trial...` fixture in this same file
// would silently read back the FIRST test's cached response instead of its own stub (see
// `ScatterISOOS.empty.test.tsx` for the same precedent on the `kind=gate` list).
