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

it("each mark names its OWN measurement — the cloud's caption says mean window sharpe, the " +
  "strip's caption says holdout sharpe — since the two are on independent axes and a reader " +
  'must not read them as one statistic (fix round 2: no more single shared caption)', async () => {
  const trials = [trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 })]
  const checks: GateCheck[] = [
    { name: 'holdout_sharpe', op: '>=', threshold: 0.5, value: 0.6, passed: true, advisory: true },
  ]
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return { ok: true, status: 200, json: async () => ({
          ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false,
          data: gateDetail(70, 'caption_probe', checks, 0.5),
        }) }
      }
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([gateRow(70, 'caption_probe')]) }
    }) as unknown as typeof fetch,
  )
  render(<TrialDistribution strategy="caption_probe" />)
  await screen.findAllByTestId('trial-point')
  await screen.findByTestId('trial-strip-svg')

  const cloudCaption = screen.getByTestId('trial-cloud-svg').querySelector('.trial-dist-caption')
  expect(cloudCaption?.textContent).toMatch(/mean window sharpe/i)
  expect(cloudCaption?.textContent).not.toMatch(/holdout sharpe/i)

  const stripCaption = screen.getByTestId('trial-strip-svg').querySelector('.trial-dist-caption')
  expect(stripCaption?.textContent).toMatch(/holdout sharpe/i)
  expect(stripCaption?.textContent).not.toMatch(/mean window sharpe/i)

  // The prose caption is what ties the two independently-scaled marks together — the causal
  // story (breadth produces the bar), stated once, in words, not implied by shared geometry.
  const summary = screen.getByTestId('trial-dist-summary').textContent ?? ''
  expect(summary).toMatch(/trial.*of search raised the bar/i)
})
