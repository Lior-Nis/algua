/** Own file — see `TrialDistribution.render.test.tsx`'s header comment: the funnel-wide trial
 * list URL is never `strategy=`-scoped, so it must be isolated per fixture across files.
 *
 * Spec §4.4: "A silently truncated trial set would make the funnel-wide distribution lie about
 * the breadth it depicts, so view 3 must render a truncation notice rather than a partial
 * histogram." The API caps a page at 500 rows and the envelope's `count` is just the returned
 * row count (never a funnel-wide total), so `runs.length === 500` is the only honest trigger —
 * unreachable on the operator's real database today (0 rows), certain once the funnel grows. */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
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

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders a truncation notice when the funnel-wide trial list comes back at exactly the API' +
  ' page cap (500) — an honest signal that more trials may exist, never a silent partial view', async () => {
  const trials = Array.from({ length: 500 }, (_, i) =>
    trial({ id: i + 1, strategy_name: `s${i}`, mean_window_sharpe: 0.01 * i }),
  )
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([]) }
    }) as unknown as typeof fetch,
  )

  render(<TrialDistribution strategy="whatever" />)
  await screen.findAllByTestId('trial-point')
  expect(screen.getByTestId('truncation-notice').textContent).toMatch(/500/i)
})
