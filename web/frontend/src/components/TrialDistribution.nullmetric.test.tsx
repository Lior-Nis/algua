/** Own file — see `TrialDistribution.render.test.tsx`'s header comment: the funnel-wide trial
 * list URL is never `strategy=`-scoped, so it must be isolated per fixture across files. */
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

it('excludes a NULL-metric trial and reports the exclusion count — never plots it at zero', async () => {
  const trials = [
    trial({ id: 1, strategy_name: 'has_metric', mean_window_sharpe: 0.3 }),
    trial({ id: 2, strategy_name: 'null_metric', mean_window_sharpe: null }),
  ]
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => listEnvelope(trials) })) as unknown as typeof fetch,
  )
  render(<TrialDistribution strategy="alpha" />)
  await screen.findAllByTestId('trial-point')
  expect(screen.getAllByTestId('trial-point').length).toBe(1)
  expect(screen.getByText(/1 trial.*excluded/i)).toBeTruthy()
})
