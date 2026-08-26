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

it("the axis caption names BOTH statistics sharing the x-axis — the trial dots' mean window " +
  "sharpe and the threshold/marker's holdout sharpe — since a reader who hasn't read the " +
  'source cannot otherwise tell the rule/marker are a different measurement than the dots', async () => {
  const trials = [trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 })]
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => listEnvelope(trials) })) as unknown as typeof fetch,
  )
  render(<TrialDistribution strategy="caption_probe" />)
  await screen.findAllByTestId('trial-point')
  const caption = document.querySelector('.trial-dist-caption')
  expect(caption?.textContent).toMatch(/mean window sharpe/i)
  expect(caption?.textContent).toMatch(/holdout sharpe/i)
})
