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

it('omits the deflation strip entirely when this strategy has no gate evaluation yet — never ' +
  'an axis drawn with nothing on it', async () => {
  const trials = [trial({ id: 1, strategy_name: 'a', mean_window_sharpe: 0.4 })]
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([]) }
    }) as unknown as typeof fetch,
  )

  render(<TrialDistribution strategy="ungated" />)
  await screen.findAllByTestId('trial-point')
  expect(screen.queryByTestId('trial-strip-svg')).toBeNull()
  expect(screen.queryByTestId('trial-dist-summary')).toBeNull()
})
