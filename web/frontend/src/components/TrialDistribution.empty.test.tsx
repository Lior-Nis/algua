/** Own file — see `TrialDistribution.render.test.tsx`'s header comment: the funnel-wide trial
 * list URL is never `strategy=`-scoped, so it must be isolated per fixture across files. */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
import TrialDistribution from './TrialDistribution'

function listEnvelope(runs: RunRow[]): ApiEnvelope<RunsListPayload> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: { count: runs.length, runs } }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders the honest empty state when there are no funnel-wide sweep trials — no svg drawn around nothing', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => listEnvelope([]) })) as unknown as typeof fetch,
  )
  const { container } = render(<TrialDistribution strategy="never_swept" />)
  expect(await screen.findByText(/no sweep trials recorded yet/i)).toBeTruthy()
  expect(container.querySelector('svg')).toBeNull()
})
