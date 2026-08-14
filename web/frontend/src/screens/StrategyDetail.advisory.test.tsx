/** Own file: api.ts's fetch cache is module-level and keyed by URL, and vitest isolates
 * the module graph per FILE — a second /api/strategy/* fixture needs its own file. */
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, SeriesPayload, StrategyDetailResponse } from '../types'
import StrategyDetail from './StrategyDetail'

/** The NORMAL shape under the factory soft gate: the gate PASSES on its binding floors
 * while advisory statistical checks fail — `passed` ignores advisory checks entirely
 * (algua/research/gates.py). The token is minted but not yet spent. */
const detail: StrategyDetailResponse = {
  ok: true,
  strategy: 'mom_breakout',
  registry: {
    id: 1,
    name: 'mom_breakout',
    stage: 'candidate',
    family: 'momentum',
    tags: [],
    author: 'agent',
    hypothesis_status: 'untested',
    derived_from: null,
    description: null,
    transitions: [],
  },
  paper: null,
  gates: {
    strategy: 'mom_breakout',
    gate_evaluations: [
      {
        id: 9,
        passed: 1,
        consumed: 0,
        actor: 'agent',
        created_at: '2026-08-02T15:00:00+00:00',
        decision: {
          passed: true,
          checks: [
            { name: 'pit_universe', op: '==', threshold: null, value: null, passed: true },
            {
              name: 'holdout_sharpe',
              op: '>=',
              threshold: 0.62,
              value: 0.31,
              passed: false,
              advisory: true,
            },
          ],
        },
      },
    ],
    forward_gate_evaluations: [],
  },
  gates_limit: 50,
  part_errors: { paper: 'cli_failed' },
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
}

const seriesEnvelope: ApiEnvelope<SeriesPayload> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    strategy: 'mom_breakout',
    lane_filter: null,
    series: { paper: [], live: null },
    truncated: { paper: false, live: null },
    n_legacy_excluded: 0,
    n_unparseable: 0,
    n_invalid_lane: 0,
  },
}

function renderDetail() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => (url.endsWith('/series') ? seriesEnvelope : detail),
    })) as unknown as typeof fetch,
  )
  render(
    <MemoryRouter initialEntries={['/s/mom_breakout']}>
      <Routes>
        <Route path="/s/:name" element={<StrategyDetail />} />
      </Routes>
    </MemoryRouter>,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('marks a failed ADVISORY check as non-vetoing instead of rendering it like a breach', async () => {
  renderDetail()

  expect(await screen.findByText(/advisory checks are recorded but never veto/)).toBeTruthy()
  expect(screen.getByText(/· advisory/)).toBeTruthy()
  // The advisory failure must NOT wear the red of a breached binding floor.
  const marks = screen.getAllByText('fail')
  expect(marks).toHaveLength(1)
  expect(marks[0].getAttribute('style')).toContain('var(--gold)')
})

it('names the ledger by the lifecycle edge it gates', async () => {
  renderDetail()
  expect(await screen.findByText(/research gate — backtested → candidate/)).toBeTruthy()
})

it('shows whether the passing gate token has been spent', async () => {
  renderDetail()
  expect(await screen.findByText('token unspent')).toBeTruthy()
})
