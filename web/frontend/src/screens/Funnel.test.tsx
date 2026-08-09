import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, FleetHealth, ListPayload, StrategyRecord } from '../types'
import Funnel from './Funnel'

function rec(name: string, stage: string): StrategyRecord {
  return {
    id: 1,
    name,
    stage,
    family: 'momentum',
    tags: [],
    author: 'agent',
    hypothesis_status: 'untested',
    derived_from: null,
    description: null,
  }
}

const strategiesEnvelope: ApiEnvelope<ListPayload<StrategyRecord>> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: { data: [rec('live_one', 'live'), rec('idea_one', 'idea'), rec('idea_two', 'idea')] },
}

const fleetEnvelope: ApiEnvelope<FleetHealth> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    ok: true,
    global_halt: false,
    alerting: [],
    summary: { total: 1, alerting: 0, by_health: { ok: 1 } },
    stale_after_sessions: 2,
    rows: [
      {
        strategy: 'live_one',
        stage: 'live',
        health: 'ok',
        staleness_sessions: 0,
        last_tick_error: null,
        kill_switch: null,
        drawdown: null,
        positions: 1,
        n_orders: 3,
      },
    ],
  },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('groups strategies by stage in lifecycle order and merges fleet health badges', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => (url.startsWith('/api/strategies') ? strategiesEnvelope : fleetEnvelope),
    })) as unknown as typeof fetch,
  )

  render(
    <MemoryRouter>
      <Funnel />
    </MemoryRouter>,
  )

  const ideaHeader = await screen.findByText('idea (2)')
  const liveHeader = screen.getByText('live (1)')
  // lifecycle order: idea before live in the document
  expect(
    ideaHeader.compareDocumentPosition(liveHeader) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy()
  expect(screen.getByText('idea_one')).toBeTruthy()
  expect(screen.getByText('idea_two')).toBeTruthy()
  // fleet-matched strategy gets a health badge; unmatched ones get none
  expect(await screen.findByText('ok')).toBeTruthy()
  expect(screen.queryByText('unknown')).toBeNull()
})
