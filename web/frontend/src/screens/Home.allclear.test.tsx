/** Own file on purpose: api.ts's fetch cache is module-level and keyed by URL, and
 * vitest isolates the module graph per FILE — so a second /api/fleet fixture can only
 * be exercised from a separate file. */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, FleetHealth, FleetRow } from '../types'
import Home from './Home'

function row(strategy: string, stage: string, health: string): FleetRow {
  return {
    strategy,
    stage,
    health,
    staleness_sessions: 0,
    last_tick_error: null,
    kill_switch: { tripped: false, reason: null, global_halt: false },
    drawdown: { peak_equity: null, last_equity: null, drawdown: null },
    positions: 0,
    n_orders: 0,
  }
}

// A realistic quiet fleet: one ticking paper strategy, plus research-stage rows that are
// `idle` BY DESIGN — no operator loop ticks them, so they are never alerting and never ok.
const quiet: ApiEnvelope<FleetHealth> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    ok: true,
    global_halt: false,
    alerting: [],
    summary: { total: 3, alerting: 0, by_health: {} },
    stale_after_sessions: 5,
    rows: [
      row('carry_calm', 'paper', 'ok'),
      row('draft_a', 'idea', 'idle'),
      row('draft_b', 'backtested', 'idle'),
    ],
  },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('reports the real ok count instead of calling every registered strategy ok', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => quiet,
    })) as unknown as typeof fetch,
  )

  render(<Home />)

  expect(await screen.findByText(/no alerting strategies/)).toBeTruthy()
  expect(screen.getByText(/1 of 3 ok/)).toBeTruthy()
  expect(screen.queryByText(/3 strategies ok/)).toBeNull()
})
