import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, FleetHealth, FleetRow } from '../types'
import Home from './Home'

function row(strategy: string, stage: string, health: string, overrides: Partial<FleetRow> = {}): FleetRow {
  return {
    strategy,
    stage,
    health,
    staleness_sessions: 3,
    last_tick_error: null,
    kill_switch: { tripped: false, reason: null, global_halt: false },
    drawdown: { peak_equity: 100_000, last_equity: 88_000, drawdown: -0.12 },
    positions: 2,
    n_orders: 5,
    ...overrides,
  }
}

const alerting = [
  row('mom_breakout', 'live', 'halted', {
    kill_switch: { tripped: true, reason: 'drawdown', global_halt: true },
  }),
  row('mean_rev_pairs', 'paper', 'stale', {
    staleness_sessions: 4,
    drawdown: { peak_equity: 50_000, last_equity: 48_500, drawdown: -0.03 },
  }),
]

const envelope: ApiEnvelope<FleetHealth> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    ok: false,
    global_halt: true,
    alerting,
    summary: { total: 3, alerting: 2, by_health: { ok: 1, stale: 1, halted: 1 } },
    stale_after_sessions: 2,
    rows: [...alerting, row('carry_calm', 'paper', 'ok', { staleness_sessions: 0 })],
  },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders the global-halt banner and the alerting list from /api/fleet', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => envelope,
    })) as unknown as typeof fetch,
  )

  render(<Home />)

  expect(await screen.findByText(/global halt/i)).toBeTruthy()
  expect(screen.getByText('mom_breakout')).toBeTruthy()
  expect(screen.getByText('mean_rev_pairs')).toBeTruthy()
  // "halted" appears both as a summary-tile label and as the health badge
  expect(screen.getAllByText('halted').length).toBeGreaterThanOrEqual(2)
  expect(screen.getByText('4 sess stale')).toBeTruthy()
  expect(screen.getByText('dd -12.0%')).toBeTruthy()
  expect(vi.mocked(fetch)).toHaveBeenCalledWith('/api/fleet', expect.anything())
})
