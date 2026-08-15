import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, FleetHealth, FleetRow } from '../types'
import Fleet from './Fleet'

function row(strategy: string, stage: string, health: string, overrides: Partial<FleetRow> = {}): FleetRow {
  return {
    strategy,
    stage,
    health,
    staleness_sessions: 3,
    last_tick_error: null,
    kill_switch: { tripped: false, reason: null, global_halt: false },
    // drawdown = 1 - last_equity/peak_equity: a POSITIVE depth below peak.
    drawdown: { peak_equity: 100_000, last_equity: 88_000, drawdown: 0.12 },
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
    drawdown: { peak_equity: 50_000, last_equity: 48_500, drawdown: 0.03 },
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
    // `fleet health` builds by_health over the ALERTING rows only — never a fleet-wide
    // histogram, and it can never contain `ok`. Fleet must count `rows` instead.
    summary: { total: 3, alerting: 2, by_health: { stale: 1, halted: 1 } },
    stale_after_sessions: 2,
    rows: [...alerting, row('carry_calm', 'paper', 'ok', { staleness_sessions: 0 })],
  },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function renderFleet() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => envelope,
    })) as unknown as typeof fetch,
  )
  render(
    <MemoryRouter>
      <Fleet />
    </MemoryRouter>,
  )
}

it('renders the global-halt banner and the alerting list', async () => {
  renderFleet()
  expect(await screen.findByText(/global halt/i)).toBeTruthy()
  expect(screen.getAllByText('mom_breakout').length).toBeGreaterThanOrEqual(1)
  expect(screen.getByText('4 sess stale')).toBeTruthy()
  expect(screen.getByText('dd 12.0%')).toBeTruthy()
})

it('counts the health tiles over every row, not just the alerting ones', async () => {
  renderFleet()
  await screen.findByText('total')
  // Scope to the tiles: "ok" also appears as a health badge in the per-strategy list below.
  const tiles = [...document.querySelectorAll('.metric-tile')]
  const tileFor = (label: string) =>
    tiles.find((t) => t.querySelector('.metric-label')?.textContent === label)
  // `ok` is absent from summary.by_health but present in rows — it must still tile.
  expect(tileFor('ok')?.querySelector('.metric-value')?.textContent).toBe('1')
  const total = Number(tileFor('total')?.querySelector('.metric-value')?.textContent)
  const perHealth = tiles
    .filter((t) =>
      ['halted', 'drift', 'stale', 'idle', 'ok'].includes(
        t.querySelector('.metric-label')?.textContent ?? '',
      ),
    )
    .reduce((n, t) => n + Number(t.querySelector('.metric-value')?.textContent), 0)
  expect(perHealth).toBe(total)
})

it('lists every strategy, each linking to its detail page', async () => {
  renderFleet()
  const link = (await screen.findByText('carry_calm')).closest('a')
  expect(link?.getAttribute('href')).toBe('/s/carry_calm')
})
