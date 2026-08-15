import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, BookPayload } from '../types'
import Money from './Money'

const book: ApiEnvelope<BookPayload> = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: {
    ok: false,
    capacity: 64,
    allocated: 2,
    count_headroom: 62,
    sum_allocations: 3142.84,
    live_allocated: 0,
    unallocated_operational: [
      {
        strategy: 'liquidity_stable_quality_momentum',
        stage: 'paper',
        since: '2026-08-14T12:12:43+00:00',
        ever_ticked: false,
      },
    ],
    slices: [
      { strategy: 'up', stage: 'paper', capital: 1571.42, last_equity: 1650.0,
        effective_ts: '2026-08-14T10:47:51+00:00', actor: 'agent', equity_error: null },
      { strategy: 'never_ticked', stage: 'paper', capital: 1571.42, last_equity: null,
        effective_ts: '2026-08-14T11:08:38+00:00', actor: 'agent', equity_error: null },
    ],
  },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function renderMoney() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => book })) as unknown as typeof fetch,
  )
  render(
    <MemoryRouter>
      <Money />
    </MemoryRouter>,
  )
}

it('surfaces a strategy stranded without capital and says why it matters', async () => {
  renderMoney()
  expect(await screen.findByText('liquidity_stable_quality_momentum')).toBeTruthy()
  expect(screen.getByText(/never ticked/)).toBeTruthy()
  expect(screen.getByText(/the operator loop skips them/)).toBeTruthy()
})

it('states that capital headroom is unknown rather than implying it is zero', async () => {
  renderMoney()
  // The account equity needs a broker call this view must never make.
  expect(await screen.findByText(/never calls the broker/)).toBeTruthy()
  expect(screen.getByText(/62 more tenants fit/)).toBeTruthy()
})

it('renders no p&l for a slice that has never ticked, instead of claiming break-even', async () => {
  renderMoney()
  const row = (await screen.findByText('never_ticked')).closest('tr')!
  // "0.0%" would assert the strategy traded and returned nothing. It has not traded.
  expect(row.textContent).not.toContain('0.0%')
  expect(row.textContent).toContain('—')
})

it('colours a positive slice return without relying on colour alone', async () => {
  renderMoney()
  const row = (await screen.findByText('up')).closest('tr')!
  expect(row.textContent).toContain('5.0%')
})
