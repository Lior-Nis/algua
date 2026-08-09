import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ActivityRow, ApiEnvelope, ListPayload } from '../types'
import Activity from './Activity'

const envelope: ApiEnvelope<ListPayload<ActivityRow>> = {
  ok: true,
  fetched_at: '2026-08-09T14:30:00Z',
  stale: false,
  data: {
    data: [
      {
        id: 2,
        ts: '2026-08-09T14:00:00+00:00',
        actor: 'agent',
        action: 'kill_switch_trip',
        strategy: 'mom_breakout',
        reason: 'drawdown breach',
      },
      {
        id: 1,
        ts: '2026-08-08T21:00:00+00:00',
        actor: 'system',
        action: 'global_halt_engage',
        strategy: null,
        reason: null,
      },
    ],
  },
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('renders activity rows with actor chips, UTC times, and date dividers', async () => {
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
      <Activity />
    </MemoryRouter>,
  )

  expect(await screen.findByText('kill_switch_trip')).toBeTruthy()
  expect(screen.getByText('global_halt_engage')).toBeTruthy()
  // "agent" appears as both a filter chip and the row's actor chip
  expect(screen.getAllByText('agent').length).toBeGreaterThanOrEqual(2)
  expect(screen.getByText('drawdown breach')).toBeTruthy()
  expect(screen.getByText('mom_breakout')).toBeTruthy()
  // UTC HH:MM:SS, plus a date divider per UTC day (the two rows span two days)
  expect(screen.getByText('14:00:00')).toBeTruthy()
  expect(screen.getByText('2026-08-09')).toBeTruthy()
  expect(screen.getByText('2026-08-08')).toBeTruthy()
  expect(vi.mocked(fetch)).toHaveBeenCalledWith('/api/activity?limit=100', expect.anything())
})
