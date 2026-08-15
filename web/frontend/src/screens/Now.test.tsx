import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ActivityRow, ApiEnvelope, ListPayload, TriagePayload } from '../types'
import Now from './Now'

/** The 2026-08-15 state that motivated the redesign: a dead loop and stranded capital, neither
 * of which the old fleet-only Home could show. */
const triage: TriagePayload = {
  ok: true,
  items: [
    {
      kind: 'loop_down',
      severity: 0,
      title: 'research loop rate limited',
      detail: 'provider usage limit reached',
      since: '2026-08-14T14:00:01+00:00',
      route: '/research',
    },
    {
      kind: 'capital_stranded',
      severity: 2,
      title: 'liquidity_stable_quality_momentum holds no capital',
      detail: 'stage paper, never ticked — the operator loop skips it',
      since: '2026-08-14T12:12:43+00:00',
      route: '/money',
    },
  ],
  sources: { fleet: true, ops: true, book: false },
  headline: {
    fleet_ok: 7,
    fleet_total: 10,
    book_allocated: 7,
    book_capacity: 64,
    loops_alerting: 1,
  },
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
}

const activity: ApiEnvelope<ListPayload<ActivityRow>> = {
  ok: true,
  fetched_at: '2026-08-15T17:04:00Z',
  stale: false,
  data: {
    data: [
      { id: 1, ts: '2026-08-15T09:00:00+00:00', actor: 'agent', action: 'gate_evaluated',
        strategy: 'momo', reason: null },
    ],
  },
}

function renderNow() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => (url.startsWith('/api/activity') ? activity : triage),
    })) as unknown as typeof fetch,
  )
  render(
    <MemoryRouter>
      <Now />
    </MemoryRouter>,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('leads with the ranked needs-you list, worst first', async () => {
  renderNow()
  expect(await screen.findByText('research loop rate limited')).toBeTruthy()
  const titles = screen.getAllByText(/loop rate limited|holds no capital/)
  // The dead machine outranks the stranded strategy.
  expect(titles[0].textContent).toContain('research loop')
})

it('routes each item to the tab that owns it', async () => {
  renderNow()
  const row = (await screen.findByText('research loop rate limited')).closest('a')
  expect(row?.getAttribute('href')).toBe('/research')
  const capital = screen.getByText(/holds no capital/).closest('a')
  expect(capital?.getAttribute('href')).toBe('/money')
})

it('names a source that failed rather than silently ranking a shorter list', async () => {
  renderNow()
  // book: false in `sources` — a partial read must never read as all-clear.
  expect(await screen.findByText(/book unavailable/)).toBeTruthy()
})

it('shows every item with the one fact that makes it actionable', async () => {
  renderNow()
  expect(await screen.findByText('provider usage limit reached')).toBeTruthy()
  expect(screen.getByText(/the operator loop skips it/)).toBeTruthy()
})

it('gives each job a headline that deep-links to its tab', async () => {
  renderNow()
  const fleetTile = (await screen.findByText('fleet ok')).closest('a')
  expect(fleetTile?.getAttribute('href')).toBe('/fleet')
  expect(within(fleetTile!).getByText('7/10')).toBeTruthy()
  expect(screen.getByText('book').closest('a')?.getAttribute('href')).toBe('/money')
})

it('absorbs the audit trail as the overnight feed', async () => {
  renderNow()
  expect(await screen.findByText('gate_evaluated')).toBeTruthy()
  expect(screen.getByText('agent')).toBeTruthy()
})
