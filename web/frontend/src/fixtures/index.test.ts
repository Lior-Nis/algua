import { describe, expect, it } from 'vitest'
import { FIXTURE_SENTINEL, resolveFixture } from './index'
import type {
  ApiEnvelope,
  BookPayload,
  FleetHealth,
  ListPayload,
  StrategyRecord,
  TriagePayload,
} from '../types'

// Enumerated from the useFetch/fetch() call sites across screens, components, and push.ts —
// every URL the REAL app can ask for. A new endpoint must be added here deliberately, not
// discovered as a blank screen in the demo build. Deliberately EXCLUDES `/api/__demo`, which
// exists solely as the sentinel's home and which no real screen ever fetches.
const REAL_ENDPOINT_URLS = [
  '/api/triage',
  '/api/fleet',
  '/api/book',
  '/api/ops',
  '/api/ideas',
  '/api/strategies',
  '/api/activity?limit=25',
  '/api/runs',
  '/api/runs?kind=gate&sort=sharpe_oos&limit=20',
  '/api/runs?kind=backtest&strategy=liquid10_adj_momentum&limit=1',
  '/api/runs?kind=sweep_trial&limit=200',
  '/api/runs/100',
  '/api/runs/series?ids=100,101',
  '/api/strategy/liquid10_adj_momentum',
  '/api/strategy/liquid10_adj_momentum/series',
  '/api/push/key',
]

describe('resolveFixture', () => {
  it('returns undefined for an unknown URL rather than a fabricated payload', () => {
    expect(resolveFixture('/api/nope')).toBeUndefined()
  })

  it('serves every endpoint the app actually calls', () => {
    for (const url of REAL_ENDPOINT_URLS) {
      expect(resolveFixture(url), url).toBeDefined()
    }
  })

  it('carries the sentinel on a fixture-only endpoint no real screen ever fetches, so a '
    + 'production build can be proven fixture-free without a debug string leaking onto a '
    + 'rendered screen', () => {
    expect(JSON.stringify(resolveFixture('/api/__demo'))).toContain(FIXTURE_SENTINEL)
  })

  it('never leaks the sentinel into a payload a real screen actually renders — the fixture-only '
    + '/api/__demo route above is the sentinel\'s ONLY home', () => {
    for (const url of REAL_ENDPOINT_URLS) {
      expect(JSON.stringify(resolveFixture(url)), url).not.toContain(FIXTURE_SENTINEL)
    }
  })

  it('is rich enough to exercise the design: >= 10 fleet rows across >= 3 stages, and the '
    + 'row count matches /api/strategies exactly (fleet_status emits one row per REGISTRY '
    + 'strategy, not just the ticked ones — a subset here would be silent drift)', () => {
    const fleet = resolveFixture('/api/fleet') as ApiEnvelope<FleetHealth>
    const strategies = resolveFixture('/api/strategies') as ApiEnvelope<ListPayload<StrategyRecord>>
    expect(fleet.data.rows.length).toBeGreaterThanOrEqual(10)
    expect(new Set(fleet.data.rows.map((r) => r.stage)).size).toBeGreaterThanOrEqual(3)
    expect(fleet.data.rows.length).toBe(strategies.data.data.length)
  })

  it('has a NON-empty triage list, so the attention slot is exercised', () => {
    const triage = resolveFixture('/api/triage') as TriagePayload
    expect(triage.items.length).toBeGreaterThan(0)
    expect(triage.sources).toEqual({ fleet: true, ops: true, book: true })
  })

  it('keeps book slices consistent with capacity', () => {
    const book = resolveFixture('/api/book') as ApiEnvelope<BookPayload>
    expect(book.data.slices.length).toBe(book.data.allocated)
    expect(book.data.allocated).toBeLessThanOrEqual(book.data.capacity)
  })
})
