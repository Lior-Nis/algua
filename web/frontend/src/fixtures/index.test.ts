import { describe, expect, it } from 'vitest'
import { FIXTURE_SENTINEL, resolveFixture } from './index'
import type { ApiEnvelope, BookPayload, FleetHealth, TriagePayload } from '../types'

describe('resolveFixture', () => {
  it('returns undefined for an unknown URL rather than a fabricated payload', () => {
    expect(resolveFixture('/api/nope')).toBeUndefined()
  })

  it('serves every endpoint the app actually calls', () => {
    // Enumerated from the useFetch call sites; a new endpoint must be added here
    // deliberately, not discovered as a blank screen in the demo build.
    for (const url of [
      '/api/triage',
      '/api/fleet',
      '/api/book',
      '/api/ops',
      '/api/ideas',
      '/api/strategies',
      '/api/runs',
      '/api/runs?kind=gate&sort=sharpe_oos&limit=20',
      '/api/strategy/liquid10_adj_momentum',
    ]) {
      expect(resolveFixture(url), url).toBeDefined()
    }
  })

  it('carries the sentinel so a production build can be proven fixture-free', () => {
    expect(JSON.stringify(resolveFixture('/api/fleet'))).toContain(FIXTURE_SENTINEL)
  })

  it('is rich enough to exercise the design: >= 10 fleet rows across >= 3 stages', () => {
    const fleet = resolveFixture('/api/fleet') as ApiEnvelope<FleetHealth>
    expect(fleet.data.rows.length).toBeGreaterThanOrEqual(10)
    expect(new Set(fleet.data.rows.map((r) => r.stage)).size).toBeGreaterThanOrEqual(3)
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
