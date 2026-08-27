/** Spec §5.2 — verbosity does not return in one commit; it returns one helpful sentence at a
 * time. This is the ratchet that resists that, in the same spirit as the module-size ratchet
 * (tests/test_module_size_ratchet.py) and the import-linter contracts.
 *
 * RAISING A CEILING IS A DELIBERATE, VISIBLE ACT. If a screen exceeds its budget, the default
 * answer is to cut words or replace them with a mark — not to edit the number.
 *
 * A floor sits alongside every ceiling: a budget test that "passes" because a screen rendered
 * nothing (a blind fetch mock, an unfixtured URL, a crashed render swallowed by an error
 * boundary) is worthless. The floor forces an investigation instead of a silent green.
 *
 * Cross-test cache leakage: `useFetch` (src/api.ts) keeps a MODULE-LEVEL cache + in-flight map,
 * shared across every hook instance for the lifetime of this file's module graph (Vitest
 * isolates module state PER TEST FILE, not per test-within-a-file). If two screens under test
 * fetched the SAME url, the second screen could get served from cache instead of from this
 * file's fetch mock, which would still be correct data here (the fixture is deterministic) but
 * would skip the loading pass and could silently mask a future case where it isn't. The four
 * screens below are verified to hit disjoint URLs — Now: /api/triage, /api/activity; Money:
 * /api/book; Research: /api/strategies, /api/ops, /api/ideas; Fleet: /api/fleet — so the shared
 * cache cannot leak fixture data between them. `cleanup()` after each test additionally tears
 * down the previous screen's effects (visibilitychange listeners, in-flight timers) so a
 * lingering mount can't fire a stray fetch against the next test's fresh mock. */
import { cleanup, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { resolveFixture } from '../fixtures'
import Fleet from '../screens/Fleet'
import Money from '../screens/Money'
import Now from '../screens/Now'
import Research from '../screens/Research'

/** Ceilings are the measured word count against the steady-state fixture + 10%, rounded up
 * (measured: Now 25, Money 21, Research 93, Fleet 11 — see task-3-report.md). This slice's
 * screens already carry the redesign's leaner markup, so these are the current, real numbers —
 * not padding for an old verbose layout. The ratchet exists so THIS is the last time a screen's
 * word count grows without a deliberate, reviewed bump to the number below.
 *
 * Floors are ~half the measured count, rounded to a whole word: comfortably below normal
 * wording variance, but high enough that a screen rendering nothing (an empty state standing in
 * for a real one, an unfixtured URL silently swallowed, a crashed render) fails loudly instead
 * of reading as a passing, empty budget. Fleet leaves this slice in slice 2; its budget entry
 * goes with it. */
export const BUDGETS: Record<string, { min: number; max: number }> = {
  Now: { min: 15, max: 28 },
  Money: { min: 12, max: 24 },
  Research: { min: 50, max: 103 },
  Fleet: { min: 6, max: 13 },
}

beforeEach(() => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = typeof input === 'string' ? input : String(input)
    const payload = resolveFixture(url)
    if (payload === undefined) return new Response('{}', { status: 404 })
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
  })
})

afterEach(() => {
  cleanup()
})

function words(el: HTMLElement): number {
  const text = el.textContent ?? ''
  return text.trim() ? text.trim().split(/\s+/).length : 0
}

describe.each([
  ['Now', Now],
  ['Money', Money],
  ['Research', Research],
  ['Fleet', Fleet],
])('%s stays inside its word budget', (name, Screen) => {
  it(`renders <= ${BUDGETS[name].max} words (and >= ${BUDGETS[name].min}) against the fixture`, async () => {
    const { container } = render(
      <MemoryRouter>
        <Screen />
      </MemoryRouter>,
    )
    await waitFor(() => expect(container.textContent).not.toBe(''))
    const count = words(container as HTMLElement)
    const { min, max } = BUDGETS[name]
    expect(count, `${name} rendered ${count} words (floor ${min}) — did it render at all?`).toBeGreaterThanOrEqual(
      min,
    )
    expect(count, `${name} rendered ${count} words (budget ${max})`).toBeLessThanOrEqual(max)
  })
})
