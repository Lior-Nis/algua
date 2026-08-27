/** Spec §5.2 — verbosity does not return in one commit; it returns one helpful sentence at a
 * time. This is the ratchet that resists that, in the same spirit as the module-size ratchet
 * (tests/test_module_size_ratchet.py) and the import-linter contracts.
 *
 * RAISING A CEILING IS A DELIBERATE, VISIBLE ACT. If a screen exceeds its budget, the default
 * answer is to cut words or replace them with a mark — not to edit the number.
 *
 * A floor sits alongside every ceiling, but the floor protects against exactly ONE failure mode:
 * TOTAL RENDER FAILURE — an unfixtured URL swallowed into a silent empty state, a crashed render,
 * a screen that never got past its loading skeleton. It is NOT a data-presence check: word count
 * cannot tell a populated table from an empty one (see the review finding below), so a separate
 * assertion does that job instead (the "actually renders fixture data" block).
 *
 * WORD-COUNT METRIC — jsdom's `textContent` inserts NO whitespace between sibling elements, so
 * naively splitting `container.textContent` on `/\s+/` silently merges an entire multi-row
 * `<table>` into one unbroken token (e.g. Money's 10-row book table contributed ~0 words to its
 * original 21-word count — LOWER than the ~25 words its own EMPTY state produced). `words()`
 * below walks every individual text node with a `TreeWalker` and joins them with an explicit
 * space, so an element boundary always counts as a word boundary, matching how a person actually
 * reads the screen. Do not swap this for `.textContent.split(/\s+/)` or `.innerText` (jsdom does
 * not implement `innerText` layout-faithfully) — both regress this exact bug.
 *
 * The "populated beats emptied" describe block below pins the property the old metric violated:
 * real fixture data must count as MORE words than the same screen with its main collection
 * emptied, never fewer. That class of bug now fails a test instead of quietly passing one.
 *
 * CROSS-TEST CACHE LEAKAGE — `useFetch` (src/api.ts) keeps a MODULE-LEVEL cache + in-flight map,
 * shared across every hook instance for the lifetime of THIS FILE's module graph (Vitest
 * isolates module state per test FILE, not per test within a file), with TTLs (10s/30s per
 * screen) that dwarf a test's real runtime. Two hazards follow, and both are handled the same
 * way: (1) the "populated vs emptied" tests deliberately mount the SAME screen (same cache key)
 * twice in one test with two different mocked responses; (2) the ratchet/presence tests always
 * expect the DEFAULT populated fixture, but a screen mounted earlier in file order could have
 * left an EMPTIED response cached under that URL. `bustFetchCache()` advances the fake system
 * clock (`vi.setSystemTime`) well past every screen's TTL before every render, so `useFetch`'s
 * own `Date.now() - hit.at >= ttlMs` staleness check always trips and it revalidates against
 * whichever mock is active for THIS render, instead of serving a previous render's cached value.
 * This was verified empirically, not assumed: the same test written without the clock jump reads
 * back the FIRST render's data on the second render, confirming the leak this guards against is
 * real (see task-3-report.md). `vi.useRealTimers()` after each test keeps the fake clock from
 * bleeding into unrelated assertions. */
import { cleanup, render, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { resolveFixture } from '../fixtures'
import { ACTIVITY, BOOK, FLEET, RUNS, TRIAGE } from '../fixtures/steady-state'
import Fleet from '../screens/Fleet'
import Money from '../screens/Money'
import Now from '../screens/Now'
import Research from '../screens/Research'
import StrategyDetail from '../screens/StrategyDetail'

/** Ceilings are the measured word count against the steady-state fixture + 10%, rounded up. This
 * slice's screens already carry the redesign's leaner markup, so these are the current, real
 * numbers, not padding for an old verbose layout. The ratchet exists so THIS is the last time a
 * screen's word count grows without a deliberate, reviewed bump to the number below.
 *
 * Floors guard ONLY against total render failure (see file header) — picked at roughly HALF the
 * measured count, comfortably below ordinary wording variance but far above what a
 * blank/near-blank/error-only render would produce. They are NOT a data-presence check; that is
 * the job of the "actually renders fixture data" block.
 *
 * `marks` is the OTHER half of spec §5.2 (fix wave 2026-08-27, FIX 7): a per-screen MINIMUM count
 * of `<svg>`/`<canvas>` elements (the logo excluded — see `marks()` below — so it can never
 * satisfy this floor by itself). Without it, a screen with zero visualizations passes the word
 * ceiling forever, which was true of Now/Money/Fleet before this fix (Research already carries
 * 12 via ScatterISOOS + RunList's sparklines). Each value below is pinned at TODAY's measured
 * count (0 where a screen genuinely has no chart yet) purely to stop a regression; slices 2 and 3
 * are expected to RAISE these numbers as marks land, the same deliberate, reviewed bump the word
 * ceiling already requires.
 *
 * NOW'S CEILING moved in THIS fix wave (57, was 61) because FIX 3/4 reshaped the triage fixture
 * to match what the real endpoint can actually emit (no `since` line, shorter `detail` strings) —
 * a legitimate content change, not slack. It will move again, and MUST be re-derived rather than
 * left high, the moment slice 2 deletes Now's activity feed: a ceiling that stays high after a
 * screen gets leaner stops enforcing anything. */
export const BUDGETS: Record<string, { min: number; max: number; marks: number }> = {
  Now: { min: 29, max: 63, marks: 0 },
  Money: { min: 43, max: 94, marks: 0 },
  Research: { min: 94, max: 206, marks: 12 },
  Fleet: { min: 38, max: 84, marks: 0 },
  // The deepest prose surface (spec §4.3 says to trim it) — and, until this fix, the one route
  // `verify-viewport.mjs` covered that this file did not. Rendered standalone below via its own
  // MemoryRouter+Route (it reads `useParams()`, unlike the other four screens). Measured 204
  // words / 4 marks (EquityChart's canvas, GateBulletCard's and TrialDistribution's svgs).
  StrategyDetail: { min: 102, max: 225, marks: 4 },
}

function mockFetch(overrides: Record<string, unknown> = {}) {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = typeof input === 'string' ? input : String(input)
    const path = url.split('?')[0]
    if (path in overrides) {
      return new Response(JSON.stringify(overrides[path]), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }
    const payload = resolveFixture(url)
    if (payload === undefined) return new Response('{}', { status: 404 })
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
  })
}

/** See file header — forces every `useFetch` mount past its TTL so it revalidates against
 * whichever mock is currently installed instead of serving a previous render's cache entry. */
function bustFetchCache() {
  vi.setSystemTime(Date.now() + 120_000)
}

beforeEach(() => {
  bustFetchCache()
  mockFetch()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

/** Walks EVERY text node under `el` and joins them with an explicit space before splitting —
 * unlike `el.textContent.split(/\s+/)`, this counts an element boundary (e.g. between two
 * `<td>`s) as a word boundary, so a table's cells don't collapse into one unbroken token. */
function words(el: HTMLElement): number {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
  const parts: string[] = []
  let node: Node | null
  while ((node = walker.nextNode())) {
    const text = (node.nodeValue ?? '').trim()
    if (text) parts.push(text)
  }
  const joined = parts.join(' ')
  return joined ? joined.split(/\s+/).length : 0
}

/** Counts `<svg>`/`<canvas>` elements — the "does this screen actually visualize anything" half
 * of spec §5.2 that the word ceiling alone cannot answer (a screen can be verbose in PROSE and
 * still carry zero marks). The brand logo (`svg.brand-lockup`, see `components/Logo.tsx`) is
 * excluded so it can never satisfy a screen's floor by itself — none of these screens render it
 * when mounted standalone (it lives in App.tsx's header), but a future screen that does must
 * still be made to earn its own mark count. */
function marks(el: HTMLElement): number {
  let count = 0
  el.querySelectorAll('svg, canvas').forEach((node) => {
    if (node.tagName.toLowerCase() === 'svg' && node.classList.contains('brand-lockup')) return
    count++
  })
  return count
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
    expect(
      count,
      `${name} rendered ${count} words (floor ${min}) — did it render at all?`,
    ).toBeGreaterThanOrEqual(min)
    expect(count, `${name} rendered ${count} words (budget ${max})`).toBeLessThanOrEqual(max)
  })

  it(`carries >= ${BUDGETS[name].marks} mark(s) (svg/canvas, excluding the logo)`, async () => {
    const { container } = render(
      <MemoryRouter>
        <Screen />
      </MemoryRouter>,
    )
    await waitFor(() => expect(container.textContent).not.toBe(''))
    const count = marks(container as HTMLElement)
    expect(count).toBeGreaterThanOrEqual(BUDGETS[name].marks)
  })
})

describe('StrategyDetail stays inside its word budget', () => {
  const { min, max, marks: minMarks } = BUDGETS.StrategyDetail

  function renderDetail() {
    return render(
      <MemoryRouter initialEntries={['/s/liquid10_adj_momentum']}>
        <Routes>
          <Route path="/s/:name" element={<StrategyDetail />} />
        </Routes>
      </MemoryRouter>,
    )
  }

  it(`renders <= ${max} words (and >= ${min}) against the fixture`, async () => {
    const { container } = renderDetail()
    await waitFor(() => expect(container.textContent).not.toBe(''))
    const count = words(container as HTMLElement)
    expect(
      count,
      `StrategyDetail rendered ${count} words (floor ${min}) — did it render at all?`,
    ).toBeGreaterThanOrEqual(min)
    expect(count, `StrategyDetail rendered ${count} words (budget ${max})`).toBeLessThanOrEqual(
      max,
    )
  })

  it(`carries >= ${minMarks} mark(s) (svg/canvas, excluding the logo)`, async () => {
    const { container } = renderDetail()
    await waitFor(() => expect(container.textContent).not.toBe(''))
    const count = marks(container as HTMLElement)
    expect(count).toBeGreaterThanOrEqual(minMarks)
  })
})

/** Word count alone cannot distinguish a populated table from an empty one (that is the whole
 * point of the review finding this file was corrected for) — so these assert directly that a
 * SPECIFIC fixture value made it into the DOM, catching a silently-empty table/list that a count
 * inside budget would happily let through. */
describe('actually renders fixture data', () => {
  it('Now shows a real triage item title', async () => {
    const { container } = render(
      <MemoryRouter>
        <Now />
      </MemoryRouter>,
    )
    await waitFor(() => expect(container.textContent).toContain(TRIAGE.items[0].title))
  })

  it('Money shows a real book-slice strategy name', async () => {
    const { container } = render(
      <MemoryRouter>
        <Money />
      </MemoryRouter>,
    )
    await waitFor(() => expect(container.textContent).toContain(BOOK.data.slices[0].strategy))
  })

  it('Research shows a real ranked-run strategy name', async () => {
    const { container } = render(
      <MemoryRouter>
        <Research />
      </MemoryRouter>,
    )
    await waitFor(() => expect(container.textContent).toContain(RUNS.data.runs[0].strategy_name))
  })

  it('Fleet shows a real strategy name from the full row list', async () => {
    const { container } = render(
      <MemoryRouter>
        <Fleet />
      </MemoryRouter>,
    )
    // Picked from `rows`, not `alerting`, so this exercises the "every strategy" list
    // specifically, not just the (separately-rendered) alert section.
    const okRow = FLEET.data.rows.find((r) => r.health === 'ok')
    if (okRow === undefined) throw new Error('fixture has no ok-health row to assert against')
    await waitFor(() => expect(container.textContent).toContain(okRow.strategy))
  })
})

const EMPTY_BOOK = {
  ...BOOK,
  data: {
    ...BOOK.data,
    slices: [],
    allocated: 0,
    sum_allocations: 0,
    count_headroom: BOOK.data.capacity,
    live_allocated: 0,
    unallocated_operational: [],
  },
}

const EMPTY_FLEET = {
  ...FLEET,
  data: {
    ...FLEET.data,
    rows: [],
    alerting: [],
    summary: { total: 0, alerting: 0, by_health: {} },
  },
}

const EMPTY_RUNS = {
  ...RUNS,
  data: { ...RUNS.data, runs: [], count: 0 },
}

const EMPTY_TRIAGE = { ...TRIAGE, items: [] }
const EMPTY_ACTIVITY = { ...ACTIVITY, data: { ...ACTIVITY.data, data: [] } }

/** The inversion this pins: Money's real 10-row book table used to count ~0 words (concatenated
 * by jsdom into one token) while ITS OWN empty state ("book empty · no capital is allocated")
 * counted MORE — a broken table would have silently passed the budget a healthy one couldn't
 * beat. This block makes that class of bug fail a test rather than pass one, for every screen
 * with a dominant fixture-driven collection. */
describe.each([
  { name: 'Now', Screen: Now, overrides: () => ({ '/api/triage': EMPTY_TRIAGE, '/api/activity': EMPTY_ACTIVITY }) },
  { name: 'Money', Screen: Money, overrides: () => ({ '/api/book': EMPTY_BOOK }) },
  { name: 'Research', Screen: Research, overrides: () => ({ '/api/runs': EMPTY_RUNS }) },
  { name: 'Fleet', Screen: Fleet, overrides: () => ({ '/api/fleet': EMPTY_FLEET }) },
])('$name: populated fixture beats an emptied one', ({ name, Screen, overrides }) => {
  it('renders strictly more words with real data than with its collection emptied', async () => {
    const populated = render(
      <MemoryRouter>
        <Screen />
      </MemoryRouter>,
    )
    await waitFor(() => expect(populated.container.textContent).not.toBe(''))
    const populatedCount = words(populated.container as HTMLElement)
    populated.unmount()

    bustFetchCache()
    mockFetch(overrides())
    const emptied = render(
      <MemoryRouter>
        <Screen />
      </MemoryRouter>,
    )
    await waitFor(() => expect(emptied.container.textContent).not.toBe(''))
    const emptiedCount = words(emptied.container as HTMLElement)
    emptied.unmount()

    expect(
      populatedCount,
      `${name}: populated=${populatedCount} words, emptied=${emptiedCount} words — ` +
        'the metric must be able to tell these apart',
    ).toBeGreaterThan(emptiedCount)
  })
})
