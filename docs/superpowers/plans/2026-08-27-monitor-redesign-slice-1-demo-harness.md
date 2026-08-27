# Monitor Redesign — Slice 1: The Demo Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static, fixture-backed build of the real monitor that renders every screen with rich data and no backend — plus the two CI invariants (no horizontal overflow, per-screen word budget) that every later slice is judged against.

**Architecture:** `getJSON` in `src/api.ts` is the single network chokepoint — every screen reaches the network through `useFetch` → `fetchShared` → `getJSON`. Demo mode swaps that one function for a fixture lookup behind a statically-known flag, so the same components render in both modes and no screen learns it is in a demo. Fixtures are reached by dynamic `import()` inside the dead branch, so a production build cannot carry them; a test over `dist/` proves it rather than trusting tree-shaking.

**Tech Stack:** React 19, TypeScript 5.9, Vite 8, Vitest 4 (jsdom), Playwright (new devDependency, `channel: 'chrome'` — no browser download).

**Spec:** `docs/superpowers/specs/2026-08-27-monitor-human-facing-redesign-design.md`

## Global Constraints

- `web/` is a **STANDALONE** uv project. NEVER add web dependencies to the root project — the root `uv.lock` is `dependency_hash` identity.
- Frontend gates: `cd web/frontend && npm run check && npm run build`.
- Backend gate: `uv run --project web pytest web/backend/tests -q`.
- Root gates: `uv run pytest -q` as its OWN isolated command; then `uv run ruff check .`, `uv run mypy algua scripts`, `uv run lint-imports`.
- **Never point anything at `data/algua.db`.** Use `ALGUA_DB_PATH=/tmp/...`.
- `git add` scoped to named paths — never `git add -A` (concurrent sessions share this repo).
- Design tokens come from `src/theme.css`. Electric (`--electric` / `--series-focus`, `#3982ff`) marks **data**, never chrome. Status colours (`--green`/`--red`/`--amber`/`--violet`) are reserved for status and never used as series colours.
- Mobile-first: **no `@media` rules, no `:hover` affordances.** The target viewport is 390px.
- This slice changes **no** screen's markup. Screens are redesigned in slices 2 and 3.

---

### Task 1: The steady-state fixture

**Files:**
- Create: `web/frontend/src/fixtures/steady-state.ts`
- Create: `web/frontend/src/fixtures/index.ts`
- Test: `web/frontend/src/fixtures/index.test.ts`

**Interfaces:**
- Consumes: the payload types in `src/types.ts` (`ApiEnvelope`, `TriagePayload`, `FleetHealth`, `BookPayload`, `OpsPayload`, `ListPayload`, `StrategyRecord`, `IdeasResponse`, `RunsListPayload`).
- Produces:
  - `resolveFixture(url: string): unknown | undefined` — the URL→payload map. Returns `undefined` for an unknown URL.
  - `FIXTURE_SENTINEL: string` — a unique string embedded in the fixture data, used by Task 2's build guard.

- [ ] **Step 1: Write the failing test**

```ts
// web/frontend/src/fixtures/index.test.ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web/frontend && npx vitest run src/fixtures/index.test.ts`
Expected: FAIL — `Failed to resolve import "./index"`.

- [ ] **Step 3: Write the fixture data**

Create `web/frontend/src/fixtures/steady-state.ts`. This is the ONE rich steady state from spec §6: strategies spread across stages, a populated funnel, real history depth, and a couple of live exceptions so the attention slot has something to show.

```ts
/** The single rich steady-state fixture (spec §6). Shaped EXACTLY like the API envelopes so
 * the demo transport is a swap, not an adapter with logic in it — a fixture that needed
 * massaging on the way out would be testing the adapter, not the screens.
 *
 * Every value here is plausible-but-invented. Nothing in this file is read in a production
 * build (see `src/transport.ts` and the dist guard in its test). */
import type {
  ApiEnvelope, BookPayload, FleetHealth, FleetRow, IdeasResponse, ListPayload,
  OpsPayload, RunsListPayload, StrategyRecord, TriagePayload,
} from '../types'

/** Embedded in the fixture data so a production build can be PROVEN fixture-free by grepping
 * `dist/` for it. Deliberately a string no real payload would ever contain. */
export const FIXTURE_SENTINEL = 'ALGUA_DEMO_FIXTURE_a7f3e1'

const FETCHED_AT = '2026-08-27T09:15:00+00:00'

function envelope<T>(data: T): ApiEnvelope<T> {
  return { ok: true, data, fetched_at: FETCHED_AT, stale: false }
}

interface Seed {
  name: string
  stage: string
  health: FleetRow['health']
  drawdown: number
  capital: number
  pnl: number
}

/** 14 strategies across 4 operational stages plus research stages. Two are unhealthy, which
 * is what gives the attention slot and the fleet grid something to disagree about. */
const SEEDS: Seed[] = [
  { name: 'liquid10_adj_momentum',        stage: 'live',           health: 'ok',    drawdown: 0.021, capital: 1800, pnl:  74.2 },
  { name: 'orderly_six_day_rebound',      stage: 'live',           health: 'stale', drawdown: 0.038, capital: 1800, pnl: -22.6 },
  { name: 'low_vol_skip_momentum_top3',   stage: 'live',           health: 'ok',    drawdown: 0.014, capital: 1800, pnl:  51.9 },
  { name: 'lagged_rank_persistence',      stage: 'paper',          health: 'ok',    drawdown: 0.045, capital: 1550, pnl:  18.4 },
  { name: 'cross_horizon_low_vol',        stage: 'paper',          health: 'drift', drawdown: 0.084, capital: 1550, pnl: -41.0 },
  { name: 'dual_horizon_skip_month',      stage: 'paper',          health: 'ok',    drawdown: 0.019, capital: 1550, pnl:  33.7 },
  { name: 'quality_momentum_qs',          stage: 'paper',          health: 'ok',    drawdown: 0.052, capital: 1550, pnl:   9.1 },
  { name: 'peer_selloff_rebound',         stage: 'paper',          health: 'ok',    drawdown: 0.028, capital: 1550, pnl:  27.3 },
  { name: 'cadenced_tail_risk',           stage: 'forward_tested', health: 'ok',    drawdown: 0.011, capital: 1200, pnl:  12.8 },
  { name: 'distributed_gains_qm',         stage: 'forward_tested', health: 'ok',    drawdown: 0.033, capital: 1200, pnl:  20.5 },
  { name: 'skip_month_persistence',       stage: 'dormant',        health: 'idle',  drawdown: 0.0,   capital: 0,    pnl:   0.0 },
  { name: 'vol_carry_neutral',            stage: 'candidate',      health: 'idle',  drawdown: 0.0,   capital: 0,    pnl:   0.0 },
  { name: 'range_compression_break',      stage: 'backtested',     health: 'idle',  drawdown: 0.0,   capital: 0,    pnl:   0.0 },
  { name: 'seasonal_turn_of_month',       stage: 'idea',           health: 'idle',  drawdown: 0.0,   capital: 0,    pnl:   0.0 },
]

const OPERATIONAL = new Set(['live', 'paper', 'forward_tested', 'dormant'])
const operational = SEEDS.filter((s) => OPERATIONAL.has(s.stage))
const allocated = SEEDS.filter((s) => s.capital > 0)

function fleetRow(s: Seed): FleetRow {
  return {
    strategy: s.name,
    stage: s.stage,
    health: s.health,
    staleness_sessions: s.health === 'stale' ? 3 : s.health === 'drift' ? 2 : 0,
    last_tick_error: null,
    kill_switch: { tripped: false, reason: null, global_halt: false },
    drawdown: {
      peak_equity: s.capital > 0 ? s.capital : null,
      last_equity: s.capital > 0 ? s.capital * (1 - s.drawdown) : null,
      drawdown: s.capital > 0 ? s.drawdown : null,
    },
    positions: s.capital > 0 ? 3 : 0,
    n_orders: s.capital > 0 ? 2 : 0,
  }
}

const rows = operational.map(fleetRow)
const alerting = rows.filter((r) => r.health !== 'ok' && r.health !== 'idle')

export const FLEET: ApiEnvelope<FleetHealth> = envelope({
  ok: alerting.length === 0,
  global_halt: false,
  alerting,
  summary: {
    total: rows.length,
    alerting: alerting.length,
    by_health: alerting.reduce<Record<string, number>>((acc, r) => {
      acc[r.health] = (acc[r.health] ?? 0) + 1
      return acc
    }, {}),
  },
  stale_after_sessions: 2,
  operational_stages: ['live', 'paper', 'forward_tested', 'dormant'],
  rows,
})

export const TRIAGE: TriagePayload = {
  ok: true,
  items: [
    {
      kind: 'strategy',
      severity: 2,
      title: 'orderly_six_day_rebound',
      detail: `marks stale · 3 sessions · ${FIXTURE_SENTINEL}`,
      since: '2026-08-24T13:30:00+00:00',
      route: '/s/orderly_six_day_rebound',
    },
    {
      kind: 'strategy',
      severity: 1,
      title: 'cross_horizon_low_vol',
      detail: 'drawdown 8.4% of 10% wall',
      since: '2026-08-26T18:05:00+00:00',
      route: '/s/cross_horizon_low_vol',
    },
  ],
  // All three sources loaded — a degraded read must NEVER render as an all-clear
  // (see TriagePayload's docstring in types.ts).
  sources: { fleet: true, ops: true, book: true },
  headline: {
    fleet_ok: rows.length - alerting.length,
    fleet_total: rows.length,
    book_allocated: allocated.length,
    book_capacity: 64,
    loops_alerting: 0,
  },
  fetched_at: FETCHED_AT,
  stale: false,
}

export const BOOK: ApiEnvelope<BookPayload> = envelope({
  ok: true,
  capacity: 64,
  allocated: allocated.length,
  count_headroom: 64 - allocated.length,
  sum_allocations: allocated.reduce((t, s) => t + s.capital, 0),
  unallocated_operational: [],
  slices: allocated.map((s) => ({
    strategy: s.name,
    stage: s.stage,
    capital: s.capital,
    last_equity: s.capital + s.pnl,
    effective_ts: FETCHED_AT,
    actor: 'agent',
  })),
  live_allocated: allocated.filter((s) => s.stage === 'live').length,
})

export const OPS: ApiEnvelope<OpsPayload> = envelope({
  ok: true,
  checked_at: FETCHED_AT,
  alerting: [],
  loops: {
    paper: { name: 'paper', state: 'ok', last_run: FETCHED_AT, consecutive_failures: 0 },
    research: { name: 'research', state: 'ok', last_run: FETCHED_AT, consecutive_failures: 0 },
  } as OpsPayload['loops'],
})

export const STRATEGIES: ApiEnvelope<ListPayload<StrategyRecord>> = envelope({
  ok: true,
  count: SEEDS.length,
  items: SEEDS.map((s) => ({
    name: s.name,
    stage: s.stage,
    created_at: '2026-06-01T00:00:00+00:00',
    updated_at: FETCHED_AT,
  })) as StrategyRecord[],
})

export const IDEAS: IdeasResponse = {
  ok: true,
  data: { ok: true, count: 0, items: [] },
  fetched_at: FETCHED_AT,
  stale: false,
} as IdeasResponse

/** Deterministic pseudo-random so the fixture is byte-stable across builds (a fixture that
 * changed every build would make the dist guard and the word budget flaky). */
function rng(seed: number): () => number {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296
    return s / 4294967296
  }
}

const runs = (() => {
  const r = rng(20260827)
  return operational.map((s, i) => {
    const meanWindow = 0.35 + r() * 1.3
    const oos = meanWindow * (0.45 + r() * 0.7)
    return {
      id: 100 + i,
      kind: 'gate',
      strategy_name: s.name,
      created_at: FETCHED_AT,
      passed: oos > 0.3,
      mean_window_sharpe: Number(meanWindow.toFixed(4)),
      min_window_sharpe: Number((meanWindow - 0.4 - r() * 0.5).toFixed(4)),
      sharpe_oos: Number(oos.toFixed(4)),
      sortino_oos: Number((oos * 1.3).toFixed(4)),
    }
  })
})()

export const RUNS: ApiEnvelope<RunsListPayload> = envelope({
  ok: true,
  count: runs.length,
  runs,
} as unknown as RunsListPayload)
```

- [ ] **Step 4: Write the resolver**

Create `web/frontend/src/fixtures/index.ts`:

```ts
/** URL -> fixture payload. Deliberately a MAP with an explicit unknown-URL miss rather than a
 * catch-all: a screen that calls an endpoint nobody fixtured must fail loudly in the demo
 * build (see `transport.ts`), not render a plausible-looking empty state that hides the gap. */
import {
  BOOK, FIXTURE_SENTINEL, FLEET, IDEAS, OPS, RUNS, STRATEGIES, TRIAGE,
} from './steady-state'
import type { StrategyDetailResponse } from '../types'

export { FIXTURE_SENTINEL }

function strategyDetail(name: string): StrategyDetailResponse {
  const row = FLEET.data.rows.find((r) => r.strategy === name) ?? FLEET.data.rows[0]
  return {
    ok: true,
    data: {
      strategy: row.strategy,
      registry: { name: row.strategy, stage: row.stage, family: 'trend-following' },
      fleet: row,
      transitions: [],
      recent_orders: [],
    },
    fetched_at: FLEET.fetched_at,
    stale: false,
  } as unknown as StrategyDetailResponse
}

export function resolveFixture(url: string): unknown | undefined {
  const path = url.split('?')[0]

  switch (path) {
    case '/api/triage':      return TRIAGE
    case '/api/fleet':       return FLEET
    case '/api/book':        return BOOK
    case '/api/ops':         return OPS
    case '/api/ideas':       return IDEAS
    case '/api/strategies':  return STRATEGIES
    case '/api/runs':        return RUNS
    // The run-ledger series endpoint deliberately never returns a per-bar OOS vector
    // (holdout_returns.returns_blob is SENSITIVE) — an empty list is the honest fixture.
    case '/api/runs/series': return { ok: true, data: { ok: true, entries: [] }, fetched_at: FLEET.fetched_at, stale: false }
    case '/api/activity':    return { ok: true, data: { ok: true, count: 0, items: [] }, fetched_at: FLEET.fetched_at, stale: false }
    case '/api/push/key':    return { ok: true, data: { key: null }, fetched_at: FLEET.fetched_at, stale: false }
  }

  const strategy = path.match(/^\/api\/strategy\/([^/]+)$/)
  if (strategy) return strategyDetail(decodeURIComponent(strategy[1]))
  if (/^\/api\/strategy\/[^/]+\/series$/.test(path)) {
    return { ok: true, data: { ok: true, lane: 'paper', rows: [] }, fetched_at: FLEET.fetched_at, stale: false }
  }
  if (/^\/api\/runs\/\d+$/.test(path)) return { ok: true, data: RUNS.data.runs[0], fetched_at: FLEET.fetched_at, stale: false }

  return undefined
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd web/frontend && npx vitest run src/fixtures/index.test.ts`
Expected: PASS (6 tests). If a payload type mismatches, fix the FIXTURE to match `types.ts` — never widen a type in `types.ts` to accept the fixture.

- [ ] **Step 6: Typecheck**

Run: `cd web/frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add web/frontend/src/fixtures/
git commit -m "feat(web): steady-state fixture for the demo harness"
```

---

### Task 2: Demo transport, the build, and both guards

**Files:**
- Create: `web/frontend/src/transport.ts`
- Modify: `web/frontend/src/api.ts` (the `getJSON` body)
- Modify: `web/frontend/package.json` (add `build:demo`, `verify:demo-build`)
- Test: `web/frontend/src/transport.test.ts`
- Test: `web/frontend/scripts/verify-demo-build.mjs`

**Interfaces:**
- Consumes: `resolveFixture`, `FIXTURE_SENTINEL` (Task 1).
- Produces: `DEMO: boolean` and `demoJSON<T>(url: string): Promise<T>` from `src/transport.ts`.

- [ ] **Step 1: Write the failing test**

```ts
// web/frontend/src/transport.test.ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { demoJSON } from './transport'

afterEach(() => vi.restoreAllMocks())

describe('demoJSON', () => {
  it('serves a fixture WITHOUT touching the network', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const payload = await demoJSON<{ ok: boolean }>('/api/fleet')
    expect(payload.ok).toBe(true)
    // Spec §6 guard 2: the demo build must never reach the network.
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('THROWS on an unfixtured URL rather than inventing an empty payload', async () => {
    // A silent empty payload would look exactly like a real empty state, which is the
    // failure mode this whole redesign exists to remove.
    await expect(demoJSON('/api/not-fixtured')).rejects.toThrow(/no fixture/i)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web/frontend && npx vitest run src/transport.test.ts`
Expected: FAIL — `Failed to resolve import "./transport"`.

- [ ] **Step 3: Write the transport**

```ts
// web/frontend/src/transport.ts
/** Demo transport (spec §6). `DEMO` is a build-time constant: Vite statically replaces
 * `import.meta.env.VITE_ALGUA_DEMO`, so in a production build this folds to `false`, the
 * branch in `getJSON` is dead, and the dynamic `import('./fixtures')` below is never reached.
 * The dynamic form matters — it keeps the fixture module out of the main graph entirely
 * rather than relying on tree-shaking to remove a static import.
 *
 * `verify-demo-build.mjs` PROVES the production bundle is fixture-free; do not downgrade
 * that check to a comment. */
export const DEMO: boolean = import.meta.env.VITE_ALGUA_DEMO === '1'

export async function demoJSON<T>(url: string): Promise<T> {
  const { resolveFixture } = await import('./fixtures')
  const payload = resolveFixture(url)
  if (payload === undefined) {
    throw new Error(
      `demo build: no fixture for ${url} — add it to src/fixtures/index.ts. ` +
        'Failing loudly is deliberate: a fabricated empty payload is indistinguishable ' +
        'from a real empty state.',
    )
  }
  return payload as T
}
```

- [ ] **Step 4: Wire it into the one chokepoint**

In `web/frontend/src/api.ts`, add the import and the guard as the FIRST statement of `getJSON`:

```ts
import { DEMO, demoJSON } from './transport'

export async function getJSON<T>(url: string): Promise<T> {
  // Demo mode swaps the TRANSPORT, not the components: every screen renders identically in
  // both modes because neither knows which one it is in.
  if (DEMO) return demoJSON<T>(url)
  const res = await fetch(url, { headers: { accept: 'application/json' } })
  // ... rest unchanged
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd web/frontend && npx vitest run src/transport.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 6: Add the demo build script**

In `web/frontend/package.json`, add to `scripts`:

```json
"build:demo": "VITE_ALGUA_DEMO=1 tsc -b && VITE_ALGUA_DEMO=1 vite build --outDir dist-demo",
"verify:demo-build": "node scripts/verify-demo-build.mjs"
```

- [ ] **Step 7: Write the dist guard**

```js
// web/frontend/scripts/verify-demo-build.mjs
/** Spec §6 guard 1: the PRODUCTION build must never bundle fixtures.
 *
 * This greps the built output for the fixture sentinel rather than trusting that Rollup
 * tree-shook the dynamic import. If this ever fails, the fix is a build-time alias that
 * removes the module from the graph — NOT deleting this check. */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

const SENTINEL = 'ALGUA_DEMO_FIXTURE_a7f3e1'

function walk(dir) {
  return readdirSync(dir).flatMap((entry) => {
    const p = join(dir, entry)
    return statSync(p).isDirectory() ? walk(p) : [p]
  })
}

const [, , mode = 'prod', dir = 'dist'] = process.argv
const files = walk(dir)
const hits = files.filter((f) => readFileSync(f, 'utf8').includes(SENTINEL))

if (mode === 'prod' && hits.length > 0) {
  console.error(`FAIL: production build carries fixture data:\n  ${hits.join('\n  ')}`)
  process.exit(1)
}
if (mode === 'demo' && hits.length === 0) {
  console.error(`FAIL: demo build in ${dir} contains NO fixture data — it would render empty.`)
  process.exit(1)
}
console.log(`ok: ${mode} build in ${dir} (${files.length} files, ${hits.length} with fixtures)`)
```

- [ ] **Step 8: Run both builds and both guards**

```bash
cd web/frontend
npm run build && node scripts/verify-demo-build.mjs prod dist
npm run build:demo && node scripts/verify-demo-build.mjs demo dist-demo
```
Expected: both print `ok:`. If the prod check FAILS, switch `src/api.ts` to a Vite `resolve.alias` that maps `./transport` to a fixture-free stub in non-demo builds, and re-run — do not weaken the guard.

- [ ] **Step 9: Ignore the demo output**

Add `dist-demo/` to `web/frontend/.gitignore`.

- [ ] **Step 10: Commit**

```bash
git add web/frontend/src/transport.ts web/frontend/src/transport.test.ts \
        web/frontend/src/api.ts web/frontend/package.json \
        web/frontend/scripts/verify-demo-build.mjs web/frontend/.gitignore
git commit -m "feat(web): demo transport + fixture-free production build guard"
```

---

### Task 3: The word-budget invariant (spec §5.2)

**Files:**
- Create: `web/frontend/src/invariants/word-budget.test.tsx`

**Interfaces:**
- Consumes: `resolveFixture` (Task 1); the existing screens `Now`, `Money`, `Research`, `Fleet`.
- Produces: `BUDGETS: Record<string, number>` (exported from the test file so slices 2–3 tighten it in one visible place).

Word count needs rendered text, not layout, so this runs in jsdom under Vitest — no browser.

- [ ] **Step 1: Write the test**

```tsx
// web/frontend/src/invariants/word-budget.test.tsx
/** Spec §5.2 — verbosity does not return in one commit; it returns one helpful sentence at a
 * time. This is the ratchet that resists that, in the same spirit as the module-size ratchet
 * (tests/test_module_size_ratchet.py) and the import-linter contracts.
 *
 * RAISING A CEILING IS A DELIBERATE, VISIBLE ACT. If a screen exceeds its budget, the default
 * answer is to cut words or replace them with a mark — not to edit the number. */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { resolveFixture } from '../fixtures'
import Money from '../screens/Money'
import Now from '../screens/Now'
import Research from '../screens/Research'

/** Measured ceilings with modest headroom. Slice 1 records the CURRENT (bad) numbers so the
 * ratchet exists from day one; slices 2 and 3 lower them to the redesigned values. */
export const BUDGETS: Record<string, number> = {
  Now: 400,
  Money: 120,
  Research: 100,
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

function words(el: HTMLElement): number {
  const text = el.textContent ?? ''
  return text.trim() ? text.trim().split(/\s+/).length : 0
}

describe.each([
  ['Now', Now],
  ['Money', Money],
  ['Research', Research],
])('%s stays inside its word budget', (name, Screen) => {
  it(`renders <= ${'${BUDGETS[name]}'} words against the fixture`, async () => {
    const { container } = render(
      <MemoryRouter>
        <Screen />
      </MemoryRouter>,
    )
    await waitFor(() => expect(container.textContent).not.toBe(''))
    const count = words(container as HTMLElement)
    expect(count, `${name} rendered ${count} words (budget ${BUDGETS[name]})`).toBeLessThanOrEqual(
      BUDGETS[name],
    )
  })
})
```

- [ ] **Step 2: Run it and RECORD the real numbers**

Run: `cd web/frontend && npx vitest run src/invariants/word-budget.test.tsx`

Whatever the measured counts are, set each `BUDGETS` entry to `measured + 10%` rounded up, then re-run so the suite is green. Report the measured numbers in the commit message — they are the before-figures the redesign is judged against.

- [ ] **Step 3: Verify the ratchet actually bites**

Temporarily set `BUDGETS.Now = 1`, run the test, confirm it FAILS with the word count in the message, then restore. A ratchet that cannot fail is decoration.

- [ ] **Step 4: Commit**

```bash
git add web/frontend/src/invariants/word-budget.test.tsx
git commit -m "test(web): per-screen word budget ratchet"
```

---

### Task 4: The no-overflow invariant (spec §5.1) and CI

**Files:**
- Create: `web/frontend/scripts/verify-viewport.mjs`
- Modify: `web/frontend/package.json` (devDependency + script)
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the `dist-demo/` output (Task 2).
- Produces: `npm run verify:viewport` — exits non-zero if any route overflows 390px.

jsdom has no layout engine, so `scrollWidth` is meaningless there. This one needs a real browser.

- [ ] **Step 1: Add Playwright**

```bash
cd web/frontend && npm install --save-dev playwright@1.62.0
```

Use `channel: 'chrome'` in the script so CI uses the runner's installed Chrome and no browser download is needed.

- [ ] **Step 2: Write the check**

```js
// web/frontend/scripts/verify-viewport.mjs
/** Spec §5.1 — the /money bug was a 483px 5-column table sizing its scroll parent instead of
 * scrolling inside it, so the PAGE grew to 500px at a 390px viewport. This turns that class of
 * defect into a test: no route may overflow horizontally at 390px.
 *
 * Serves the DEMO build, so every route has rich data — an empty screen cannot overflow and
 * would make this check vacuous. */
import { createServer } from 'node:http'
import { readFileSync, existsSync } from 'node:fs'
import { extname, join, normalize } from 'node:path'
import { chromium } from 'playwright'

const ROUTES = ['/', '/money', '/research', '/s/liquid10_adj_momentum']
const DIR = 'dist-demo'
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
                '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
                '.woff2': 'font/woff2', '.woff': 'font/woff', '.webmanifest': 'application/manifest+json' }

const server = createServer((req, res) => {
  const url = req.url.split('?')[0]
  const candidate = join(DIR, normalize(url).replace(/^(\.\.[/\\])+/, ''))
  const file = existsSync(candidate) && extname(candidate) ? candidate : join(DIR, 'index.html')
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' })
  res.end(readFileSync(file))
})

await new Promise((r) => server.listen(0, '127.0.0.1', r))
const base = `http://127.0.0.1:${server.address().port}`

const browser = await chromium.launch({ channel: 'chrome' })
const page = await browser.newPage({ viewport: { width: 390, height: 844 } })
const failures = []

for (const route of ROUTES) {
  await page.goto(base + route, { waitUntil: 'networkidle' })
  await page.waitForTimeout(600)
  const m = await page.evaluate(() => {
    const vw = window.innerWidth
    const offenders = []
    document.querySelectorAll('*').forEach((el) => {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.right > vw + 1) {
        offenders.push(`${el.tagName}.${String(el.className).slice(0, 40)} right=${Math.round(r.right)}`)
      }
    })
    return { vw, scrollWidth: document.documentElement.scrollWidth, offenders: offenders.slice(0, 5) }
  })
  const overflow = m.scrollWidth > m.vw || m.offenders.length > 0
  console.log(`${overflow ? 'FAIL' : 'ok  '} ${route} vw=${m.vw} scrollWidth=${m.scrollWidth}`)
  if (overflow) failures.push(`${route}: scrollWidth=${m.scrollWidth} vw=${m.vw}\n    ${m.offenders.join('\n    ')}`)
}

await browser.close()
server.close()

if (failures.length) {
  console.error(`\nHorizontal overflow at 390px:\n  ${failures.join('\n  ')}`)
  process.exit(1)
}
console.log(`\nok: ${ROUTES.length} routes, no horizontal overflow at 390px`)
```

Add to `package.json` scripts: `"verify:viewport": "node scripts/verify-viewport.mjs"`.

- [ ] **Step 3: Run it and confirm it CATCHES the known bug**

Run: `cd web/frontend && npm run build:demo && npm run verify:viewport`

Expected: **FAIL on `/money`**, naming the `.data-table` / `.table-scroll` offender. This is the confirmed bug from spec §1.1 and slice 3 fixes it structurally. Confirming the failure here is the proof the check works.

- [ ] **Step 4: Add the narrowest CSS fix so the suite is green**

Slice 3 removes the table entirely; until then the container must clip rather than grow. In `web/frontend/src/theme.css`, on `.table-scroll`:

```css
.table-scroll { overflow-x: auto; min-width: 0; max-width: 100%; }
```

Re-run `npm run verify:viewport`. Expected: all routes `ok`.

- [ ] **Step 5: Wire all four checks into CI**

In `.github/workflows/ci.yml`, in the `web` job after `Frontend build`:

```yaml
      - name: Production build carries no fixtures
        run: node scripts/verify-demo-build.mjs prod dist
        working-directory: web/frontend
      - name: Demo build
        run: npm run build:demo
        working-directory: web/frontend
      - name: Demo build carries fixtures
        run: node scripts/verify-demo-build.mjs demo dist-demo
        working-directory: web/frontend
      - name: No horizontal overflow at 390px
        run: npm run verify:viewport
        working-directory: web/frontend
```

- [ ] **Step 6: Run the full gate set**

```bash
cd web/frontend && npm run check && npm run build
```
```bash
uv run --project web pytest web/backend/tests -q
```
```bash
uv run pytest -q
```
(its own isolated command, then:)
```bash
uv run ruff check . && uv run mypy algua scripts && uv run lint-imports
```

- [ ] **Step 7: Commit**

```bash
git add web/frontend/scripts/verify-viewport.mjs web/frontend/package.json \
        web/frontend/package-lock.json web/frontend/src/theme.css .github/workflows/ci.yml
git commit -m "test(web): no-horizontal-overflow invariant at 390px; fix .table-scroll clipping"
```

---

## Self-Review

**Spec coverage.** §1.1 viewport bug → Task 4 (proves the failure, then the narrow fix; the structural fix lands in slice 3). §5.1 → Task 4. §5.2 → Task 3. §6 fixture → Task 1. §6 demo build + both guards → Task 2. §2/§3/§4 screens → **slices 2 and 3, not this plan** (this slice deliberately changes no screen markup). §7 decisions → slices 2–3. §8 non-goals → nothing here touches the CLI/API.

**Placeholders.** None: every step has runnable code or an exact command with an expected result.

**Type consistency.** `resolveFixture` / `FIXTURE_SENTINEL` are defined in Task 1 and consumed under those exact names in Tasks 2 and 3. `DEMO` / `demoJSON` are defined in Task 2 and consumed in `api.ts` in the same task. The sentinel string `ALGUA_DEMO_FIXTURE_a7f3e1` is identical in `steady-state.ts` and `verify-demo-build.mjs`.

**Known risk carried deliberately.** Task 2 Step 8 may fail if Rollup keeps the dynamic-import chunk; the step names the fallback (a build-time alias) rather than leaving the implementer to invent one or weaken the guard.

## Follow-on slices

- **Slice 2 — Now:** attention slot bound to `/api/triage` (whose `sources` field already exists to prevent a degraded read rendering as all-clear), fleet grid, deltas, delete the activity feed, four tabs to three, Electric off the tab bar.
- **Slice 3 — Money + Research:** equity hero, capacity as filled slots, contribution bars replacing the table (removes the §1.1 bug at its cause), the funnel.
