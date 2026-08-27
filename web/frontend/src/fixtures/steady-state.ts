/** The single rich steady-state fixture (spec §6). Shaped EXACTLY like the API envelopes so
 * the demo transport is a swap, not an adapter with logic in it — a fixture that needed
 * massaging on the way out would be testing the adapter, not the screens.
 *
 * Every value here is plausible-but-invented. Nothing in this file is read in a production
 * build (see `src/transport.ts` and the dist guard in its test). */
import type {
  ActivityRow,
  ApiEnvelope,
  BookPayload,
  FleetHealth,
  FleetRow,
  IdeaRow,
  IdeasResponse,
  ListPayload,
  OpsPayload,
  RunRow,
  RunsListPayload,
  StrategyRecord,
  TriageItem,
  TriagePayload,
} from '../types'

/** Embedded in the fixture data so a production build can be PROVEN fixture-free by grepping
 * `dist/` for it. Deliberately a string no real payload would ever contain. */
export const FIXTURE_SENTINEL = 'ALGUA_DEMO_FIXTURE_a7f3e1'

export const FETCHED_AT = '2026-08-27T09:15:00+00:00'

export function envelope<T>(data: T): ApiEnvelope<T> {
  return { ok: true, data, fetched_at: FETCHED_AT, stale: false }
}

/** Deterministic pseudo-random so the fixture is byte-stable across builds (a fixture that
 * changed every build would make the dist guard and the word budget flaky). */
export function rng(seed: number): () => number {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296
    return s / 4294967296
  }
}

export interface Seed {
  name: string
  stage: string
  health: FleetRow['health']
  drawdown: number
  capital: number
  pnl: number
  /** registry id — also the identity `StrategyRecord`/`RegistryDetail` key off of. */
  id: number
  family: string | null
  tags: string[]
  hypothesisStatus: 'untested' | 'supported' | 'refuted' | 'inconclusive'
  derivedFrom: string | null
  description: string | null
}

/** The stages an operator loop actually ticks (algua/execution/fleet_health.py
 * `OPERATIONAL_STAGES` — dormant is explicitly NOT in that set, unlike the informal
 * "operational-ish" grouping used elsewhere in this file for book/allocation purposes). */
export const OPERATIONAL_STAGES: string[] = ['live', 'paper', 'forward_tested']

/** 14 strategies spanning all 7 lifecycle stages — `fleet_status()` emits one row per
 * REGISTRY strategy, not just the ticked ones, so every seed here gets a fleet row. Two are
 * unhealthy, which is what gives the attention slot and the fleet grid something to disagree
 * about. */
export const SEEDS: Seed[] = [
  {
    name: 'liquid10_adj_momentum', stage: 'live', health: 'ok', drawdown: 0.021, capital: 1800, pnl: 74.2,
    id: 1, family: 'cross_sectional_momentum', tags: ['momentum', 'universe:liquid10'],
    hypothesisStatus: 'supported', derivedFrom: null,
    description: 'Cross-sectional momentum over the liquid10 universe, monthly rebalance.',
  },
  {
    name: 'orderly_six_day_rebound', stage: 'live', health: 'stale', drawdown: 0.038, capital: 1800, pnl: -22.6,
    id: 2, family: 'mean_reversion', tags: ['mean-reversion', 'short-horizon'],
    hypothesisStatus: 'supported', derivedFrom: null,
    description: 'Six-day orderly-pullback rebound within the liquid10 universe.',
  },
  {
    name: 'low_vol_skip_momentum_top3', stage: 'live', health: 'ok', drawdown: 0.014, capital: 1800, pnl: 51.9,
    id: 3, family: 'cross_sectional_momentum', tags: ['momentum', 'low-vol', 'skip-month'],
    hypothesisStatus: 'supported', derivedFrom: 'liquid10_adj_momentum',
    description: 'Low-vol-screened momentum, skip-month, top-3 names only.',
  },
  {
    name: 'lagged_rank_persistence', stage: 'paper', health: 'ok', drawdown: 0.045, capital: 1550, pnl: 18.4,
    id: 4, family: 'rank_persistence', tags: ['momentum', 'rank-persistence'],
    hypothesisStatus: 'supported', derivedFrom: null,
    description: "Persistence of last month's cross-sectional rank into the current month.",
  },
  {
    name: 'cross_horizon_low_vol', stage: 'paper', health: 'drift', drawdown: 0.084, capital: 1550, pnl: -41.0,
    id: 5, family: 'low_vol', tags: ['low-vol', 'multi-horizon'],
    hypothesisStatus: 'supported', derivedFrom: null,
    description: 'Blends a short- and a long-horizon low-vol screen.',
  },
  {
    name: 'dual_horizon_skip_month', stage: 'paper', health: 'ok', drawdown: 0.019, capital: 1550, pnl: 33.7,
    id: 6, family: 'cross_sectional_momentum', tags: ['momentum', 'skip-month'],
    hypothesisStatus: 'supported', derivedFrom: 'liquid10_adj_momentum',
    description: 'Skip-month momentum blended across two lookback horizons.',
  },
  {
    name: 'quality_momentum_qs', stage: 'paper', health: 'ok', drawdown: 0.052, capital: 1550, pnl: 9.1,
    id: 7, family: 'quality_momentum', tags: ['momentum', 'quality'],
    hypothesisStatus: 'supported', derivedFrom: null,
    description: 'Momentum screened by a quality-score factor.',
  },
  {
    name: 'peer_selloff_rebound', stage: 'paper', health: 'ok', drawdown: 0.028, capital: 1550, pnl: 27.3,
    id: 8, family: 'mean_reversion', tags: ['mean-reversion', 'peer-relative'],
    hypothesisStatus: 'supported', derivedFrom: 'orderly_six_day_rebound',
    description: 'Rebound after a peer-relative, not absolute, selloff.',
  },
  {
    name: 'cadenced_tail_risk', stage: 'forward_tested', health: 'ok', drawdown: 0.011, capital: 1200, pnl: 12.8,
    id: 9, family: 'tail_risk', tags: ['tail-risk', 'cadenced'],
    hypothesisStatus: 'supported', derivedFrom: null,
    description: 'Cadenced tail-risk overlay, rebalanced on a fixed schedule.',
  },
  {
    name: 'distributed_gains_qm', stage: 'forward_tested', health: 'ok', drawdown: 0.033, capital: 1200, pnl: 20.5,
    id: 10, family: 'quality_momentum', tags: ['momentum', 'quality'],
    hypothesisStatus: 'supported', derivedFrom: 'quality_momentum_qs',
    description: 'Quality-momentum variant with gains spread across more names.',
  },
  {
    name: 'skip_month_persistence', stage: 'dormant', health: 'idle', drawdown: 0.0, capital: 0, pnl: 0.0,
    id: 11, family: 'rank_persistence', tags: ['momentum', 'skip-month'],
    hypothesisStatus: 'supported', derivedFrom: 'lagged_rank_persistence',
    description: 'Skip-month rank persistence; benched pending a universe refresh.',
  },
  {
    name: 'vol_carry_neutral', stage: 'candidate', health: 'idle', drawdown: 0.0, capital: 0, pnl: 0.0,
    id: 12, family: 'carry', tags: ['carry', 'vol-neutral'],
    hypothesisStatus: 'supported', derivedFrom: null,
    description: 'Vol-neutral carry, awaiting a paper book slice.',
  },
  {
    name: 'range_compression_break', stage: 'backtested', health: 'idle', drawdown: 0.0, capital: 0, pnl: 0.0,
    id: 13, family: 'volatility_breakout', tags: ['breakout', 'range-compression'],
    hypothesisStatus: 'untested', derivedFrom: null,
    description: 'Breakout following a multi-session range compression.',
  },
  {
    name: 'seasonal_turn_of_month', stage: 'idea', health: 'idle', drawdown: 0.0, capital: 0, pnl: 0.0,
    id: 14, family: null, tags: ['seasonality'],
    hypothesisStatus: 'untested', derivedFrom: null, description: null,
  },
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
    last_tick_error: s.health === 'stale' ? 'no fresh mark received in 3 sessions' : null,
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

// Worst-offender-first, matching `fleet_status()`'s own sort key EXACTLY
// (algua/execution/fleet_health.py `_SEVERITY` + `fleet_status`'s `rows.sort(...)`, ties broken
// by strategy name) — a fixture in seed/registration order would look right in this file and
// scramble against the real API the moment a screen is designed against row position (FIX 5).
const HEALTH_SEVERITY: Record<string, number> = { halted: 0, drift: 1, stale: 2, idle: 3, ok: 4 }

function byFleetOrder(a: FleetRow, b: FleetRow): number {
  const sa = HEALTH_SEVERITY[a.health] ?? 99
  const sb = HEALTH_SEVERITY[b.health] ?? 99
  return sa - sb || a.strategy.localeCompare(b.strategy)
}

// One row per REGISTRY strategy (fleet_status() emits idle rows for research-stage
// strategies too — see OPERATIONAL_STAGES above), not just the ticked ones.
const rows = SEEDS.map(fleetRow).sort(byFleetOrder)
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
  operational_stages: OPERATIONAL_STAGES,
  rows,
})

// Mirrors `web/backend/triage.py::_fleet_items` field-for-field (no loop/capital items in this
// fixture — OPS is all-`ok` and BOOK's `unallocated_operational` is empty) so a screen designed
// against this data cannot invent a field the real endpoint never sends (FIX 3): `severity` is
// ALWAYS `SEVERITY['strategy']` (3), `title` is `f"{name} {health}"`, `detail` follows the exact
// kill_switch-reason -> staleness -> stage fallback chain, and `since` is ALWAYS `None` for a
// strategy item — the real endpoint has no per-strategy "since" timestamp to offer. (There is no
// `drawdown N% of M% wall` detail string anywhere in triage.py — that kind of item does not
// exist.) Spec §4.1's exception card asks for an age the real API cannot supply; see the fix-wave
// report for what to amend there instead of inventing it here.
function triageDetailFor(row: FleetRow): string {
  if (row.kill_switch?.reason) return row.kill_switch.reason
  if (typeof row.staleness_sessions === 'number') return `${row.staleness_sessions} sessions since last tick`
  return `stage ${row.stage} — no tick evidence`
}

const TRIAGE_ITEMS: TriageItem[] = alerting
  .map(
    (row): TriageItem => ({
      kind: 'strategy',
      severity: 3,
      title: `${row.strategy} ${row.health}`,
      detail: triageDetailFor(row),
      since: null,
      route: `/s/${row.strategy}`,
    }),
  )
  // build_triage() re-sorts the merged item list by (severity, title); every item here shares
  // severity 3, so the tiebreak (title, alphabetical) is the whole order.
  .sort((a, b) => a.title.localeCompare(b.title))

export const TRIAGE: TriagePayload = {
  ok: true,
  items: TRIAGE_ITEMS,
  // All three sources loaded — a degraded read must NEVER render as an all-clear
  // (see TriagePayload's docstring in types.ts).
  sources: { fleet: true, ops: true, book: true },
  headline: {
    // Fleet-wide `ok` count from ALL rows, exactly like triage.py's `by_health.get("ok", 0)` —
    // NOT `rows.length - alerting.length`, which double-counts `idle` research-stage strategies
    // as healthy (FIX 4; that miscount is exactly what spec §7 forbids).
    fleet_ok: rows.filter((r) => r.health === 'ok').length,
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

// Loop names per algua/operator/loop_health.py LOOPS = ("research", "paper", "mergeback").
// LoopRow's real key is `health`, not `state`/`name` — Research.tsx reads
// `ops.loops.research.health` / `.detail` / `.last_ok_at` / `.consecutive_failures` and
// `ops.loops.mergeback.queue_depth`.
export const OPS: ApiEnvelope<OpsPayload> = envelope({
  ok: true,
  checked_at: FETCHED_AT,
  alerting: [],
  loops: {
    research: {
      health: 'ok', last_run_at: FETCHED_AT, last_ok_at: FETCHED_AT,
      consecutive_failures: 0, session: 'codex',
    },
    paper: { health: 'ok', last_run_at: FETCHED_AT, last_ok_at: FETCHED_AT, consecutive_failures: 0 },
    mergeback: {
      health: 'ok', last_run_at: FETCHED_AT, last_ok_at: FETCHED_AT,
      consecutive_failures: 0, queue_depth: 0,
    },
  },
})

// ListPayload<T> is `{ data: T[] }` (bare-array CLI commands arrive wrapped this way inside the
// envelope) — so /api/strategies is `{ok, data: {data: [...]}, fetched_at, stale}`, confirmed by
// Research.tsx's `strategies.data.data.data` read.
export const STRATEGIES: ApiEnvelope<ListPayload<StrategyRecord>> = envelope({
  data: SEEDS.map(
    (s): StrategyRecord => ({
      id: s.id,
      name: s.name,
      stage: s.stage,
      family: s.family,
      tags: s.tags,
      author: 'agent',
      hypothesis_status: s.hypothesisStatus,
      derived_from: s.derivedFrom,
      description: s.description,
    }),
  ),
})

const IDEA_ROWS: IdeaRow[] = [
  {
    id: 301,
    title: 'Turn-of-month seasonal drift, liquid10',
    hypothesis:
      'Liquid10 names drift positive across the last and first two sessions of the calendar month.',
    family: null,
    tags: ['seasonality'],
    source_type: 'paper',
    source_ref: 'Ariel (1987) monthly effect',
    source_date: '2026-08-20',
    status: 'authored',
    authored_strategy_id: 14,
    created_at: '2026-08-21T10:00:00+00:00',
  },
  {
    id: 302,
    title: 'Options-implied skew as a momentum-crash flag',
    hypothesis: 'A steepening put skew ahead of a momentum drawdown predicts its severity.',
    family: null,
    tags: ['momentum', 'risk-overlay'],
    source_type: 'forum',
    source_ref: null,
    source_date: '2026-08-22',
    required_data: ['options chain (implied vol surface)'],
    status: 'needs_data',
    created_at: '2026-08-22T14:00:00+00:00',
  },
  {
    id: 303,
    title: 'Cross-listed ADR lead-lag rebound',
    hypothesis: 'A US-listed ADR lags its home-market close overnight, creating a next-open rebound.',
    family: null,
    tags: ['mean-reversion', 'cross-listing'],
    source_type: 'thesis',
    source_ref: null,
    source_date: '2026-08-24',
    status: 'open',
    created_at: '2026-08-24T09:30:00+00:00',
  },
]

// IdeasResponse is itself a flattened envelope (ok/fetched_at/stale sit alongside `ideas`, not
// nested under a `data` key) — confirmed by Research.tsx reading `resp.ideas` / `resp.stats`
// / `resp.stats_window_days` directly off the fetched object.
export const IDEAS: IdeasResponse = {
  ok: true,
  ideas: IDEA_ROWS,
  stats: { window_days: 14, counts: { open: 1, needs_data: 1, authored: 1 } },
  stats_window_days: 14,
  fetched_at: FETCHED_AT,
  stale: false,
}

const ACTIVITY_ROWS: ActivityRow[] = [
  {
    id: 9001, ts: '2026-08-27T08:00:00+00:00', actor: 'agent',
    action: 'paper.trade_tick', strategy: 'lagged_rank_persistence', reason: null,
  },
  {
    id: 9000, ts: '2026-08-26T18:05:00+00:00', actor: 'agent',
    action: 'fleet.alert', strategy: 'cross_horizon_low_vol', reason: 'drawdown 8.4% of 10% wall',
  },
  {
    id: 8999, ts: '2026-08-24T13:30:00+00:00', actor: 'agent',
    action: 'fleet.alert', strategy: 'orderly_six_day_rebound', reason: 'marks stale · 3 sessions',
  },
  {
    id: 8998, ts: '2026-08-21T10:00:00+00:00', actor: 'agent',
    action: 'idea.authored', strategy: 'seasonal_turn_of_month', reason: null,
  },
]

export const ACTIVITY: ApiEnvelope<ListPayload<ActivityRow>> = envelope({ data: ACTIVITY_ROWS })

const runs = (() => {
  const r = rng(20260827)
  return operational.map((s, i): RunRow => {
    const meanWindow = 0.35 + r() * 1.3
    const oos = meanWindow * (0.45 + r() * 0.7)
    return {
      id: 100 + i,
      kind: 'gate',
      strategy_name: s.name,
      strategy_id: s.id,
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
  count: runs.length,
  runs,
})
