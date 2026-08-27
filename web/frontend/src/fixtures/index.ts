/** URL -> fixture payload. Deliberately a MAP with an explicit unknown-URL miss rather than a
 * catch-all: a screen that calls an endpoint nobody fixtured must fail loudly in the demo
 * build (see `transport.ts`), not render a plausible-looking empty state that hides the gap. */
import {
  ACTIVITY,
  BOOK,
  FETCHED_AT,
  FIXTURE_SENTINEL,
  FLEET,
  IDEAS,
  OPS,
  RUNS,
  SEEDS,
  STRATEGIES,
  TRIAGE,
  envelope,
  rng,
  type Seed,
} from './steady-state'
import type {
  GateCheck,
  GateDecision,
  GateRow,
  GatesPayload,
  PaperRollup,
  RecentOrder,
  RegistryDetail,
  RunDetail,
  RunSeriesPayload,
  SeriesPayload,
  SeriesRow,
  StrategyDetailResponse,
  Transition,
} from '../types'

export { FIXTURE_SENTINEL }

/** Lifecycle order (algua/contracts/lifecycle.py), used to reconstruct a plausible transition
 * history and gate-history depth for any seed on demand. */
const LIFECYCLE_ORDER = ['idea', 'backtested', 'candidate', 'paper', 'forward_tested', 'live']

const STAGE_GAP_DAYS: Record<string, number> = {
  idea: 4, backtested: 6, candidate: 15, paper: 26, forward_tested: 11, live: 1, dormant: 30,
}

const STAGE_REASON: Record<string, string> = {
  idea: 'sourced from the idea pool',
  backtested: 'strategy module authored and backtested',
  candidate: 'research promote passed the integrity floor',
  paper: 'paper intake allocated a book slice',
  forward_tested: 'paper promote passed the forward-evidence gate',
  live: 'go-live approved on a fresh forward certificate',
  dormant: 'benched pending a universe refresh',
}

function addDays(iso: string, days: number): string {
  const d = new Date(iso)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().replace(/\.\d{3}Z$/, '+00:00')
}

/** Walks the lifecycle backward from `FETCHED_AT` so the CURRENT stage lands closest to "now"
 * and earlier stages land progressively further in the past — byte-stable, no `Date.now()`. */
function transitionsFor(seed: Seed): Transition[] {
  const idx = LIFECYCLE_ORDER.indexOf(seed.stage)
  const path = idx === -1 ? ['idea'] : LIFECYCLE_ORDER.slice(0, idx + 1)
  if (seed.stage === 'dormant') path.push('dormant')
  const newestFirst = [...path].reverse()
  let ts = FETCHED_AT
  const dated = newestFirst.map((stage) => {
    const at = ts
    ts = addDays(ts, -(STAGE_GAP_DAYS[stage] ?? 7))
    return { stage, at }
  })
  const chronological = [...dated].reverse()
  return chronological.map(
    (d, i): Transition => ({
      from_stage: i === 0 ? null : chronological[i - 1].stage,
      to_stage: d.stage,
      actor: d.stage === 'live' ? 'human' : 'agent',
      reason: STAGE_REASON[d.stage] ?? null,
      created_at: d.at,
    }),
  )
}

function registryDetailFor(seed: Seed): RegistryDetail {
  return {
    id: seed.id,
    name: seed.name,
    stage: seed.stage,
    family: seed.family,
    tags: seed.tags,
    author: 'agent',
    hypothesis_status: seed.hypothesisStatus,
    derived_from: seed.derivedFrom,
    description: seed.description,
    transitions: transitionsFor(seed),
  }
}

const SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'AVGO', 'COST', 'ADBE', 'LIN']

function recentOrdersFor(seed: Seed): RecentOrder[] {
  if (seed.capital <= 0) return []
  const a = SYMBOLS[seed.id % SYMBOLS.length]
  const b = SYMBOLS[(seed.id + 3) % SYMBOLS.length]
  return [
    { symbol: a, side: 'buy', status: 'filled', broker_order_id: `ord_${seed.id}01`, submitted_ts: FETCHED_AT },
    {
      symbol: b, side: 'sell', status: 'filled',
      broker_order_id: `ord_${seed.id}00`, submitted_ts: addDays(FETCHED_AT, -1),
    },
  ]
}

/** Only strategies the operator loop actually ticks (live/paper/forward_tested/dormant) carry a
 * `FleetRow` — reuse it rather than re-deriving health/drawdown a second time. */
function paperRollupFor(seed: Seed): PaperRollup | null {
  const row = FLEET.data.rows.find((r) => r.strategy === seed.name)
  if (row === undefined) return null
  return {
    strategy: row.strategy,
    stage: row.stage,
    health: row.health,
    staleness_sessions: row.staleness_sessions,
    stale_after_sessions: FLEET.data.stale_after_sessions,
    last_tick_error: row.last_tick_error,
    kill_switch: row.kill_switch,
    drawdown: row.drawdown,
    last_tick: row.positions !== null && row.positions > 0 ? { tick_ts: FETCHED_AT, reconcile_ok: true } : null,
    positions: row.positions,
    n_orders: row.n_orders,
    recent_orders: recentOrdersFor(seed),
  }
}

function buildGateCheck(
  name: string,
  threshold: number,
  value: number,
  opts?: { advisory?: boolean },
): GateCheck {
  return {
    name,
    op: '>',
    threshold,
    value: Number(value.toFixed(4)),
    passed: value > threshold,
    advisory: opts?.advisory ?? false,
  }
}

/** A dormant seed in this fixture was benched from the PAPER lane (never reached
 * forward_tested) — treat it as ranked at 'paper' for gate-history depth purposes. */
function stageRank(stage: string): number {
  if (stage === 'dormant') return LIFECYCLE_ORDER.indexOf('paper')
  return LIFECYCLE_ORDER.indexOf(stage)
}

function buildGateRow(seed: Seed, id: number, kind: 'research' | 'forward'): GateRow {
  const r = rng(seed.id * 7919 + (kind === 'research' ? 1 : 2))
  const holdoutSharpe = 0.15 + r() * 0.7
  const checks: GateCheck[] =
    kind === 'research'
      ? [
          { name: 'pit_universe', passed: true },
          buildGateCheck('holdout_observations', 63, 90 + r() * 400),
          buildGateCheck('holdout_sharpe_floor', 0, holdoutSharpe),
          buildGateCheck('dsr_confidence', 0.9, 0.6 + r() * 0.35, { advisory: true }),
        ]
      : [
          buildGateCheck('forward_sharpe_floor', holdoutSharpe * 0.5, holdoutSharpe * (0.6 + r() * 0.6)),
          buildGateCheck('sessions_observed', 63, 63 + r() * 60),
          { name: 'account_hygiene', passed: true },
        ]
  const decision: GateDecision = {
    passed: checks.every((c) => c.advisory === true || c.passed !== false),
    checks,
    dsr_confidence: kind === 'research' ? Number((0.6 + r() * 0.35).toFixed(4)) : undefined,
    appraisal_ratio: Number((0.8 + r() * 0.9).toFixed(4)),
  }
  return {
    id,
    passed: decision.passed === true,
    actor: 'agent',
    created_at: FETCHED_AT,
    consumed: true,
    decision,
    fdr_p_value: null,
    fdr_alpha_level: null,
  }
}

/** Only strategies at `candidate` or beyond have cleared the research gate at all; only
 * `forward_tested`/`live` (or a dormant seed benched from paper — see `stageRank`) have
 * attempted the forward gate too. */
function gatesFor(seed: Seed): GatesPayload | null {
  const rank = stageRank(seed.stage)
  if (rank === -1 || rank < LIFECYCLE_ORDER.indexOf('candidate')) return null
  const reachedForward = rank >= LIFECYCLE_ORDER.indexOf('forward_tested')
  return {
    strategy: seed.name,
    gate_evaluations: [buildGateRow(seed, 1000 + seed.id * 10, 'research')],
    forward_gate_evaluations: reachedForward ? [buildGateRow(seed, 1000 + seed.id * 10 + 1, 'forward')] : [],
  }
}

function strategyDetail(name: string): StrategyDetailResponse {
  const seed = SEEDS.find((s) => s.name === name) ?? SEEDS[0]
  return {
    ok: true,
    strategy: seed.name,
    registry: registryDetailFor(seed),
    paper: paperRollupFor(seed),
    gates: gatesFor(seed),
    gates_limit: 20,
    fetched_at: FETCHED_AT,
    stale: false,
  }
}

function buildSeriesRows(seedNum: number, base: number, n: number): SeriesRow[] {
  const r = rng(seedNum)
  let equity = base
  let peak = base
  const out: SeriesRow[] = []
  for (let i = 0; i < n; i++) {
    equity = equity * (1 + (r() - 0.45) * 0.02)
    peak = Math.max(peak, equity)
    out.push({
      id: 2000 + i,
      tick_ts: addDays(FETCHED_AT, -(n - i)),
      recorded_at: addDays(FETCHED_AT, -(n - i)),
      equity: Number(equity.toFixed(2)),
      peak_equity: Number(peak.toFixed(2)),
      reconcile_ok: true,
    })
  }
  return out
}

function seriesFor(name: string): SeriesPayload {
  const seed = SEEDS.find((s) => s.name === name)
  const hasSlice = seed !== undefined && seed.capital > 0
  const rowsForLane = hasSlice ? buildSeriesRows(seed.id * 31, seed.capital, 8) : []
  const isLive = seed?.stage === 'live'
  return {
    strategy: name,
    lane_filter: null,
    series: {
      paper: hasSlice && !isLive ? rowsForLane : hasSlice ? [] : null,
      live: hasSlice && isLive ? rowsForLane : null,
    },
    truncated: { paper: false, live: false },
    n_legacy_excluded: 0,
    n_unparseable: 0,
    n_invalid_lane: 0,
  }
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
    case '/api/activity':    return ACTIVITY
    // The run-ledger series endpoint deliberately never returns a per-bar OOS vector
    // (holdout_returns.returns_blob is SENSITIVE) — an empty series map is the honest fixture.
    case '/api/runs/series': return envelope<RunSeriesPayload>({ series: {} })
    case '/api/push/key':    return { key: null }
  }

  const series = path.match(/^\/api\/strategy\/([^/]+)\/series$/)
  if (series) return envelope(seriesFor(decodeURIComponent(series[1])))

  const strategy = path.match(/^\/api\/strategy\/([^/]+)$/)
  if (strategy) return strategyDetail(decodeURIComponent(strategy[1]))

  const runId = path.match(/^\/api\/runs\/(\d+)$/)
  if (runId) {
    const run = RUNS.data.runs.find((r) => r.id === Number(runId[1])) ?? RUNS.data.runs[0]
    return envelope<RunDetail>({ ...run, extra_metrics: {} })
  }

  return undefined
}
