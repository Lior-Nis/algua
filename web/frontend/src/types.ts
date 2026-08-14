/** TS mirrors of the web backend's normalized payloads (spec: slice B). */

/** Every /api/* response is wrapped in this envelope by the backend. */
export interface ApiEnvelope<T> {
  ok: boolean
  data: T
  /** ISO timestamp of when the underlying CLI read happened. */
  fetched_at: string
  /** True when the backend served last-good cache after a CLI failure. */
  stale: boolean
  cache_age_s?: number | null
  last_error_code?: string | null
  last_error_at?: string | null
}

/** Health verdicts the fleet rollup emits (worst → best). */
export type Health = 'halted' | 'drift' | 'stale' | 'idle' | 'ok'

export interface KillSwitchState {
  tripped: boolean | null
  reason: string | null
  global_halt: boolean | null
}

export interface DrawdownState {
  peak_equity: number | null
  last_equity: number | null
  /** DEPTH below peak as a POSITIVE fraction: `1 - last_equity/peak_equity`
   * (algua/execution/fleet_health.py), so 0.12 = 12% below peak and 0 = at peak. */
  drawdown: number | null
}

export interface FleetRow {
  strategy: string
  stage: string
  health: string
  staleness_sessions: number | null
  last_tick_error: string | null
  kill_switch: KillSwitchState | null
  drawdown: DrawdownState | null
  positions: number | null
  n_orders: number | null
}

export interface FleetSummary {
  total: number
  alerting: number
  by_health: Record<string, number>
}

export interface FleetHealth {
  ok: boolean
  global_halt: boolean
  /** Pre-sorted worst-offender-first by the CLI. */
  alerting: FleetRow[]
  summary: FleetSummary
  stale_after_sessions: number | null
  rows: FleetRow[]
}

/** Bare-array CLI commands arrive wrapped as {"data": [...]} inside the envelope. */
export interface ListPayload<T> {
  data: T[]
}

/** One `registry list` row (algua/cli/registry_cmd.py _record_json). */
export interface StrategyRecord {
  id: number
  name: string
  stage: string
  family: string | null
  tags: string[]
  author: string
  hypothesis_status: string
  derived_from: string | null
  description: string | null
}

/** One stage_transitions row from `registry show`. */
export interface Transition {
  id?: number
  from_stage: string | null
  to_stage: string
  actor: string
  reason: string | null
  created_at: string
  code_hash?: string | null
  config_hash?: string | null
  dependency_hash?: string | null
}

export interface RegistryDetail extends StrategyRecord {
  transitions: Transition[]
}

/** One paper_orders row (algua/execution/order_state.py recent_orders). */
export interface RecentOrder {
  symbol: string
  side: string
  status: string
  broker_order_id: string | null
  submitted_ts: string | null
}

/** `paper show` top-level rollup (algua/execution/fleet_health.py strategy_health
 * + recent_orders layered on by the CLI). */
export interface PaperRollup {
  strategy: string
  stage: string
  health: string
  staleness_sessions: number | null
  stale_after_sessions: number | null
  last_tick_error: string | null
  kill_switch: KillSwitchState | null
  drawdown: DrawdownState | null
  last_tick: Record<string, unknown> | null
  positions: number | null
  n_orders: number | null
  recent_orders: RecentOrder[]
}

/** One checks[] entry of a projected gate decision (gate_history.py _CHECK_FIELDS). */
export interface GateCheck {
  name?: string
  op?: string
  threshold?: number | null
  value?: number | null
  passed?: boolean
}

/** Allowlist-projected decision_json — scalars + dicts of scalars + checks[]. */
export interface GateDecision {
  passed?: boolean
  checks?: GateCheck[]
  [key: string]: unknown
}

/** One row of either gate ledger (`registry gates`); column sets differ but the
 * fields this UI reads are shared. SQLite booleans arrive as 0/1. */
export interface GateRow {
  id: number
  passed: number | boolean
  actor: string
  created_at: string
  consumed?: number | boolean | null
  decision: GateDecision | null
  decision_error?: string
  decision_dropped_keys?: string[]
  fdr_p_value?: number | null
  fdr_alpha_level?: number | null
  [key: string]: unknown
}

export interface GatesPayload {
  strategy: string
  gate_evaluations: GateRow[]
  forward_gate_evaluations: GateRow[]
}

/** GET /api/strategy/{name} — composite, NOT the standard envelope shape.
 * A null part with a part_errors code = real backend failure for that part. */
export interface StrategyDetailResponse {
  ok: boolean
  strategy: string
  registry: RegistryDetail
  paper: PaperRollup | null
  gates: GatesPayload | null
  part_errors?: { paper?: string; gates?: string }
  fetched_at: string
  stale: boolean
  cache_age_s?: number | null
  last_error_code?: string | null
}

/** One tick_snapshots row from `fleet series`. */
export interface SeriesRow {
  id: number
  tick_ts: string
  recorded_at: string
  equity: number
  peak_equity: number | null
  reconcile_ok: boolean
}

export type Lane = 'paper' | 'live'

/** GET /api/strategy/{name}/series payload. null lane = not requested; [] = empty. */
export interface SeriesPayload {
  strategy: string
  lane_filter: string | null
  series: { paper: SeriesRow[] | null; live: SeriesRow[] | null }
  truncated: { paper: boolean | null; live: boolean | null }
  n_legacy_excluded: number
  n_unparseable: number
  n_invalid_lane: number
}

/** One audit_log row from `audit log` (GET /api/activity). */
export interface ActivityRow {
  id?: number
  ts: string
  actor: string
  action: string
  strategy: string | null
  reason: string | null
}

/** One idea-pool row (algua/cli/idea_cmd.py _idea_json). */
export interface IdeaRow {
  id: number
  title: string
  hypothesis: string
  family: string | null
  tags: string[]
  source_type: string
  source_ref: string | null
  source_date: string | null
  source_note?: string | null
  required_data?: string[]
  status: string
  authored_strategy_id?: number | null
  duplicate_of_idea_id?: number | null
  created_at: string
  updated_at?: string
}

/** GET /api/ideas — ideas list + `idea stats` counts window. */
export interface IdeasResponse {
  ok: boolean
  ideas: ListPayload<IdeaRow> | IdeaRow[]
  stats: Record<string, unknown> | null
  stats_window_days: number
  fetched_at: string
  stale: boolean
  last_error_code?: string | null
}
