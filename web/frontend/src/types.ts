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
  /** Fraction, e.g. -0.12 = 12% below peak. */
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
