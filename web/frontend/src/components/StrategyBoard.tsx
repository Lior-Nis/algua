import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, useFetch } from '../api'
import type { UseFetchResult } from '../api'
import { num, pct } from '../format'
import type { ApiEnvelope, StrategyBoardGate, StrategyBoardPayload, StrategyBoardRow } from '../types'
import PassMark from './PassMark'
import Sparkline from './Sparkline'
import StaleNotice from './StaleNotice'
import StageChip from './StageChip'

const SCHEMA_VERSION = 'strategy-board/v1'

type MetricKey = Exclude<keyof StrategyBoardGate, 'run_id' | 'created_at' | 'passed' | 'n_obs_oos'>

interface SortOption {
  key: MetricKey
  label: string
  format: (v: number | null) => string
}

/** Exploratory comparison axes. Every axis is HIGHER-first — drawdown included, because
 * `max_drawdown_oos` is stored negative, so the shallowest drawdown is the highest value. Nulls
 * sort last on every axis (see `orderBoardRows`). Picking one reorders rows only; it never picks
 * a different run for a strategy and never feeds a promotion path. */
export const SORT_OPTIONS: SortOption[] = [
  { key: 'sharpe_oos', label: 'oos sharpe', format: (v) => num(v) },
  { key: 'total_return_oos', label: 'oos return', format: (v) => pct(v) },
  { key: 'max_drawdown_oos', label: 'oos drawdown', format: (v) => pct(v) },
]

const LIFECYCLE_ORDER = [
  'idea', 'backtested', 'candidate', 'paper', 'forward_tested', 'live', 'dormant', 'retired',
]

const SPARKLINE_LABEL =
  'sparkline: worst walk-forward window → mean window → holdout result, against zero'

function finite(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function nullableFinite(v: unknown): boolean {
  return v === null || finite(v) !== null
}

export function isBoardPayload(value: unknown): value is StrategyBoardPayload {
  if (value === null || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  if (candidate.schema_version !== SCHEMA_VERSION || !Array.isArray(candidate.rows)) return false
  if (!Number.isInteger(candidate.count) || candidate.count !== candidate.rows.length) return false
  return candidate.rows.every((value) => {
    if (value === null || typeof value !== 'object') return false
    const row = value as Record<string, unknown>
    if (!Number.isInteger(row.strategy_id) || typeof row.name !== 'string' ||
        (row.family !== null && typeof row.family !== 'string') || typeof row.stage !== 'string') {
      return false
    }
    if (row.latest_gate === null) return true
    if (typeof row.latest_gate !== 'object') return false
    const gate = row.latest_gate as Record<string, unknown>
    return Number.isInteger(gate.run_id) && typeof gate.created_at === 'string' &&
      (gate.passed === null || typeof gate.passed === 'boolean') &&
      nullableFinite(gate.sharpe_oos) && nullableFinite(gate.total_return_oos) &&
      nullableFinite(gate.max_drawdown_oos) &&
      (gate.n_obs_oos === null || (Number.isInteger(gate.n_obs_oos) && Number(gate.n_obs_oos) >= 0)) &&
      nullableFinite(gate.mean_window_sharpe) && nullableFinite(gate.min_window_sharpe) &&
      nullableFinite(gate.pct_positive_windows)
  })
}

function ErrorCode({ error }: { error: Error | null }) {
  const code = error instanceof ApiError && error.code !== null
    ? error.code
    : (error?.message ?? 'unknown error')
  return <div className="error-code">{code}</div>
}

function RefreshError({ error }: { error: Error | null }) {
  if (error === null) return null
  return (
    <section className="banner-stale">
      <span className="micro-label">refresh failed</span>
      <ErrorCode error={error} />
    </section>
  )
}

/** Pure, stable comparison order. `null` keeps the server's lifecycle order untouched. Otherwise:
 * finite values higher-first, missing/non-finite values last (never coerced to 0), ties and
 * the missing group keep their incoming (lifecycle) order. Returns a new array. */
export function orderBoardRows(
  rows: readonly StrategyBoardRow[],
  sortKey: MetricKey | null,
): StrategyBoardRow[] {
  if (sortKey === null) return [...rows]
  return rows
    .map((row, index) => ({ row, index, value: finite(row.latest_gate?.[sortKey]) }))
    .sort((a, b) => {
      if (a.value === null || b.value === null) {
        if (a.value === b.value) return a.index - b.index
        return a.value === null ? 1 : -1
      }
      return b.value - a.value || a.index - b.index
    })
    .map((entry) => entry.row)
}

/**
 * The research strategy board: exactly one row per registered strategy, joined to that
 * identity's latest gate run (`runs board`). Stage filtering and comparison ordering are
 * client-side over the complete payload — they reorder rows and never replace which run
 * represents a strategy. Read-only: rows navigate to the existing drill-down and nothing here
 * mutates state.
 */
export default function StrategyBoard() {
  const query = useFetch<ApiEnvelope<StrategyBoardPayload>>('/api/strategy-board')
  return <StrategyBoardView query={query} />
}

export function StrategyBoardView({
  query,
}: {
  query: UseFetchResult<ApiEnvelope<StrategyBoardPayload>>
}) {
  const [stage, setStage] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<MetricKey | null>(null)
  const [controlsOpen, setControlsOpen] = useState(false)

  const board = isBoardPayload(query.data?.data) ? query.data.data : undefined

  if (board === undefined) {
    if (query.data === undefined && query.loading) {
      return <div className="skeleton-block" style={{ height: 220 }} aria-hidden="true" />
    }
    // A failed or unrecognised read is an error, never an empty board.
    return (
      <section className="panel error-panel">
        <span className="micro-label">fetch failed</span>
        {query.data !== undefined ? <div className="error-code">unsupported_schema</div> :
          <ErrorCode error={query.error} />}
        <button type="button" className="retry-btn" onClick={query.refetch}>
          retry
        </button>
      </section>
    )
  }

  if (board.rows.length === 0) {
    return (
      <>
        <StaleNotice env={query.data} />
        <RefreshError error={query.error} />
        <section className="panel all-clear">
          <span className="micro-label">strategy board</span>
          no strategies registered — add one with <code>algua registry add &lt;name&gt;</code>
        </section>
      </>
    )
  }

  const stages = [
    ...LIFECYCLE_ORDER.filter((s) => board.rows.some((r) => r.stage === s)),
    ...new Set(board.rows.map((r) => r.stage).filter((s) => !LIFECYCLE_ORDER.includes(s))),
  ]
  const activeStage = stage !== null && stages.includes(stage) ? stage : null
  const filtered = activeStage === null ? board.rows : board.rows.filter((r) => r.stage === activeStage)
  const visible = orderBoardRows(filtered, sortKey)
  const sortOption = SORT_OPTIONS.find((o) => o.key === sortKey)
  const extra = sortOption !== undefined && sortOption.key !== 'sharpe_oos'
    ? sortOption
    : undefined

  return (
    <section aria-labelledby="strategy-board-heading">
      <StaleNotice env={query.data} />
      <RefreshError error={query.error} />
      <h2 id="strategy-board-heading" className="micro-label" style={{ marginBottom: 6 }}>
        strategy board
      </h2>
      <div className="dim-note num">
        {activeStage === null ? board.count : `${visible.length} of ${board.count}`} strategies ·
        latest gate run each
      </div>
      <div className="dim-note">oos = holdout sample · wf = walk-forward windows</div>

      <button
        type="button"
        className="board-controls-toggle"
        aria-expanded={controlsOpen}
        aria-controls="board-controls"
        onClick={() => setControlsOpen((open) => !open)}
      >
        <span>board controls</span>
        <span className="num">{activeStage === null ? 'all stages' : activeStage}</span>
        <span className="num">{sortOption?.label ?? 'lifecycle'}</span>
      </button>

      {controlsOpen && (
        <div id="board-controls" className="board-controls-panel">
          <div className="chip-row" role="group" aria-label="strategy stage">
          <button
            type="button"
            className={activeStage === null ? 'filter-chip active' : 'filter-chip'}
            aria-pressed={activeStage === null}
            onClick={() => setStage(null)}
          >
            all <span className="num">{board.count}</span>
          </button>
          {stages.map((s) => (
            <button
              key={s}
              type="button"
              className={s === activeStage ? 'filter-chip active' : 'filter-chip'}
              aria-pressed={s === activeStage}
              onClick={() => setStage(s === activeStage ? null : s)}
            >
              {s} <span className="num">{board.rows.filter((r) => r.stage === s).length}</span>
            </button>
          ))}
          </div>

          <div className="chip-row" role="group" aria-label="comparison order">
          <button
            type="button"
            className={sortKey === null ? 'filter-chip active' : 'filter-chip'}
            aria-pressed={sortKey === null}
            onClick={() => setSortKey(null)}
          >
            lifecycle
          </button>
          {SORT_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              type="button"
              className={opt.key === sortKey ? 'filter-chip active' : 'filter-chip'}
              aria-pressed={opt.key === sortKey}
              onClick={() => setSortKey(opt.key === sortKey ? null : opt.key)}
            >
              {opt.label}
            </button>
          ))}
          </div>
        </div>
      )}
      {sortKey !== null && (
        <div className="dim-note" style={{ marginBottom: 8 }}>
          exploratory order only — not a promotion signal
        </div>
      )}

      <div className="row-list">
        {visible.map((row) => (
          <BoardRow key={row.strategy_id} row={row} extra={extra} />
        ))}
      </div>
    </section>
  )
}

function BoardRow({ row, extra }: { row: StrategyBoardRow; extra: SortOption | undefined }) {
  const gate = row.latest_gate
  return (
    <Link
      to={`/s/${encodeURIComponent(row.name)}`}
      className="board-row"
      data-testid="board-row"
      data-strategy={row.name}
      data-run={gate === null ? '' : String(gate.run_id)}
    >
      <div className="board-row-head">
        <span className="row-link board-name" data-testid="board-row-name">
          {row.name}
        </span>
        <StageChip stage={row.stage} />
      </div>
      {gate === null ? (
        <div className="dim-note">no usable gate evidence</div>
      ) : (
        <div className="board-rail">
          <span className="board-metric">
            <span className="board-k">oos</span>
            {num(finite(gate.sharpe_oos))}
            <span className="board-n"> n={gate.n_obs_oos ?? '—'}</span>
          </span>
          <span className="board-metric">
            <span className="board-k">wf</span>
            {num(finite(gate.mean_window_sharpe))}
          </span>
          {extra !== undefined && (
            <span className="board-metric">
              <span className="board-k">{extra.label}</span>
              {extra.format(finite(gate[extra.key]))}
            </span>
          )}
          <Sparkline
            values={[finite(gate.min_window_sharpe), finite(gate.mean_window_sharpe), finite(gate.sharpe_oos)]}
            label={SPARKLINE_LABEL}
          />
          <PassMark passed={gate.passed} />
        </div>
      )}
    </Link>
  )
}
