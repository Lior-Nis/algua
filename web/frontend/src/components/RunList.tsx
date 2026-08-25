import { useState } from 'react'
import { Link } from 'react-router-dom'
import { runsUrl, useFetch } from '../api'
import { num } from '../format'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
import Sparkline from './Sparkline'

// A bounded leaderboard page, not a ledger dump — the ranked list answers "who's winning",
// not "show me everything".
const LIMIT = 20

interface SortOption {
  /** Must be a `METRIC_COLUMNS` name (algua/registry/store/runs.py) — the store is the single
   * semantic gate for `sort`; this list is a curated display SUBSET of that vocabulary, not a
   * second copy of it. */
  key: string
  label: string
}

// Curated to metrics where "higher ranks first" reads naturally (no drawdown/vol columns,
// which are "lower is better" and would read backwards under a DESC-only ranking).
export const SORT_OPTIONS: SortOption[] = [
  { key: 'sharpe_oos', label: 'oos sharpe' },
  { key: 'mean_window_sharpe', label: 'is sharpe' },
  { key: 'sortino_oos', label: 'oos sortino' },
]

export const DEFAULT_SORT = SORT_OPTIONS[0].key

/** `RunRow`'s index signature types every column outside the explicit few as `unknown` (see its
 * doc comment) — `min_window_sharpe` is one of those, so a direct property read needs this
 * narrowing the same way the explicitly-typed `mean_window_sharpe`/`sharpe_oos` fields don't. */
function asNullableNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function metricValue(run: RunRow, sort: string): number | null {
  return asNullableNumber(run[sort])
}

function PassMark({ passed }: { passed: number | boolean | null | undefined }) {
  if (passed !== true && passed !== 1 && passed !== false && passed !== 0) {
    return (
      <span className="pass-mark" style={{ color: 'var(--text-dim)' }}>
        unknown
      </span>
    )
  }
  const ok = passed === true || passed === 1
  return (
    <span className="pass-mark" style={{ color: ok ? 'var(--green)' : 'var(--red)' }}>
      {ok ? 'pass' : 'fail'}
    </span>
  )
}

/**
 * View 1 — the ranked run list (spec §6.1). One sort metric at a time via a chip row (never
 * column headers — a multi-column sortable table is the worst affordance on a 390px screen).
 *
 * Sort is entirely SERVER-side: switching a chip re-requests `/api/runs` with a new `sort`
 * param, and rows render in EXACTLY the order the response returns. This is deliberate, not an
 * omission — `list_runs` already does `ORDER BY {sort} IS NULL, {sort} DESC` so a NULL
 * sort-metric row sorts last, never outranking a real number; a client-side re-sort (even one
 * that means well) is exactly how that guarantee gets silently defeated, e.g. by coalescing a
 * NULL to 0 and having it float above a genuinely negative Sharpe.
 *
 * `kind=gate` only: a pass/fail verdict and an OOS metric are gate-run concepts (per
 * `scripts/seed_runs_dev.py`, only gate rows set `passed` and carry `sharpe_oos`) — the same
 * filter `ScatterISOOS` uses for the same reason.
 *
 * The sparkline draws `[min_window_sharpe, mean_window_sharpe, sharpe_oos]` — the walk-forward
 * worst window, the walk-forward mean, and the realized holdout result — three scalars already
 * on the row (no extra `/api/runs/series` round trip per row, and no per-bar vector: the holdout
 * return series is deliberately never exposed, see `run_views.py`'s `_series_entry` docstring).
 */
export default function RunList() {
  const [sort, setSort] = useState<string>(DEFAULT_SORT)
  const { data } = useFetch<ApiEnvelope<RunsListPayload>>(
    runsUrl({ kind: 'gate', sort, limit: LIMIT }),
  )
  const runs = data?.data.runs ?? []
  // Same convention as ScatterISOOS: "no data yet" and "still loading" render identically (the
  // honest empty state), rather than flashing chip/row chrome around an empty array before the
  // fetch resolves — real rows replace it the moment they arrive.
  const isEmpty = runs.length === 0

  return (
    <section>
      <div className="micro-label" style={{ marginBottom: 6 }}>
        ranked runs
      </div>
      {isEmpty ? (
        <section className="panel all-clear">
          <span className="micro-label">no runs</span>
          no runs recorded yet — the ledger fills when the operator loop runs
        </section>
      ) : (
        <>
          <div className="chip-row" style={{ marginBottom: 8 }}>
            {SORT_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                type="button"
                className={opt.key === sort ? 'filter-chip active' : 'filter-chip'}
                onClick={() => setSort(opt.key)}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <div className="row-list">
            {runs.map((run) => (
              <Link
                key={run.id}
                to={`/s/${encodeURIComponent(run.strategy_name)}`}
                className="funnel-row"
                data-testid="run-row"
                data-strategy={run.strategy_name}
              >
                <div className="funnel-row-main">
                  <span className="row-link" data-testid="run-row-name">
                    {run.strategy_name}
                  </span>
                </div>
                <div className="funnel-row-chips">
                  <span className="run-row-metric">{num(metricValue(run, sort))}</span>
                  <Sparkline
                    values={[
                      asNullableNumber(run.min_window_sharpe),
                      run.mean_window_sharpe,
                      run.sharpe_oos,
                    ]}
                  />
                  <PassMark passed={run.passed} />
                </div>
              </Link>
            ))}
          </div>
        </>
      )}
    </section>
  )
}
