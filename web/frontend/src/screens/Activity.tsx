import { Fragment, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import { utcDate, utcTime } from '../format'
import type { ActivityRow, ApiEnvelope, ListPayload } from '../types'

const ACTORS = ['agent', 'human', 'system'] as const
type Actor = (typeof ACTORS)[number]

const ACTOR_COLORS: Record<string, string> = {
  agent: 'var(--cyan)',
  human: 'var(--gold)',
  system: 'var(--text-dim)',
}

export function ActorChip({ actor }: { actor: string }) {
  return (
    <span className="stage-chip" style={{ color: ACTOR_COLORS[actor] ?? 'var(--text-dim)' }}>
      {actor}
    </span>
  )
}

export default function Activity() {
  const [actor, setActor] = useState<Actor | null>(null)
  const [limit, setLimit] = useState(100)
  const [strategyFilter, setStrategyFilter] = useState('')

  const url = `/api/activity?limit=${limit}${actor !== null ? `&actor=${actor}` : ''}`
  const { data, error, loading, refetch } = useFetch<ApiEnvelope<ListPayload<ActivityRow>>>(url)
  const setFetchedAt = useSetFetchedAt()

  const fetchedAt = data?.fetched_at ?? null
  useEffect(() => {
    setFetchedAt(fetchedAt)
    return () => setFetchedAt(null)
  }, [fetchedAt, setFetchedAt])

  const filters = (
    <section className="filter-bar">
      <div className="chip-row">
        {ACTORS.map((a) => (
          <button
            key={a}
            type="button"
            className={a === actor ? 'filter-chip active' : 'filter-chip'}
            onClick={() => setActor(actor === a ? null : a)}
          >
            {a}
          </button>
        ))}
      </div>
      <input
        type="search"
        className="text-filter"
        placeholder="filter by strategy"
        value={strategyFilter}
        onChange={(e) => setStrategyFilter(e.target.value)}
      />
    </section>
  )

  if (data === undefined) {
    if (loading) {
      return (
        <>
          {filters}
          <div className="skeleton-block" style={{ height: 260 }} aria-hidden="true" />
        </>
      )
    }
    return (
      <>
        {filters}
        <section className="panel error-panel">
          <span className="micro-label">fetch failed</span>
          <div className="error-code">
            {error instanceof ApiError && error.code !== null
              ? error.code
              : (error?.message ?? 'unknown error')}
          </div>
          <button type="button" className="retry-btn" onClick={refetch}>
            retry
          </button>
        </section>
      </>
    )
  }

  const rows = data.data.data
  const needle = strategyFilter.trim().toLowerCase()
  const visible = needle
    ? rows.filter((r) => (r.strategy ?? '').toLowerCase().includes(needle))
    : rows

  let prevDate: string | null = null

  return (
    <>
      {filters}
      {visible.length === 0 ? (
        <section className="panel all-clear">
          <span className="micro-label">no activity</span>
          nothing matches
        </section>
      ) : (
        <section className="alert-list">
          {visible.map((row, i) => {
            const date = utcDate(row.ts)
            const showDate = date !== prevDate
            prevDate = date
            return (
              <Fragment key={row.id ?? `${row.ts}-${i}`}>
                {showDate && <div className="date-divider num">{date}</div>}
                <div className="activity-row">
                  <div className="activity-row-head">
                    <span className="num dim-note">{utcTime(row.ts)}</span>
                    <ActorChip actor={row.actor} />
                    <span className="activity-action">{row.action}</span>
                    {row.strategy != null && row.strategy !== '' && (
                      <Link
                        to={`/s/${encodeURIComponent(row.strategy)}`}
                        className="row-link"
                      >
                        {row.strategy}
                      </Link>
                    )}
                  </div>
                  {row.reason != null && row.reason !== '' && (
                    <div className="dim-note activity-reason">{row.reason}</div>
                  )}
                </div>
              </Fragment>
            )
          })}
        </section>
      )}
      {rows.length >= limit && (
        <button type="button" className="load-more-btn" onClick={() => setLimit(limit * 2)}>
          load more
        </button>
      )}
    </>
  )
}
