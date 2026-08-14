import { useEffect, useState } from 'react'
import { ApiError, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import MetricTile from '../components/MetricTile'
import StaleNotice from '../components/StaleNotice'
import { utcDate } from '../format'
import type { IdeaRow, IdeasResponse } from '../types'

/** Idea lifecycle order (algua/contracts/idea.py IdeaStatus); unknowns append. */
const STATUS_ORDER = ['open', 'needs_data', 'authored', 'refuted', 'discarded']

function ideaList(resp: IdeasResponse): IdeaRow[] {
  const ideas = resp.ideas
  if (Array.isArray(ideas)) return ideas
  return ideas?.data ?? []
}

/** `idea stats` emits {window_days, counts:{status: n}}; tolerate a pre-flattened map too. */
function statusCounts(stats: IdeasResponse['stats']): Record<string, number> {
  if (stats === null || typeof stats !== 'object') return {}
  const source =
    typeof stats.counts === 'object' && stats.counts !== null
      ? (stats.counts as Record<string, unknown>)
      : stats
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(source)) {
    if (typeof v === 'number' && Number.isFinite(v)) out[k] = v
  }
  return out
}

export default function Ideas() {
  const { data, error, loading, refetch } = useFetch<IdeasResponse>('/api/ideas')
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const setFetchedAt = useSetFetchedAt()

  const fetchedAt = data?.fetched_at ?? null
  useEffect(() => {
    setFetchedAt(fetchedAt)
    return () => setFetchedAt(null)
  }, [fetchedAt, setFetchedAt])

  if (data === undefined) {
    if (loading) return <div className="skeleton-block" style={{ height: 260 }} aria-hidden="true" />
    return (
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
    )
  }

  const ideas = ideaList(data)
  const counts = statusCounts(data.stats)
  const listCounts: Record<string, number> = {}
  for (const idea of ideas) listCounts[idea.status] = (listCounts[idea.status] ?? 0) + 1
  // "total" is an aggregate in windowed_idea_counts(), not a status — never a filter chip.
  const statusesPresent = new Set(
    [...Object.keys(counts), ...ideas.map((i) => i.status)].filter((s) => s !== 'total'),
  )
  const statuses = [
    ...STATUS_ORDER.filter((s) => statusesPresent.has(s)),
    ...[...statusesPresent].filter((s) => !STATUS_ORDER.includes(s)),
  ]
  // The windowed counts stay visible — they are the funnel-breadth signal — but on
  // their own line, so they can never be read as part of the all-time histogram.
  const windowed = statuses
    .filter((s) => s in counts && counts[s] > 0)
    .map((s) => `${s} ${counts[s]}`)

  const visible = statusFilter === null ? ideas : ideas.filter((i) => i.status === statusFilter)

  return (
    <>
      <StaleNotice env={data} />
      <section>
        <div className="micro-label" style={{ marginBottom: 6 }}>
          idea pool
        </div>
        {/* The tiles are ALL-TIME, counted off the same list rendered below and filtered
            by the same chips. `idea stats` is a DIFFERENT denominator (trailing window),
            so mixing it into this row made a histogram that could not add up: "in pool"
            counted every idea ever while the status tiles counted only the last 90d. */}
        <div className="tile-row">
          <MetricTile label="all ideas" value={ideas.length} />
          {statuses
            .filter((s) => s in listCounts)
            .map((s) => (
              <MetricTile key={s} label={s} value={listCounts[s]} />
            ))}
        </div>
        {windowed.length > 0 && (
          <div className="dim-note num" style={{ marginTop: 6 }}>
            last {data.stats_window_days}d: {windowed.join(' · ')}
          </div>
        )}
      </section>

      {statuses.length > 0 && (
        <div className="chip-row">
          {statuses.map((s) => (
            <button
              key={s}
              type="button"
              className={s === statusFilter ? 'filter-chip active' : 'filter-chip'}
              onClick={() => setStatusFilter(statusFilter === s ? null : s)}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {visible.length === 0 ? (
        <section className="panel all-clear">
          <span className="micro-label">no ideas</span>
          nothing matches
        </section>
      ) : (
        <section className="alert-list">
          {visible.map((idea) => (
            <div key={idea.id} className="idea-row">
              <div className="idea-row-head">
                <span className="idea-title">{idea.title}</span>
                <span className="stage-chip">{idea.status}</span>
              </div>
              <div className="dim-note idea-hypothesis">{idea.hypothesis}</div>
              <div className="dim-note num">
                {idea.family !== null ? `${idea.family} · ` : ''}
                {utcDate(idea.created_at)}
              </div>
            </div>
          ))}
        </section>
      )}
    </>
  )
}
