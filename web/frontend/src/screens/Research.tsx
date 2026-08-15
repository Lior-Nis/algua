import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import HealthBadge from '../components/HealthBadge'
import MetricTile from '../components/MetricTile'
import StaleNotice from '../components/StaleNotice'
import { utcDate, utcDateTime } from '../format'
import type {
  ApiEnvelope,
  FleetHealth,
  IdeaRow,
  IdeasResponse,
  ListPayload,
  OpsPayload,
  StrategyRecord,
} from '../types'

/** Lifecycle order (algua/contracts/lifecycle.py); unknown stages append at the end. */
const STAGE_ORDER = [
  'idea',
  'backtested',
  'candidate',
  'paper',
  'forward_tested',
  'live',
  'dormant',
  'retired',
]

/** Used only if `fleet health` did not ship its authoritative `operational_stages`. */
const OPERATIONAL_STAGES_FALLBACK = ['live', 'paper', 'forward_tested']

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

export default function Research() {
  const strategies = useFetch<ApiEnvelope<ListPayload<StrategyRecord>>>('/api/strategies')
  // Reuses the Now/Fleet cache entries (same URL + ttl).
  const fleet = useFetch<ApiEnvelope<FleetHealth>>('/api/fleet', { ttlMs: 10_000 })
  const ops = useFetch<ApiEnvelope<OpsPayload>>('/api/ops', { ttlMs: 30_000 })
  const ideas = useFetch<IdeasResponse>('/api/ideas')
  const setFetchedAt = useSetFetchedAt()

  const fetchedAt = strategies.data?.fetched_at ?? null
  useEffect(() => {
    setFetchedAt(fetchedAt)
    return () => setFetchedAt(null)
  }, [fetchedAt, setFetchedAt])

  if (strategies.data === undefined) {
    if (strategies.loading) {
      return <div className="skeleton-block" style={{ height: 260 }} aria-hidden="true" />
    }
    const error = strategies.error
    return (
      <section className="panel error-panel">
        <span className="micro-label">fetch failed</span>
        <div className="error-code">
          {error instanceof ApiError && error.code !== null
            ? error.code
            : (error?.message ?? 'unknown error')}
        </div>
        <button type="button" className="retry-btn" onClick={strategies.refetch}>
          retry
        </button>
      </section>
    )
  }

  const records = strategies.data.data.data
  const healthByName = new Map<string, string>()
  for (const row of fleet.data?.data.rows ?? []) healthByName.set(row.strategy, row.health)
  // Health is only an ALERT on the stages an operator loop ticks; elsewhere it is context,
  // so the badge must not carry alarm colour (see HealthBadge's `muted`).
  const operational = new Set(
    fleet.data?.data.operational_stages ?? OPERATIONAL_STAGES_FALLBACK,
  )

  const byStage = new Map<string, StrategyRecord[]>()
  for (const rec of records) {
    const group = byStage.get(rec.stage)
    if (group) group.push(rec)
    else byStage.set(rec.stage, [rec])
  }
  const stages = [
    ...STAGE_ORDER.filter((s) => byStage.has(s)),
    ...[...byStage.keys()].filter((s) => !STAGE_ORDER.includes(s)),
  ]

  return (
    <>
      <StaleNotice env={strategies.data} />

      <ResearchLoop ops={ops.data?.data} />

      <section>
        <div className="micro-label" style={{ marginBottom: 6 }}>
          funnel
        </div>
        {stages.length === 0 ? (
          <section className="panel all-clear">
            <span className="micro-label">funnel empty</span>
            no strategies registered
          </section>
        ) : (
          stages.map((stage) => {
            const group = byStage.get(stage)!
            return (
              <details key={stage} className="stage-section" open>
                <summary className="stage-summary">
                  <span className="micro-label">
                    {stage} ({group.length})
                  </span>
                </summary>
                <div className="row-list">
                  {group.map((rec) => (
                    <div key={rec.name} className="funnel-row">
                      <div className="funnel-row-main">
                        <Link to={`/s/${encodeURIComponent(rec.name)}`} className="row-link">
                          {rec.name}
                        </Link>
                        {rec.family !== null && <span className="dim-note">{rec.family}</span>}
                      </div>
                      <div className="funnel-row-chips">
                        <span className="stage-chip">{rec.hypothesis_status}</span>
                        {healthByName.has(rec.name) && (
                          <HealthBadge
                            health={healthByName.get(rec.name)!}
                            muted={!operational.has(rec.stage)}
                          />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )
          })
        )}
      </section>

      <IdeaPool query={ideas} />
    </>
  )
}

/** The top of the funnel. A stopped research loop is the difference between "no ideas today"
 * and "nothing will ever arrive again", and only this panel can tell them apart. */
function ResearchLoop({ ops }: { ops: OpsPayload | undefined }) {
  if (ops === undefined) return null
  const research = ops.loops.research
  const mergeback = ops.loops.mergeback
  if (research === undefined) return null
  const down = research.health !== 'ok' && research.health !== 'idle'
  return (
    <section className={down ? 'panel loop-panel down' : 'panel loop-panel'}>
      <span className="micro-label">research loop</span>
      <div className="loop-line">
        <HealthBadge health={research.health} />
        {research.detail != null && <span className="dim-note">{research.detail}</span>}
      </div>
      {research.last_ok_at != null && (
        <div className="dim-note num">last clean run {utcDateTime(research.last_ok_at)}</div>
      )}
      {typeof research.consecutive_failures === 'number' && research.consecutive_failures > 0 && (
        <div className="dim-note">{research.consecutive_failures} consecutive failed runs</div>
      )}
      {mergeback !== undefined && typeof mergeback.queue_depth === 'number' && (
        <div className="dim-note">
          merge-back queue {mergeback.queue_depth}
          {mergeback.health === 'stale' && ' — wedged'}
        </div>
      )}
    </section>
  )
}

function IdeaPool({ query }: { query: ReturnType<typeof useFetch<IdeasResponse>> }) {
  const [statusFilter, setStatusFilter] = useState<string | null>(null)

  if (query.data === undefined) {
    if (query.loading) {
      return <div className="skeleton-block" style={{ height: 160 }} aria-hidden="true" />
    }
    return (
      <section className="panel">
        <span className="micro-label">idea pool</span>
        <div className="dim-note">ideas unavailable</div>
      </section>
    )
  }

  const data = query.data
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
  // The windowed counts stay visible — they are the funnel-breadth signal — but on their own
  // line, so they can never be read as part of the all-time histogram.
  const windowed = statuses
    .filter((s) => s in counts && counts[s] > 0)
    .map((s) => `${s} ${counts[s]}`)
  const visible = statusFilter === null ? ideas : ideas.filter((i) => i.status === statusFilter)

  return (
    <section>
      <div className="micro-label" style={{ marginBottom: 6 }}>
        idea pool
      </div>
      {/* The tiles are ALL-TIME, counted off the same list rendered below and filtered by the
          same chips. `idea stats` is a DIFFERENT denominator (trailing window), so mixing it
          into this row made a histogram that could not add up. */}
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

      {statuses.length > 0 && (
        <div className="chip-row" style={{ marginTop: 8 }}>
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
        <div className="alert-list" style={{ marginTop: 8 }}>
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
        </div>
      )}
    </section>
  )
}
