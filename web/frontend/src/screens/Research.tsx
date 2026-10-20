import { useEffect, useState } from 'react'
import { ApiError, runsUrl, useFetch } from '../api'
import type { UseFetchResult } from '../api'
import { useSetFetchedAt } from '../App'
import EvidenceDisclosure from '../components/EvidenceDisclosure'
import EvidenceStatus from '../components/EvidenceStatus'
import { ScatterISOOSView } from '../components/ScatterISOOS'
import StaleNotice from '../components/StaleNotice'
import { isBoardPayload, StrategyBoardView } from '../components/StrategyBoard'
import { utcDate, utcDateTime } from '../format'
import type {
  ApiEnvelope,
  IdeaRow,
  IdeaScorecardBucket,
  IdeasResponse,
  ListPayload,
  OpsPayload,
  RunsListPayload,
  StrategyBoardPayload,
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

/** Idea lifecycle order (algua/contracts/idea.py IdeaStatus); unknowns append. */
const STATUS_ORDER = ['open', 'needs_data', 'authored', 'refuted', 'discarded']

function ideaList(resp: IdeasResponse): IdeaRow[] {
  const ideas = resp.ideas
  if (Array.isArray(ideas)) return ideas
  return ideas?.data ?? []
}

/** `null` below the scorecard's `min_n_for_rates` floor — too few attempts to trust a rate. */
function formatRate(rate: number | null): string {
  return rate === null ? '--' : `${(rate * 100).toFixed(1)}%`
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

/**
 * Research — the run-ledger surface (spec Q7). Top to bottom: the research-loop liveness panel,
 * the funnel as a one-line count strip, the IS-vs-OOS scatter, the strategy board (one row per
 * registered strategy and its latest gate run), then the idea pool.
 */
export default function Research() {
  const strategies = useFetch<ApiEnvelope<ListPayload<StrategyRecord>>>('/api/strategies')
  const ops = useFetch<ApiEnvelope<OpsPayload>>('/api/ops', { ttlMs: 30_000 })
  const ideas = useFetch<IdeasResponse>('/api/ideas')
  const board = useFetch<ApiEnvelope<StrategyBoardPayload>>('/api/strategy-board')
  const gateRuns = useFetch<ApiEnvelope<RunsListPayload>>(runsUrl({ kind: 'gate' }))
  const setFetchedAt = useSetFetchedAt()

  const fetchedAt = strategies.data?.fetched_at ?? null
  useEffect(() => {
    setFetchedAt(fetchedAt)
    return () => setFetchedAt(null)
  }, [fetchedAt, setFetchedAt])

  const records = strategies.data?.data.data ?? []
  const byStage = new Map<string, number>()
  for (const rec of records) byStage.set(rec.stage, (byStage.get(rec.stage) ?? 0) + 1)
  const stages = [
    ...STAGE_ORDER.filter((s) => byStage.has(s)),
    ...[...byStage.keys()].filter((s) => !STAGE_ORDER.includes(s)),
  ]

  return (
    <main className="decision-page research-page">
      {strategies.data !== undefined && <StaleNotice env={strategies.data} />}

      <ResearchDecision
        strategies={strategies}
        ops={ops}
        ideas={ideas}
        board={board}
        gateRuns={gateRuns}
      />

      <div className="research-overview-grid">
        <section className="funnel-panel" aria-labelledby="funnel-heading">
          <h2 id="funnel-heading" className="micro-label">lifecycle inventory</h2>
        {strategies.data === undefined ? (
          <EvidenceStatus
            title="lifecycle inventory"
            state={strategies.loading ? 'loading' : 'failed'}
            detail={strategies.loading ? 'loading registry strategies' : (strategies.error?.message ?? 'registry unavailable')}
            onRetry={strategies.loading ? undefined : strategies.refetch}
          />
        ) : stages.length === 0 ? (
          strategies.data.stale ? (
            <EvidenceStatus
              title="lifecycle inventory"
              state="stale"
              detail="cached registry response contains no strategies"
              onRetry={strategies.refetch}
            />
          ) : (
            <EvidenceStatus title="lifecycle inventory" state="empty" detail="no strategies registered" />
          )
        ) : (
          <>
            {strategies.data.stale && (
              <EvidenceStatus title="lifecycle inventory" state="stale" detail="showing cached registry counts" onRetry={strategies.refetch} />
            )}
            <div className="funnel-inventory" aria-label="strategy inventory">
              {stages.map((stage) => {
                const count = byStage.get(stage) ?? 0
                const max = Math.max(...byStage.values())
                return (
                  <div key={stage} className="funnel-stage" aria-label={`${stage}: ${count} strategies`}>
                    <div className="funnel-stage-label">
                      <span>{stage}</span><span className="num">{count}</span>
                    </div>
                    <div className="funnel-track" aria-hidden="true">
                      <span style={{ width: `${(count / max) * 100}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}
        </section>

        <ScatterISOOSView query={gateRuns} />
      </div>

      <StrategyBoardView query={board} />

      <IdeaPool query={ideas} />
    </main>
  )
}

type SourceStatus = 'current' | 'loading' | 'stale' | 'failed'

interface DecisionSource {
  label: string
  status: SourceStatus
  detail?: string
}

function sourceStatus<T>(
  label: string,
  query: UseFetchResult<T>,
  stale: boolean,
  valid = true,
): DecisionSource {
  if (query.data === undefined) {
    return {
      label,
      status: query.loading ? 'loading' : 'failed',
      detail: query.error instanceof ApiError && query.error.code !== null
        ? query.error.code
        : (query.error?.message ?? 'unavailable'),
    }
  }
  if (!valid) return { label, status: 'failed', detail: 'unsupported data' }
  if (stale || query.error !== null) return { label, status: 'stale', detail: 'last read not current' }
  return { label, status: 'current' }
}

function ResearchDecision({
  strategies,
  ops,
  ideas,
  board,
  gateRuns,
}: {
  strategies: UseFetchResult<ApiEnvelope<ListPayload<StrategyRecord>>>
  ops: UseFetchResult<ApiEnvelope<OpsPayload>>
  ideas: UseFetchResult<IdeasResponse>
  board: UseFetchResult<ApiEnvelope<StrategyBoardPayload>>
  gateRuns: UseFetchResult<ApiEnvelope<RunsListPayload>>
}) {
  const sources = [
    sourceStatus('registry', strategies, strategies.data?.stale === true),
    sourceStatus('operations', ops, ops.data?.stale === true),
    sourceStatus('strategy board', board, board.data?.stale === true, isBoardPayload(board.data?.data)),
    sourceStatus('gate evidence', gateRuns, gateRuns.data?.stale === true),
    sourceStatus('ideas', ideas, ideas.data?.stale === true, ideas.data?.ok !== false),
  ]
  const research = ops.data?.data.loops.research
  const mergeback = ops.data?.data.loops.mergeback
  const loopBlocked = (ops.data !== undefined && research === undefined)
    || (research !== undefined && !['ok', 'idle'].includes(research.health))
  const mergebackBlocked = ops.data !== undefined
    && (mergeback === undefined || !['ok', 'idle'].includes(mergeback.health))
  const operationsStale = sources.some(
    (source) => source.label === 'operations' && source.status === 'stale',
  )
  const failed = sources.filter((source) => source.status === 'failed')
  const stale = sources.filter((source) => source.status === 'stale')
  const loading = sources.filter((source) => source.status === 'loading')

  let state = 'current'
  let consequence = 'Research flow is observable.'
  let reason = 'All required sources are current.'
  let checkpoint = 'Next gate or lifecycle change.'
  if (failed.length > 0) {
    state = 'blocked'
    consequence = 'Research decisions are blocked.'
    reason = `${failed.map((source) => source.label).join(', ')} unavailable.`
    checkpoint = `Restore ${failed[0].label} and refresh.`
  } else if (loopBlocked || mergebackBlocked) {
    state = 'blocked'
    consequence = 'New evidence may not arrive.'
    const causes: string[] = []
    if (operationsStale) causes.push('Operations evidence is stale; conclusions use cached loop state.')
    if (research === undefined) {
      causes.push('Research loop state is absent.')
    } else if (!['ok', 'idle'].includes(research.health)) {
      causes.push(`${operationsStale ? 'Cached research' : 'Research'} loop ${research.health}${research.detail ? `: ${research.detail}` : ''}.`)
    }
    if (mergeback === undefined) {
      causes.push('Merge-back loop state is absent.')
    } else if (!['ok', 'idle'].includes(mergeback.health)) {
      causes.push(mergeback.health === 'stale'
        ? `Merge-back queue ${mergeback.queue_depth ?? '—'} is wedged.`
        : `Merge-back loop ${mergeback.health}${mergeback.detail ? `: ${mergeback.detail}` : ''}.`)
    }
    reason = causes.join(' ')
    checkpoint = 'Next clean research loop run.'
  } else if (stale.length > 0) {
    state = 'partial'
    consequence = 'Current evidence is incomplete.'
    reason = `${stale.map((source) => source.label).join(', ')} stale.`
    checkpoint = 'Next successful source refresh.'
  } else if (loading.length > 0) {
    state = 'loading'
    consequence = 'No current decision yet.'
    reason = `Waiting for ${loading.map((source) => source.label).join(', ')}.`
    checkpoint = 'Required reads complete.'
  }

  return (
    <section className="decision-report" data-state={state} aria-labelledby="research-decision-heading">
      <div className="decision-report-head">
        <div>
          <h1 id="research-decision-heading" className="micro-label">research decision</h1>
          <div className="decision-state">{state}</div>
        </div>
        <div className="decision-consequence">{consequence}</div>
      </div>
      <div className="decision-facts">
        <div><span className="micro-label">reason</span><span>{reason}</span></div>
        <div><span className="micro-label">next checkpoint</span><span>{checkpoint}</span></div>
      </div>
      <div className="decision-sources" aria-label="decision sources">
        {sources.map((source) => (
          <span key={source.label} data-state={source.status} title={source.detail}>
            {source.label} <b>{source.status}</b>
          </span>
        ))}
      </div>
      {research !== undefined && state === 'blocked' && (
        <div className="decision-raw-line">
          <span>research loop</span>
          {typeof research.consecutive_failures === 'number' && research.consecutive_failures > 0 && (
            <span>{research.consecutive_failures} consecutive failed runs</span>
          )}
          {research.last_ok_at != null && <span className="num">last clean {utcDateTime(research.last_ok_at)}</span>}
        </div>
      )}
    </section>
  )
}

/** The idea pool used to be this screen's largest panel while holding ZERO rows on `main` — the
 * exact real estate the run-ledger views (spec Q7) needed. It now collapses to a bare count
 * until it actually has rows; the full tile/chip/list view returns automatically the moment
 * `ideas.length > 0`, unchanged from before. */
function IdeaPool({ query }: { query: ReturnType<typeof useFetch<IdeasResponse>> }) {
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [expandedIdea, setExpandedIdea] = useState<number | null>(null)

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

  if (ideas.length === 0) {
    return (
      <section className="panel">
        <span className="micro-label">idea pool</span>
        <div className="dim-note num">0 ideas</div>
      </section>
    )
  }

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
      {data.depth != null && (
        <div className="dim-note num">
          {data.depth.open_unclaimed} open · {data.depth.claimed} claimed · refill at{' '}
          {data.depth.refill_at} · ceiling {data.depth.ceiling}
        </div>
      )}
      {/* The tiles are ALL-TIME, counted off the same list rendered below and filtered by the
          same chips. `idea stats` is a DIFFERENT denominator (trailing window), so mixing it
          into this row made a histogram that could not add up. */}
      <div className="idea-counts num">
        <span>{ideas.length} total</span>
        {statuses.filter((s) => s in listCounts).map((s) => (
          <span key={s}>{s} {listCounts[s]}</span>
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
              aria-pressed={s === statusFilter}
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
              <button
                type="button"
                className="idea-row-head"
                aria-expanded={expandedIdea === idea.id}
                aria-controls={`idea-evidence-${idea.id}`}
                onClick={() => setExpandedIdea(expandedIdea === idea.id ? null : idea.id)}
              >
                <span className="idea-title">{idea.title}</span>
                <span className="stage-chip">{idea.status}</span>
                <span className="idea-date num">{utcDate(idea.created_at)}</span>
              </button>
              {expandedIdea === idea.id && (
                <div id={`idea-evidence-${idea.id}`} className="idea-evidence">
                  <div className="idea-hypothesis">{idea.hypothesis}</div>
                  <div className="dim-note num">
                    {idea.family !== null ? `${idea.family} · ` : ''}
                    {idea.source_type}{idea.source_ref !== null ? ` · ${idea.source_ref}` : ''}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {data.scorecard?.by_venue != null && Object.keys(data.scorecard.by_venue).length > 0 && (
        <EvidenceDisclosure
          title="yield by venue"
          summary={`${Object.keys(data.scorecard.by_venue).length} venues`}
        >
          <VenueYieldTable byVenue={data.scorecard.by_venue} />
        </EvidenceDisclosure>
      )}
    </section>
  )
}

/** The scorecard's feedback edge (spec §7): which inspiration venues actually produce survivors,
 * not just ideas. Hidden entirely with zero venues — an empty table would just be noise. */
function VenueYieldTable({ byVenue }: { byVenue: Record<string, IdeaScorecardBucket> | undefined }) {
  const entries = byVenue != null ? Object.entries(byVenue) : []
  if (entries.length === 0) return null
  return (
    <div>
      <table>
        <thead>
          <tr>
            <th>venue</th>
            <th>n</th>
            <th>integrity</th>
            <th>walk-forward</th>
            <th>survival</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([venue, bucket]) => (
            <tr key={venue}>
              <td>{venue}</td>
              <td className="num">{bucket.n}</td>
              <td className="num">{formatRate(bucket.integrity_yield)}</td>
              <td className="num">{formatRate(bucket.walkforward_yield)}</td>
              <td className="num">{formatRate(bucket.survival_yield)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
