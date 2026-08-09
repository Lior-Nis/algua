import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import HealthBadge from '../components/HealthBadge'
import type { ApiEnvelope, FleetHealth, ListPayload, StrategyRecord } from '../types'

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

export default function Funnel() {
  const strategies = useFetch<ApiEnvelope<ListPayload<StrategyRecord>>>('/api/strategies')
  // Reuses Home's /api/fleet cache entry (same URL + ttl).
  const fleet = useFetch<ApiEnvelope<FleetHealth>>('/api/fleet', { ttlMs: 10_000 })
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

  if (stages.length === 0) {
    return (
      <section className="panel all-clear">
        <span className="micro-label">funnel empty</span>
        no strategies registered
      </section>
    )
  }

  return (
    <>
      {stages.map((stage) => {
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
                      <HealthBadge health={healthByName.get(rec.name)!} />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </details>
        )
      })}
    </>
  )
}
