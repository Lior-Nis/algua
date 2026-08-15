import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import HealthBadge, { healthColor } from '../components/HealthBadge'
import MetricTile from '../components/MetricTile'
import StageChip from '../components/StageChip'
import StaleNotice from '../components/StaleNotice'
import { pct } from '../format'
import type { ApiEnvelope, FleetHealth, FleetRow } from '../types'

/** Worst → best; orders the per-health summary tiles. */
const HEALTH_ORDER = ['halted', 'drift', 'stale', 'idle', 'ok']

/**
 * Fleet-wide health histogram, counted from `rows` (EVERY strategy).
 *
 * NOT `summary.by_health` — `fleet health` builds that over the ALERTING rows only
 * (algua/cli/fleet_cmd.py), so rendering it beside `summary.total` silently dropped every
 * non-alerting strategy and could never show an `ok` tile at all.
 */
function healthHistogram(rows: FleetRow[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const row of rows) counts[row.health] = (counts[row.health] ?? 0) + 1
  return counts
}

export default function Fleet() {
  const { data, error, loading, refetch } = useFetch<ApiEnvelope<FleetHealth>>('/api/fleet', {
    ttlMs: 10_000,
  })
  const setFetchedAt = useSetFetchedAt()

  const fetchedAt = data?.fetched_at ?? null
  useEffect(() => {
    setFetchedAt(fetchedAt)
    return () => setFetchedAt(null)
  }, [fetchedAt, setFetchedAt])

  if (data === undefined) {
    if (loading) return <FleetSkeleton />
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

  const fleet = data.data
  const byHealth = healthHistogram(fleet.rows ?? [])
  const healthKeys = [
    ...HEALTH_ORDER.filter((h) => h in byHealth),
    ...Object.keys(byHealth).filter((h) => !HEALTH_ORDER.includes(h)),
  ]

  return (
    <>
      <StaleNotice env={data} />

      {fleet.global_halt && <section className="banner-halt">global halt active</section>}

      <section className="tile-row">
        <MetricTile label="total" value={fleet.summary.total} />
        <MetricTile
          label="alerting"
          value={fleet.summary.alerting}
          color={fleet.summary.alerting > 0 ? 'var(--red)' : 'var(--text-dim)'}
        />
        {healthKeys.map((h) => (
          <MetricTile key={h} label={h} value={byHealth[h]} color={healthColor(h)} />
        ))}
      </section>

      {fleet.alerting.length > 0 ? (
        <section>
          <div className="micro-label" style={{ marginBottom: 6 }}>
            alerting — worst first
          </div>
          <div className="alert-list">
            {fleet.alerting.map((row) => (
              <AlertRow key={row.strategy} row={row} />
            ))}
          </div>
        </section>
      ) : (
        <section className="panel all-clear">
          <span className="micro-label">all clear</span>
          {/* NOT "{total} strategies ok": `ok` is a specific verdict (fresh, parseable tick
              evidence) and most of the fleet — every idea/backtested/retired strategy — is
              `idle` by design and simply not watched. */}
          no alerting strategies · {byHealth.ok ?? 0} of {fleet.summary.total} ok
        </section>
      )}

      <section>
        <div className="micro-label" style={{ marginBottom: 6 }}>
          every strategy
        </div>
        <div className="row-list">
          {[...(fleet.rows ?? [])].map((row) => (
            <Link
              key={row.strategy}
              to={`/s/${encodeURIComponent(row.strategy)}`}
              className="funnel-row"
            >
              <div className="funnel-row-main">
                <span className="row-link">{row.strategy}</span>
              </div>
              <div className="funnel-row-chips">
                <StageChip stage={row.stage} />
                <HealthBadge health={row.health} />
              </div>
            </Link>
          ))}
        </div>
      </section>
    </>
  )
}

function AlertRow({ row }: { row: FleetRow }) {
  return (
    <div className="alert-row">
      <div className="alert-row-main">
        <span className="alert-row-name">{row.strategy}</span>
        <span className="alert-row-chips">
          <StageChip stage={row.stage} />
          <HealthBadge health={row.health} />
        </span>
      </div>
      <div className="alert-row-metrics">
        <span>
          {row.staleness_sessions !== null ? `${row.staleness_sessions} sess stale` : '— sess'}
        </span>
        <span>dd {pct(row.drawdown?.drawdown)}</span>
      </div>
    </div>
  )
}

function FleetSkeleton() {
  return (
    <>
      <div className="tile-row" aria-hidden="true">
        <div className="skeleton-block" style={{ height: 64 }} />
        <div className="skeleton-block" style={{ height: 64 }} />
        <div className="skeleton-block" style={{ height: 64 }} />
      </div>
      <div className="skeleton-block" style={{ height: 180 }} aria-hidden="true" />
    </>
  )
}
