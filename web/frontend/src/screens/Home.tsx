import { useEffect, useState } from 'react'
import { ApiError, formatTimestamp, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import HealthBadge, { healthColor } from '../components/HealthBadge'
import MetricTile from '../components/MetricTile'
import StageChip from '../components/StageChip'
import { alertsState, disableAlerts, enableAlerts, type AlertsState } from '../push'
import type { ApiEnvelope, FleetHealth, FleetRow } from '../types'

/** Worst → best; orders the per-health summary tiles. */
const HEALTH_ORDER = ['halted', 'drift', 'stale', 'idle', 'ok']

function pct(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  return `${(x * 100).toFixed(1)}%`
}

export default function Home() {
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
    if (loading) return <HomeSkeleton />
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
  const byHealth = fleet.summary.by_health
  const healthKeys = [
    ...HEALTH_ORDER.filter((h) => h in byHealth),
    ...Object.keys(byHealth).filter((h) => !HEALTH_ORDER.includes(h)),
  ]

  return (
    <>
      {data.stale && (
        <section className="banner-stale">
          <span className="micro-label">stale data</span>
          <div>
            showing cached data from {formatTimestamp(data.fetched_at)}
            {data.last_error_code != null && <>, last error {data.last_error_code}</>}
          </div>
        </section>
      )}

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
          {fleet.summary.total} strategies ok
        </section>
      )}

      <AlertsPanel />
    </>
  )
}

function AlertsPanel() {
  const [state, setState] = useState<AlertsState | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let alive = true
    void alertsState().then((s) => {
      if (alive) setState(s)
    })
    return () => {
      alive = false
    }
  }, [])

  // Unknown yet or genuinely unsupported: no control at all.
  if (state === null || state === 'unsupported') return null

  const run = (action: () => Promise<AlertsState>) => {
    if (busy) return
    setBusy(true)
    action()
      .then(setState)
      // On failure, re-derive the truth from the browser instead of guessing.
      .catch(() => alertsState().then(setState))
      .finally(() => setBusy(false))
  }

  return (
    <section className="panel">
      <span className="micro-label">push alerts</span>
      {state === 'off' && (
        <button
          type="button"
          className="retry-btn"
          disabled={busy}
          onClick={() => run(enableAlerts)}
        >
          enable alerts
        </button>
      )}
      {state === 'on' && (
        <div>
          alerts on{' '}
          <button
            type="button"
            className="retry-btn"
            disabled={busy}
            onClick={() => run(disableAlerts)}
          >
            disable
          </button>
        </div>
      )}
      {state === 'denied' && <div>notifications blocked — enable in browser settings</div>}
      {state === 'needs-install' && <div>install to home screen to enable alerts</div>}
    </section>
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

function HomeSkeleton() {
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
