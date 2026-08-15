import { Fragment, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import ActorChip from '../components/ActorChip'
import StaleNotice from '../components/StaleNotice'
import { utcDate, utcDateTime, utcTime } from '../format'
import { alertsState, disableAlerts, enableAlerts, type AlertsState } from '../push'
import type { ActivityRow, ApiEnvelope, ListPayload, TriageItem, TriagePayload } from '../types'

/** Kind -> accent. Only conditions that STOP something get an alarm colour; a wedged queue is
 * work piling up with nothing lost yet, so it stays amber rather than red. */
const KIND_COLOR: Record<TriageItem['kind'], string> = {
  loop_down: 'var(--red)',
  global_halt: 'var(--red)',
  capital_stranded: 'var(--amber)',
  strategy: 'var(--amber)',
  queue_wedged: 'var(--text-dim)',
}

const ACTIVITY_LIMIT = 25

export default function Now() {
  const triage = useFetch<TriagePayload>('/api/triage', { ttlMs: 10_000 })
  const activity = useFetch<ApiEnvelope<ListPayload<ActivityRow>>>(
    `/api/activity?limit=${ACTIVITY_LIMIT}`,
  )
  const setFetchedAt = useSetFetchedAt()

  const fetchedAt = triage.data?.fetched_at ?? null
  useEffect(() => {
    setFetchedAt(fetchedAt)
    return () => setFetchedAt(null)
  }, [fetchedAt, setFetchedAt])

  if (triage.data === undefined) {
    if (triage.loading) return <div className="skeleton-block" style={{ height: 280 }} aria-hidden="true" />
    const error = triage.error
    return (
      <section className="panel error-panel">
        <span className="micro-label">fetch failed</span>
        <div className="error-code">
          {error instanceof ApiError && error.code !== null
            ? error.code
            : (error?.message ?? 'unknown error')}
        </div>
        <button type="button" className="retry-btn" onClick={triage.refetch}>
          retry
        </button>
      </section>
    )
  }

  const { items, headline, sources } = triage.data
  const degraded = Object.entries(sources).filter(([, ok]) => !ok).map(([name]) => name)

  return (
    <>
      <StaleNotice env={triage.data} />

      {/* A source that failed to load MUST be named. Silently ranking a shorter list would
          present a partial read as an all-clear, which is the exact failure this screen exists
          to prevent. */}
      {degraded.length > 0 && (
        <section className="banner-stale">
          <span className="micro-label">partial read</span>
          <div>{degraded.join(', ')} unavailable — this list may be incomplete</div>
        </section>
      )}

      <section className="needs-you">
        <div className="needs-you-head">
          <span className="micro-label">needs you</span>
          <span className={items.length > 0 ? 'needs-you-count active' : 'needs-you-count'}>
            {items.length}
          </span>
        </div>
        {items.length === 0 ? (
          // One line, not an empty panel: the all-clear case is the common case and must cost
          // nothing to read.
          <div className="all-clear-line">nothing needs you</div>
        ) : (
          <div className="row-list">
            {items.map((item, i) => (
              <TriageRow key={`${item.kind}-${item.title}-${i}`} item={item} />
            ))}
          </div>
        )}
      </section>

      <section className="tile-row">
        <HeadlineTile
          label="fleet ok"
          value={
            headline.fleet_total === null
              ? '—'
              : `${headline.fleet_ok}/${headline.fleet_total}`
          }
          to="/fleet"
        />
        <HeadlineTile
          label="book"
          value={
            headline.book_capacity === null
              ? '—'
              : `${headline.book_allocated}/${headline.book_capacity}`
          }
          to="/money"
        />
        <HeadlineTile
          label="loops down"
          value={headline.loops_alerting}
          to="/research"
          alarm={headline.loops_alerting > 0}
        />
      </section>

      <ActivityFeed query={activity} />
      <AlertsPanel />
    </>
  )
}

function TriageRow({ item }: { item: TriageItem }) {
  return (
    <Link to={item.route} className="triage-row">
      <span className="triage-rule" style={{ background: KIND_COLOR[item.kind] }} aria-hidden="true" />
      <div className="triage-body">
        <div className="triage-title">{item.title}</div>
        {item.detail !== null && item.detail !== '' && (
          <div className="dim-note">{item.detail}</div>
        )}
        {item.since !== null && <div className="dim-note num">since {utcDateTime(item.since)}</div>}
      </div>
    </Link>
  )
}

function HeadlineTile({
  label,
  value,
  to,
  alarm = false,
}: {
  label: string
  value: React.ReactNode
  to: string
  alarm?: boolean
}) {
  return (
    <Link to={to} className="metric-tile headline-tile">
      <div className="metric-value" style={alarm ? { color: 'var(--red)' } : undefined}>
        {value}
      </div>
      <div className="metric-label">{label}</div>
    </Link>
  )
}

/** The audit trail, absorbed from the old Activity tab. "What did the machine do while I was
 * asleep" is a Now question, not a separate destination. */
function ActivityFeed({
  query,
}: {
  query: ReturnType<typeof useFetch<ApiEnvelope<ListPayload<ActivityRow>>>>
}) {
  if (query.data === undefined) {
    if (query.loading) {
      return <div className="skeleton-block" style={{ height: 140 }} aria-hidden="true" />
    }
    return (
      <section className="panel">
        <span className="micro-label">recent activity</span>
        <div className="dim-note">activity unavailable</div>
      </section>
    )
  }
  const rows = query.data.data.data
  if (rows.length === 0) {
    return (
      <section className="panel">
        <span className="micro-label">recent activity</span>
        <div className="dim-note">nothing recorded yet</div>
      </section>
    )
  }
  let prevDate: string | null = null
  return (
    <section className="panel">
      <span className="micro-label">recent activity</span>
      <div className="alert-list">
        {rows.map((row, i) => {
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
                    <Link to={`/s/${encodeURIComponent(row.strategy)}`} className="row-link">
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
      </div>
    </section>
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
      // On failure, re-derive the truth from the browser instead of guessing.
      .then(setState)
      .catch(() => alertsState().then(setState))
      .finally(() => setBusy(false))
  }

  return (
    <section className="panel">
      <span className="micro-label">push alerts</span>
      {state === 'off' && (
        <button type="button" className="retry-btn" disabled={busy} onClick={() => run(enableAlerts)}>
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
