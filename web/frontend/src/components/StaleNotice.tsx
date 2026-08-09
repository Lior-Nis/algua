import { formatTimestamp } from '../api'

/** The envelope fields staleness rendering needs — structural, so composed responses
 * (e.g. /api/ideas) qualify without carrying the full ApiEnvelope shape. */
interface StaleFields {
  stale: boolean
  fetched_at: string
  last_error_code?: string | null
}

/** Gold stale-data banner — staleness must read as an alert on EVERY screen, never
 * silently render cached data as fresh (same contract as Home's fleet banner). */
export default function StaleNotice({ env }: { env: StaleFields | undefined }) {
  if (!env?.stale) return null
  return (
    <section className="banner-stale">
      <span className="micro-label">stale data</span>
      <div>
        showing cached data from {formatTimestamp(env.fetched_at)}
        {env.last_error_code != null && <>, last error {env.last_error_code}</>}
      </div>
    </section>
  )
}
