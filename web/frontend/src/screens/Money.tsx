import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import MetricTile from '../components/MetricTile'
import StageChip from '../components/StageChip'
import StaleNotice from '../components/StaleNotice'
import { num, pct, utcDate } from '../format'
import type { ApiEnvelope, BookPayload, BookSlice } from '../types'

/** Slice P&L against its allocated capital. Null when the strategy has never ticked — an
 * un-ticked slice has no return, and rendering 0.0% would claim it broke even. */
function sliceReturn(row: BookSlice): number | null {
  if (row.last_equity === null || !Number.isFinite(row.last_equity)) return null
  if (!Number.isFinite(row.capital) || row.capital <= 0) return null
  return row.last_equity / row.capital - 1
}

export default function Money() {
  const { data, error, loading, refetch } = useFetch<ApiEnvelope<BookPayload>>('/api/book')
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

  const book = data.data
  const deployed = book.slices.reduce(
    (total, row) => total + (row.last_equity ?? row.capital),
    0,
  )

  return (
    <>
      <StaleNotice env={data} />

      <section className="tile-row">
        <MetricTile label="allocated" value={`${book.allocated}/${book.capacity}`} />
        <MetricTile label="committed" value={num(book.sum_allocations)} />
        <MetricTile label="marked at" value={num(deployed)} />
        {book.live_allocated > 0 && (
          <MetricTile label="live" value={book.live_allocated} color="var(--electric)" />
        )}
      </section>

      {/* Stated, not hidden: the account equity needs a broker call this view must not make, so
          capital headroom is genuinely unknown here rather than zero. */}
      <div className="dim-note">
        {book.count_headroom} more tenants fit. Capital headroom needs the account equity, which
        this read-only view never calls the broker for.
      </div>

      {book.unallocated_operational.length > 0 && (
        <section className="panel needs-capital">
          <span className="micro-label">holding no capital</span>
          <div className="dim-note" style={{ marginBottom: 8 }}>
            These are in an operational stage with no slice, so the operator loop skips them
            entirely. Allocating is a capital decision — terminal only.
          </div>
          <div className="row-list">
            {book.unallocated_operational.map((row) => (
              <Link
                key={row.strategy}
                to={`/s/${encodeURIComponent(row.strategy)}`}
                className="funnel-row"
              >
                <div className="funnel-row-main">
                  <span className="row-link">{row.strategy}</span>
                  <span className="dim-note">
                    {row.ever_ticked ? 'lost its slice' : 'never ticked'}
                    {row.since !== null && ` · since ${utcDate(row.since)}`}
                  </span>
                </div>
                <div className="funnel-row-chips">
                  <StageChip stage={row.stage} />
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}

      <section>
        <div className="micro-label" style={{ marginBottom: 6 }}>
          book slices
        </div>
        {book.slices.length === 0 ? (
          <section className="panel all-clear">
            <span className="micro-label">book empty</span>
            no capital is allocated
          </section>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>strategy</th>
                  <th>stage</th>
                  <th className="num">slice</th>
                  <th className="num">marked</th>
                  <th className="num">p&amp;l</th>
                </tr>
              </thead>
              <tbody>
                {book.slices.map((row) => {
                  const ret = sliceReturn(row)
                  return (
                    <tr key={row.strategy}>
                      <td>
                        <Link
                          to={`/s/${encodeURIComponent(row.strategy)}`}
                          className="row-link"
                        >
                          {row.strategy}
                        </Link>
                      </td>
                      <td className="dim-note">{row.stage}</td>
                      <td className="num">{num(row.capital)}</td>
                      <td className="num">{num(row.last_equity)}</td>
                      <td
                        className="num"
                        style={
                          ret === null || ret === 0
                            ? undefined
                            : { color: ret > 0 ? 'var(--green)' : 'var(--red)' }
                        }
                      >
                        {ret === null ? '—' : pct(ret)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  )
}
