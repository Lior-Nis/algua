import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { ApiError, formatTimestamp, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import EquityChart from '../components/EquityChart'
import HealthBadge, { healthColor } from '../components/HealthBadge'
import MetricTile from '../components/MetricTile'
import StageChip from '../components/StageChip'
import { num, pct, utcDateTime } from '../format'
import type {
  ApiEnvelope,
  GateRow,
  PaperRollup,
  SeriesPayload,
  StrategyDetailResponse,
  Transition,
} from '../types'

export default function StrategyDetail() {
  const { name = '' } = useParams()
  const enc = encodeURIComponent(name)
  const detail = useFetch<StrategyDetailResponse>(`/api/strategy/${enc}`)
  const series = useFetch<ApiEnvelope<SeriesPayload>>(`/api/strategy/${enc}/series`)
  const setFetchedAt = useSetFetchedAt()

  const fetchedAt = detail.data?.fetched_at ?? null
  useEffect(() => {
    setFetchedAt(fetchedAt)
    return () => setFetchedAt(null)
  }, [fetchedAt, setFetchedAt])

  if (detail.data === undefined) {
    if (detail.loading) return <div className="skeleton-block" style={{ height: 320 }} aria-hidden="true" />
    const error = detail.error
    return (
      <section className="panel error-panel">
        <span className="micro-label">fetch failed</span>
        <div className="error-code">
          {error instanceof ApiError && error.code !== null
            ? error.code
            : (error?.message ?? 'unknown error')}
        </div>
        <button type="button" className="retry-btn" onClick={detail.refetch}>
          retry
        </button>
      </section>
    )
  }

  const d = detail.data
  const registry = d.registry
  const paper = d.paper
  const gates = d.gates
  const partErrors = d.part_errors ?? {}
  const metaBits = [
    registry?.family ?? null,
    registry && registry.tags.length > 0 ? registry.tags.join(', ') : null,
    registry?.author ?? null,
  ].filter((x): x is string => x !== null)

  return (
    <>
      {d.stale && (
        <section className="banner-stale">
          <span className="micro-label">stale data</span>
          <div>
            showing cached data from {formatTimestamp(d.fetched_at)}
            {d.last_error_code != null && <>, last error {d.last_error_code}</>}
          </div>
        </section>
      )}

      <section className="detail-header">
        <div className="detail-title-row">
          <h1 className="detail-name">{d.strategy}</h1>
          {registry && <StageChip stage={registry.stage} />}
        </div>
        {metaBits.length > 0 && <div className="dim-note">{metaBits.join(' · ')}</div>}
        {registry?.description != null && registry.description !== '' && (
          <p className="detail-description">{registry.description}</p>
        )}
      </section>

      {partErrors.paper != null && (
        <section className="banner-stale">
          <span className="micro-label">degraded</span>
          <div>paper state unavailable: {partErrors.paper}</div>
        </section>
      )}
      {partErrors.gates != null && (
        <section className="banner-stale">
          <span className="micro-label">degraded</span>
          <div>gate history unavailable: {partErrors.gates}</div>
        </section>
      )}

      {series.data !== undefined ? (
        <EquityChart payload={series.data.data} />
      ) : series.loading ? (
        <div className="skeleton-block" style={{ height: 220 }} aria-hidden="true" />
      ) : (
        <section className="banner-stale">
          <span className="micro-label">degraded</span>
          <div>
            equity series unavailable
            {series.error instanceof ApiError && series.error.code !== null
              ? `: ${series.error.code}`
              : ''}
          </div>
        </section>
      )}

      {paper !== null && <PaperTiles paper={paper} />}

      {gates !== null && <GatesSection gates={gates} limit={d.gates_limit} />}

      {registry && registry.transitions.length > 0 && (
        <TransitionsTimeline transitions={registry.transitions} />
      )}

      {paper !== null && paper.recent_orders.length > 0 && <OrdersTable paper={paper} />}
    </>
  )
}

function PaperTiles({ paper }: { paper: PaperRollup }) {
  const tripped = paper.kill_switch?.tripped === true
  return (
    <section className="tile-row">
      <MetricTile label="health" value={<HealthBadge health={paper.health} />} />
      <MetricTile
        label="sess stale"
        value={paper.staleness_sessions ?? '—'}
        color={healthColor(paper.health)}
      />
      <MetricTile label="drawdown" value={pct(paper.drawdown?.drawdown)} />
      <MetricTile label="last equity" value={num(paper.drawdown?.last_equity)} />
      <MetricTile label="orders" value={paper.n_orders ?? '—'} />
      {tripped && (
        <MetricTile
          label="kill switch"
          value={paper.kill_switch?.reason ?? 'tripped'}
          color="var(--red)"
        />
      )}
    </section>
  )
}

function PassMark({
  passed,
  advisory = false,
}: {
  passed: number | boolean | null | undefined
  advisory?: boolean
}) {
  // The gate projection strips malformed fields — a missing verdict is UNKNOWN data,
  // not a real failed check.
  if (passed !== true && passed !== 1 && passed !== false && passed !== 0) {
    return (
      <span className="pass-mark" style={{ color: 'var(--text-dim)' }}>
        unknown
      </span>
    )
  }
  const ok = passed === true || passed === 1
  // An ADVISORY check has no veto power, so a failed one is a warning inside a passing
  // gate — gold, never the red of a breached binding floor.
  const color = advisory
    ? ok
      ? 'var(--text-dim)'
      : 'var(--amber)'
    : ok
      ? 'var(--green)'
      : 'var(--red)'
  return (
    <span className="pass-mark" style={{ color }}>
      {ok ? 'pass' : 'fail'}
    </span>
  )
}

function asNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function GatesSection({
  gates,
  limit,
}: {
  gates: NonNullable<StrategyDetailResponse['gates']>
  limit: number | undefined
}) {
  const [latest, ...older] = gates.gate_evaluations
  const forwards = gates.forward_gate_evaluations
  // `registry gates` returns the newest N per ledger; a saturated ledger has older rows
  // that are simply not here, so a bare count would imply a complete history.
  const truncated = (rows: GateRow[]) =>
    limit !== undefined && rows.length >= limit ? ` — newest ${limit}` : ''
  // The two ledgers gate DIFFERENT lifecycle edges and must be named as such: an
  // unqualified "gates: pass" on a forward_tested/live strategy reads as the forward
  // certificate when it is actually the (possibly much older) research gate.
  return (
    <section className="panel gates-panel">
      <span className="micro-label">research gate — backtested → candidate</span>
      {latest === undefined ? (
        <div className="dim-note">no research gate evaluations</div>
      ) : (
        <LatestGate row={latest} />
      )}
      {older.length > 0 && (
        <details className="sub-details">
          <summary className="micro-label">
            research gate history ({older.length}
            {truncated(gates.gate_evaluations)})
          </summary>
          <div className="row-list">
            {older.map((row) => (
              <CompactGateRow key={row.id} row={row} />
            ))}
          </div>
        </details>
      )}
      {forwards.length > 0 && (
        <details className="sub-details">
          <summary className="micro-label">
            forward gate — paper → forward_tested ({forwards.length}
            {truncated(forwards)})
          </summary>
          <div className="row-list">
            {forwards.map((row) => (
              <CompactGateRow key={row.id} row={row} />
            ))}
          </div>
        </details>
      )}
    </section>
  )
}

function LatestGate({ row }: { row: GateRow }) {
  const decision = row.decision
  const checks = decision?.checks ?? []
  // fdr_* live as ledger columns; dsr_confidence / appraisal_ratio only inside decision.
  const fdrP = asNum(row.fdr_p_value)
  const fdrAlpha = asNum(row.fdr_alpha_level)
  const dsrConfidence = asNum(decision?.dsr_confidence)
  const appraisalRatio = asNum(decision?.appraisal_ratio)
  const scalars: string[] = []
  if (fdrP !== null || fdrAlpha !== null) {
    scalars.push(`fdr p ${num(fdrP, 5)} vs α ${num(fdrAlpha, 5)}`)
  }
  if (dsrConfidence !== null) scalars.push(`dsr conf ${num(dsrConfidence, 4)}`)
  if (appraisalRatio !== null) scalars.push(`appraisal ${num(appraisalRatio, 3)}`)

  return (
    <div className="latest-gate">
      <div className="gate-verdict-row">
        <PassMark passed={row.passed} />
        <span className="dim-note num">{utcDateTime(row.created_at)}</span>
        <span className="stage-chip">{row.actor}</span>
        <TokenChip row={row} />
      </div>
      {decision === null ? (
        <div className="dim-note">decision unparseable: {row.decision_error ?? 'unknown'}</div>
      ) : (
        <>
          {checks.length > 0 && (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>check</th>
                    <th>op</th>
                    <th>threshold</th>
                    <th>value</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {checks.map((c, i) => (
                    <tr key={`${c.name ?? 'check'}-${i}`}>
                      <td>
                        {c.name ?? '—'}
                        {c.advisory === true && (
                          <span className="dim-note"> · advisory</span>
                        )}
                      </td>
                      <td className="dim-note">{c.op ?? '—'}</td>
                      <td className="num">{num(c.threshold, 4)}</td>
                      <td className="num">{num(c.value, 4)}</td>
                      <td>
                        <PassMark passed={c.passed} advisory={c.advisory === true} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {/* Without this, a passing gate that carries failed advisory checks — the NORMAL
              shape under the factory soft gate — reads as "promoted despite failing". */}
          {checks.some((c) => c.advisory === true) && (
            <div className="dim-note">
              advisory checks are recorded but never veto — only the binding checks decide
              the verdict
            </div>
          )}
          {scalars.length > 0 && <div className="dim-note num">{scalars.join(' · ')}</div>}
        </>
      )}
    </div>
  )
}

/** A PASSING gate mints a single-use token; the transition SPENDS it. Whether that token
 * is still unspent is the difference between "cleared the gate and moved" and "cleared the
 * gate, transition never happened" — invisible until now. Only meaningful on a pass. */
function TokenChip({ row }: { row: GateRow }) {
  const passed = row.passed === true || row.passed === 1
  if (!passed || row.consumed === null || row.consumed === undefined) return null
  const spent = row.consumed === true || row.consumed === 1
  return (
    <span className="stage-chip" style={{ color: spent ? 'var(--text-dim)' : 'var(--electric)' }}>
      {spent ? 'token spent' : 'token unspent'}
    </span>
  )
}

function CompactGateRow({ row }: { row: GateRow }) {
  return (
    <div className="compact-gate-row">
      <PassMark passed={row.passed} />
      <span className="dim-note num">{utcDateTime(row.created_at)}</span>
      <span className="stage-chip">{row.actor}</span>
      <TokenChip row={row} />
    </div>
  )
}

function TransitionsTimeline({ transitions }: { transitions: Transition[] }) {
  const rows = [...transitions].reverse()
  return (
    <section className="panel">
      <span className="micro-label">transitions</span>
      <div className="row-list timeline">
        {rows.map((t, i) => (
          <div key={t.id ?? `${t.created_at}-${i}`} className="timeline-row">
            <div className="timeline-main">
              <span className="timeline-stages">
                {t.from_stage ?? '∅'} → {t.to_stage}
              </span>
              <span className="stage-chip">{t.actor}</span>
            </div>
            {t.reason != null && t.reason !== '' && <div className="dim-note">{t.reason}</div>}
            <div className="dim-note num">{utcDateTime(t.created_at)}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function OrdersTable({ paper }: { paper: PaperRollup }) {
  return (
    <section className="panel">
      <span className="micro-label">recent orders</span>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>submitted</th>
              <th>symbol</th>
              <th>side</th>
              <th>status</th>
            </tr>
          </thead>
          <tbody>
            {paper.recent_orders.map((o, i) => (
              <tr key={o.broker_order_id ?? `${o.symbol}-${i}`}>
                <td className="num">{utcDateTime(o.submitted_ts)}</td>
                <td>{o.symbol}</td>
                <td>{o.side}</td>
                <td className="dim-note">{o.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
