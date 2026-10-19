import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { ApiError, formatTimestamp, useFetch } from '../api'
import { useSetFetchedAt } from '../App'
import EquityChart, { drawableSeriesPointCount } from '../components/EquityChart'
import EvidenceDisclosure from '../components/EvidenceDisclosure'
import EvidenceStatus from '../components/EvidenceStatus'
import GateBulletCard from '../components/GateBulletCard'
import HealthBadge, { healthColor } from '../components/HealthBadge'
import MetricTile from '../components/MetricTile'
import PassMark from '../components/PassMark'
import ReturnOverlay from '../components/ReturnOverlay'
import StageChip from '../components/StageChip'
import TrialDistribution from '../components/TrialDistribution'
import { num, pct, utcDateTime } from '../format'
import type {
  ApiEnvelope,
  GateRow,
  PaperRollup,
  SeriesPayload,
  StrategyDetailResponse,
  Transition,
} from '../types'
import { deriveStrategyDecision } from './strategyDecision'
import type { SeriesEvidenceState } from './strategyDecision'

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
  const detailStale = d.stale || detail.error !== null
  const seriesPayload = series.data?.data
  const paperRows = seriesPayload?.series.paper ?? []
  const liveRows = seriesPayload?.series.live ?? []
  const rawSeriesCount = paperRows.length + liveRows.length
  const drawableByLane = [
    drawableSeriesPointCount(seriesPayload?.series.paper ?? null),
    drawableSeriesPointCount(seriesPayload?.series.live ?? null),
  ]
  const seriesThinState: 'empty' | 'incomplete' | 'current' = rawSeriesCount === 0
    ? 'empty'
    : drawableByLane.some((count) => count >= 2)
      ? 'current'
      : 'incomplete'
  const seriesStale = series.data !== undefined && (series.data.stale || series.error !== null)
  const seriesState: SeriesEvidenceState = series.data !== undefined
    ? seriesStale
      ? 'stale'
      : seriesThinState
    : series.loading
      ? 'loading'
      : 'failed'
  const decision = deriveStrategyDecision(
    detailStale === d.stale ? d : { ...d, stale: true },
    seriesState,
  )
  const seriesSummary = seriesStale ? `stale · ${seriesThinState}` : seriesState
  const metaBits = [
    registry?.family ?? null,
    registry && registry.tags.length > 0 ? registry.tags.join(', ') : null,
    registry?.author ?? null,
  ].filter((x): x is string => x !== null)

  return (
    <main className="decision-page strategy-detail-page">
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

      <section
        className="decision-report strategy-decision"
        data-state={decision.tone}
        aria-labelledby="strategy-decision-heading"
      >
        <div className="decision-report-head">
          <div>
            <h2 id="strategy-decision-heading" className="micro-label">trader decision</h2>
            <div className="decision-state">{decision.state}</div>
          </div>
          <div className="decision-consequence">{decision.consequence}</div>
        </div>
        <div className="decision-facts">
          <div><span className="micro-label">reason</span><span>{decision.reason}</span></div>
          <div>
            <span className="micro-label">next checkpoint</span>
            <span>{decision.nextCheckpoint}</span>
          </div>
        </div>
        {detailStale && (
          <div className="decision-raw-line">
            cached {formatTimestamp(d.fetched_at)}
            {(d.last_error_code != null || detail.error !== null) && (
              <> · last error {d.last_error_code ?? detail.error?.message}</>
            )}
          </div>
        )}
      </section>

      <div className="strategy-evidence-grid">
        <EvidenceDisclosure
          title="performance evidence"
          summary={`${seriesSummary}${paper !== null ? ` · paper ${paper.health}` : ''}`}
        >
          <EquityChart
            payload={series.data?.data}
            loading={series.loading}
            error={series.error}
            stale={seriesStale}
            onRetry={series.refetch}
          />
          <ReturnOverlay strategy={d.strategy} family={registry?.family ?? null} />
          {paper !== null && <PaperTiles paper={paper} />}
          {partErrors.paper != null && (
            <EvidenceStatus title="paper state" state="failed" detail={partErrors.paper} />
          )}
        </EvidenceDisclosure>

        <EvidenceDisclosure
          title="research evidence"
          summary={gates === null ? 'unavailable' : `${gates.gate_evaluations.length} research gates`}
        >
          {gates !== null ? (
            <GatesSection gates={gates} limit={d.gates_limit} strategy={d.strategy} />
          ) : (
            <EvidenceStatus
              title="gate history"
              state="failed"
              detail={partErrors.gates ?? 'unavailable'}
            />
          )}
        </EvidenceDisclosure>

        <EvidenceDisclosure
          title="raw provenance"
          summary={`${registry?.transitions.length ?? 0} transitions · ${paper === null ? 'orders unavailable' : `${paper.recent_orders.length} orders`}`}
        >
          {registry && registry.transitions.length > 0 ? (
            <TransitionsTimeline transitions={registry.transitions} />
          ) : (
            <EvidenceStatus title="transition history" state="empty" detail="no recorded transitions" />
          )}
          {paper !== null && paper.recent_orders.length > 0 ? (
            <OrdersTable paper={paper} />
          ) : paper === null ? (
            <EvidenceStatus
              title="recent orders"
              state="failed"
              detail={partErrors.paper ?? 'paper rollup unavailable'}
            />
          ) : (
            <EvidenceStatus title="recent orders" state="empty" detail="no recorded orders" />
          )}
        </EvidenceDisclosure>
      </div>
    </main>
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

function asNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function GatesSection({
  gates,
  limit,
  strategy,
}: {
  gates: NonNullable<StrategyDetailResponse['gates']>
  limit: number | undefined
  strategy: string
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
        <LatestGate row={latest} strategy={strategy} />
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

function LatestGate({ row, strategy }: { row: GateRow; strategy: string }) {
  const decision = row.decision
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
          <GateBulletCard checks={decision.checks ?? []} />
          {scalars.length > 0 && <div className="dim-note num">{scalars.join(' · ')}</div>}
          <TrialDistribution strategy={strategy} />
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
