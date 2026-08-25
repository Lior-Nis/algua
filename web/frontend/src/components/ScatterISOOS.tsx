import { runsUrl, useFetch } from '../api'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
import ChartFrame from './ChartFrame'

const HEIGHT = 240 // ChartFrame body height — fixed whether empty or populated.

// Internal SVG coordinate space. Not tied to the container's real pixel width — a viewBox
// scales to fit, so no ResizeObserver dance is needed the way EquityChart's canvas host does.
const PLOT = { width: 340, height: 200, left: 34, right: 12, top: 14, bottom: 24 }
const POINT_RADIUS = 6 // >=8px rendered diameter (mobile tap-target sizing, even though tap does nothing here)

// A point counts as an "outlier" worth direct-labelling only once its OOS Sharpe clears the
// walk-forward figure by at least this fraction of the plotted (padded) value range — a hair
// above the diagonal is still honest; direct labels are reserved for the runs that matter.
const OUTLIER_GAP_FRACTION = 0.2
const MAX_OUTLIER_LABELS = 3

function isFiniteMetric(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

export interface ScatterPoint {
  id: number
  strategy: string
  x: number
  y: number
  cx: number
  cy: number
  gap: number
  above: boolean
  outlier: boolean
}

export interface ScatterGeometry {
  points: ScatterPoint[]
  excludedCount: number
  diagonal: { x1: number; y1: number; x2: number; y2: number }
  domain: { min: number; max: number }
}

/** Pure layout: scalar run rows -> SVG-space points + the y=x reference line, both derived from
 * ONE shared domain (x and y are the same Sharpe scale) — so a point with equal IS/OOS values is
 * GUARANTEED to land exactly on the diagonal, not merely near a line that happens to look 45
 * degrees. `buildScatterGeometry` and the diagonal's endpoints both go through `scaleX`/`scaleY`
 * below; nothing about the diagonal is hand-picked independently of the points.
 *
 * A run missing EITHER metric is excluded and counted, never coerced to 0 — a missing
 * measurement plotted at zero would fabricate a data point on the most decision-relevant chart
 * in the app. */
export function buildScatterGeometry(runs: RunRow[]): ScatterGeometry {
  const finite = runs.filter(
    (r) => isFiniteMetric(r.mean_window_sharpe) && isFiniteMetric(r.sharpe_oos),
  )
  const excludedCount = runs.length - finite.length
  if (finite.length === 0) {
    return {
      points: [],
      excludedCount,
      diagonal: { x1: 0, y1: 0, x2: 0, y2: 0 },
      domain: { min: 0, max: 0 },
    }
  }

  const values = finite.flatMap((r) => [r.mean_window_sharpe as number, r.sharpe_oos as number])
  let domainMin = Math.min(...values)
  let domainMax = Math.max(...values)
  if (domainMin === domainMax) {
    // A single value (or every run tied) — pad an arbitrary but symmetric window around it so
    // the scale functions below never divide by zero.
    domainMin -= 0.5
    domainMax += 0.5
  }
  const span = domainMax - domainMin
  domainMin -= span * 0.15
  domainMax += span * 0.15
  const range = domainMax - domainMin

  const plotW = PLOT.width - PLOT.left - PLOT.right
  const plotH = PLOT.height - PLOT.top - PLOT.bottom
  const scaleX = (v: number) => PLOT.left + ((v - domainMin) / range) * plotW
  // y grows DOWNWARD in SVG space, so the larger value maps to the smaller pixel y.
  const scaleY = (v: number) => PLOT.top + (1 - (v - domainMin) / range) * plotH

  const raw = finite.map((r) => {
    const x = r.mean_window_sharpe as number
    const y = r.sharpe_oos as number
    return {
      id: r.id,
      strategy: r.strategy_name,
      x,
      y,
      cx: scaleX(x),
      cy: scaleY(y),
      gap: y - x,
      above: y > x,
    }
  })

  const threshold = range * OUTLIER_GAP_FRACTION
  const outlierIds = new Set<number>()
  for (const p of [...raw].sort((a, b) => b.gap - a.gap)) {
    if (outlierIds.size >= MAX_OUTLIER_LABELS) break
    if (p.gap >= threshold) outlierIds.add(p.id)
  }

  const points: ScatterPoint[] = raw.map((p) => ({ ...p, outlier: outlierIds.has(p.id) }))
  const diagonal = {
    x1: scaleX(domainMin),
    y1: scaleY(domainMin),
    x2: scaleX(domainMax),
    y2: scaleY(domainMax),
  }
  return { points, excludedCount, diagonal, domain: { min: domainMin, max: domainMax } }
}

function fmt(v: number): string {
  return v.toFixed(2)
}

export default function ScatterISOOS() {
  const { data } = useFetch<ApiEnvelope<RunsListPayload>>(runsUrl({ kind: 'gate' }))
  const runs = data?.data.runs ?? []
  const geometry = buildScatterGeometry(runs)
  const isEmpty = geometry.points.length === 0

  const emptyLabel =
    runs.length === 0
      ? 'no gate runs recorded yet'
      : `${geometry.excludedCount} gate run${geometry.excludedCount === 1 ? '' : 's'} recorded, ` +
        'none with both IS and OOS metrics yet'

  const excludedNote =
    geometry.excludedCount > 0
      ? `${geometry.excludedCount} run${geometry.excludedCount === 1 ? '' : 's'} excluded — ` +
        'missing an IS or OOS metric'
      : null

  return (
    <ChartFrame title="is vs oos" isEmpty={isEmpty} emptyLabel={emptyLabel} height={HEIGHT}>
      <div className="scatter-body">
        <svg
          className="scatter-svg"
          viewBox={`0 0 ${PLOT.width} ${PLOT.height}`}
          role="img"
          aria-label="in-sample versus out-of-sample Sharpe scatter, diagonal is the honest line"
        >
          <rect
            className="scatter-plot-area"
            x={PLOT.left}
            y={PLOT.top}
            width={PLOT.width - PLOT.left - PLOT.right}
            height={PLOT.height - PLOT.top - PLOT.bottom}
          />
          <line
            data-testid="diagonal"
            className="scatter-diagonal"
            x1={geometry.diagonal.x1}
            y1={geometry.diagonal.y1}
            x2={geometry.diagonal.x2}
            y2={geometry.diagonal.y2}
          />
          <text
            className="scatter-tick"
            x={PLOT.left}
            y={PLOT.height - PLOT.bottom + 12}
          >
            {fmt(geometry.domain.min)}
          </text>
          <text
            className="scatter-tick"
            x={PLOT.width - PLOT.right}
            y={PLOT.height - PLOT.bottom + 12}
            textAnchor="end"
          >
            {fmt(geometry.domain.max)}
          </text>
          <text
            className="scatter-caption"
            x={(PLOT.left + PLOT.width - PLOT.right) / 2}
            y={PLOT.height - 4}
            textAnchor="middle"
          >
            mean window sharpe (is)
          </text>
          <text className="scatter-caption" x={0} y={PLOT.top - 4}>
            oos
          </text>
          {geometry.points.map((p) => (
            <g key={p.id} data-testid="scatter-point" data-region={p.above ? 'above' : 'below'}>
              <circle
                className={p.above ? 'scatter-point-above' : 'scatter-point-below'}
                cx={p.cx}
                cy={p.cy}
                r={POINT_RADIUS}
              />
              {p.outlier && (
                <text
                  className="scatter-point-label"
                  x={p.cx + POINT_RADIUS + 3}
                  y={p.cy - POINT_RADIUS}
                >
                  {p.strategy}
                </text>
              )}
            </g>
          ))}
        </svg>
        {excludedNote != null && (
          <div className="chart-footnote">{excludedNote}</div>
        )}
      </div>
    </ChartFrame>
  )
}
