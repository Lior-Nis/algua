import { runDetailUrl, runsUrl, useFetch } from '../api'
import type { ApiEnvelope, RunDetail, RunRow, RunsListPayload } from '../types'
import ChartFrame from './ChartFrame'

const HEIGHT = 220 // ChartFrame body height — fixed whether empty or populated.

// Internal SVG coordinate space (viewBox scales to fit, same convention as ScatterISOOS/
// GateBulletCard — no ResizeObserver dance needed).
const PLOT = { width: 340, height: 190, left: 16, right: 16 }
const LABEL_Y = 12 // threshold rule's direct label
const RULE_TOP = 20 // the rule spans the whole plot, judged against the entire swarm
const SWARM_TOP = 30
const SWARM_HEIGHT = 66
const MARKER_ROW_Y = 122 // dedicated row BELOW the swarm — a distinct position, not a jittered dot
const AXIS_Y = 148
const CAPTION_Y = 178
const POINT_RADIUS = 3.5 // trial dots — small, honest, individually plotted (N~70, not binned)
const MARKER_RADIUS = 7 // this-strategy marker — visibly larger than a trial dot

function isFiniteMetric(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

/** Deterministic pseudo-random unit value from an integer seed (a run id). A strip/dot plot at
 * this N (~70) needs SOME vertical spread to keep same-valued trials from stacking into one
 * illegible column, but the spread must be stable across re-renders (and identical in every
 * test run) rather than reshuffling each time `useFetch` refires — a hash of the id, not
 * `Math.random()`, gives the same trial the same row every time. */
function hashUnit(seed: number): number {
  let x = Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b) >>> 0
  x ^= x >>> 13
  x = Math.imul(x, 0xc2b2ae35) >>> 0
  x ^= x >>> 16
  return (x >>> 0) / 4294967296
}

export interface TrialPoint {
  id: number
  strategy: string
  value: number
  cx: number
  cy: number
}

export interface ThresholdGeometry {
  value: number
  x: number
}

export interface OwnMarkerGeometry {
  value: number
  passed: boolean | null
  cx: number
  cy: number
}

export interface TrialDistributionGeometry {
  points: TrialPoint[]
  excludedCount: number
  domain: { min: number; max: number }
  threshold: ThresholdGeometry | null
  marker: OwnMarkerGeometry | null
}

/** Pure layout: funnel-wide sweep-trial rows + this strategy's deflated bar/holdout result ->
 * SVG-space geometry. A trial missing `mean_window_sharpe` is excluded and counted, never
 * coerced to 0 — a missing measurement plotted at zero would fabricate a data point on the very
 * chart that argues the bar is trustworthy. The shared domain always stretches to cover the
 * threshold and the marker even when they sit far outside the trial cluster (the real case this
 * chart exists to show: holdout 0.025 against a bar of 2.677) — clipping either off the canvas
 * would hide the argument the chart is making. */
export function buildTrialDistributionGeometry(
  trials: RunRow[],
  effectiveMinHoldoutSharpe: number | null | undefined,
  ownHoldout: { value: number | null | undefined; passed: boolean | null | undefined } | null,
): TrialDistributionGeometry {
  const finite = trials.filter((t) => isFiniteMetric(t.mean_window_sharpe))
  const excludedCount = trials.length - finite.length
  const threshold = isFiniteMetric(effectiveMinHoldoutSharpe) ? effectiveMinHoldoutSharpe : null
  const ownValue =
    ownHoldout != null && isFiniteMetric(ownHoldout.value) ? (ownHoldout.value as number) : null

  if (finite.length === 0) {
    return { points: [], excludedCount, domain: { min: 0, max: 0 }, threshold: null, marker: null }
  }

  const values = finite.map((t) => t.mean_window_sharpe as number)
  let domainMin = Math.min(...values)
  let domainMax = Math.max(...values)
  if (threshold !== null) {
    domainMin = Math.min(domainMin, threshold)
    domainMax = Math.max(domainMax, threshold)
  }
  if (ownValue !== null) {
    domainMin = Math.min(domainMin, ownValue)
    domainMax = Math.max(domainMax, ownValue)
  }
  if (domainMin === domainMax) {
    domainMin -= 0.5
    domainMax += 0.5
  }
  const span = domainMax - domainMin
  domainMin -= span * 0.12
  domainMax += span * 0.12
  const range = domainMax - domainMin

  const plotW = PLOT.width - PLOT.left - PLOT.right
  const scaleX = (v: number) => PLOT.left + ((v - domainMin) / range) * plotW

  const points: TrialPoint[] = finite.map((t) => ({
    id: t.id,
    strategy: t.strategy_name,
    value: t.mean_window_sharpe as number,
    cx: scaleX(t.mean_window_sharpe as number),
    cy: SWARM_TOP + hashUnit(t.id) * SWARM_HEIGHT,
  }))

  const thresholdGeom: ThresholdGeometry | null =
    threshold !== null ? { value: threshold, x: scaleX(threshold) } : null

  const marker: OwnMarkerGeometry | null =
    ownValue !== null
      ? {
          value: ownValue,
          passed:
            ownHoldout?.passed === true ? true : ownHoldout?.passed === false ? false : null,
          cx: scaleX(ownValue),
          cy: MARKER_ROW_Y,
        }
      : null

  return {
    points,
    excludedCount,
    domain: { min: domainMin, max: domainMax },
    threshold: thresholdGeom,
    marker,
  }
}

function fmt(v: number): string {
  return v.toFixed(2)
}

/**
 * View 3 — the funnel trial distribution + the deflated bar (spec §6.1). This renders the
 * argument that kills most strategies: try enough variants and the best one looks good by luck,
 * so the promotion gate's holdout-Sharpe bar is DEFLATED by how much searching the funnel has
 * done (`effective_min_holdout_sharpe`, `algua/research/gates.py`). A general-purpose experiment
 * tracker has no concept of breadth deflation — this is the most algua-specific view in the
 * slice.
 *
 * Funnel-wide, not per-sweep: a single sweep is 5-8 combos, but the gate's breadth figure is
 * ACCUMULATED across the funnel window, so the trial source is `/api/runs?kind=sweep_trial` with
 * NO `strategy` filter — every strategy's trials, the same population the bar was computed
 * against. `limit=500` (the API's max) rather than the 100 default: silently truncating the one
 * chart whose entire point is showing accumulated breadth would misrepresent the very thing it
 * argues about.
 *
 * At N~70 a dot/strip plot is chosen over a histogram: it shows every individual trial honestly
 * rather than imposing bins that would hide exactly how close (or far) the cluster sits from the
 * bar. Trial dots get a deterministic vertical jitter (`hashUnit`) purely to keep same-valued
 * trials legible — the y position carries no meaning.
 *
 * The threshold rule and the strategy's own holdout marker are the two things the chart is FOR,
 * so each gets its own colour lane, not a third categorical hue: the rule is `--series-focus`
 * (Electric, the one thing that matters on this screen) and is drawn as a bold rule spanning the
 * whole plot — real data, not axis chrome. The marker uses the STATUS palette (pass=green,
 * fail=red), because it fundamentally IS a pass/fail verdict against the rule, and is shaped as a
 * diamond on its own dedicated row below the swarm (never a circle at swarm height) so it can
 * never be mistaken for one more trial dot. Both carry a direct text label — there is no hover on
 * mobile to fall back on.
 *
 * Source for the threshold/marker: this strategy's own latest research-gate run
 * (`/api/runs?strategy=&kind=gate&limit=1` -> `/api/runs/{id}`, the SAME two-step waterfall
 * `GateBulletCard` uses), reading `gate_decision.effective_min_holdout_sharpe` for the bar and
 * the `holdout_sharpe` check's `value`/`passed` for the marker — the same check the bullet card
 * renders, so the two views can never disagree about what "the bar" or "the result" was.
 */
export default function TrialDistribution({ strategy }: { strategy: string }) {
  const trialsFetch = useFetch<ApiEnvelope<RunsListPayload>>(
    runsUrl({ kind: 'sweep_trial', limit: 500 }),
  )
  const gateListFetch = useFetch<ApiEnvelope<RunsListPayload>>(
    runsUrl({ strategy, kind: 'gate', limit: 1 }),
  )
  const gateRunId = gateListFetch.data?.data?.runs?.[0]?.id ?? null
  const gateDetailFetch = useFetch<ApiEnvelope<RunDetail>>(
    gateRunId !== null ? runDetailUrl(gateRunId) : null,
  )

  const trials = trialsFetch.data?.data.runs ?? []
  const decision = gateDetailFetch.data?.data?.gate_decision ?? null
  const effectiveMinHoldoutSharpe =
    typeof decision?.effective_min_holdout_sharpe === 'number'
      ? (decision.effective_min_holdout_sharpe as number)
      : null
  const holdoutCheck = decision?.checks?.find((c) => c.name === 'holdout_sharpe') ?? null
  const ownHoldout = holdoutCheck
    ? {
        value: typeof holdoutCheck.value === 'number' ? holdoutCheck.value : null,
        passed:
          holdoutCheck.passed === true ? true : holdoutCheck.passed === false ? false : null,
      }
    : null

  const geometry = buildTrialDistributionGeometry(trials, effectiveMinHoldoutSharpe, ownHoldout)
  const isEmpty = geometry.points.length === 0

  const emptyLabel =
    trials.length === 0
      ? 'no sweep trials recorded yet'
      : `${geometry.excludedCount} sweep trial${geometry.excludedCount === 1 ? '' : 's'} ` +
        'recorded, none with a mean-window-sharpe metric yet'

  const excludedNote =
    geometry.excludedCount > 0
      ? `${geometry.excludedCount} trial${geometry.excludedCount === 1 ? '' : 's'} excluded — ` +
        'missing a mean-window-sharpe metric'
      : null

  const rightEdge = PLOT.width - PLOT.right
  // Anchor buffers are sized to each label's OWN text ("deflated bar N.NN" vs the longer
  // "this strategy N.NN · advisory" qualifier) so a label near the right edge flips to
  // right-anchored instead of running off the viewBox — re-measured after the advisory
  // qualifier lengthened the marker label (fix round 1).
  const thresholdLabelAnchor =
    geometry.threshold !== null && geometry.threshold.x > rightEdge - 85 ? 'end' : 'start'
  const markerLabelAnchor =
    geometry.marker !== null && geometry.marker.cx > rightEdge - 140 ? 'end' : 'start'

  return (
    <ChartFrame title="funnel trial distribution" isEmpty={isEmpty} emptyLabel={emptyLabel} height={HEIGHT}>
      <div className="trial-dist-body">
        <svg
          className="trial-dist-svg"
          viewBox={`0 0 ${PLOT.width} ${PLOT.height}`}
          role="img"
          aria-label={
            `funnel-wide sweep trial distribution, ${geometry.points.length} trials plotted` +
            (geometry.threshold !== null
              ? `; deflated bar at ${fmt(geometry.threshold.value)}`
              : '') +
            (geometry.marker !== null
              ? `; this strategy's holdout result ${fmt(geometry.marker.value)}, ` +
                `${geometry.marker.passed === true ? 'clears the deflated bar' : geometry.marker.passed === false ? 'below the deflated bar' : 'verdict unknown'}` +
                ' (advisory check, does not veto the gate)'
              : '')
          }
        >
          <rect
            className="trial-dist-plot-area"
            x={PLOT.left}
            y={RULE_TOP}
            width={PLOT.width - PLOT.left - PLOT.right}
            height={AXIS_Y - RULE_TOP}
          />
          {geometry.points.map((p) => (
            <circle
              key={p.id}
              data-testid="trial-point"
              className="trial-dist-point"
              cx={p.cx}
              cy={p.cy}
              r={POINT_RADIUS}
            />
          ))}
          {geometry.threshold !== null && (
            <>
              <line
                data-testid="threshold-line"
                className="trial-dist-threshold"
                x1={geometry.threshold.x}
                x2={geometry.threshold.x}
                y1={RULE_TOP}
                y2={AXIS_Y}
              />
              <text
                data-testid="threshold-label"
                className="trial-dist-threshold-label"
                x={geometry.threshold.x}
                y={LABEL_Y}
                textAnchor={thresholdLabelAnchor}
              >
                deflated bar {fmt(geometry.threshold.value)}
              </text>
            </>
          )}
          {geometry.marker !== null && (
            <>
              <line
                className="trial-dist-guide"
                x1={geometry.marker.cx}
                x2={geometry.marker.cx}
                y1={SWARM_TOP}
                y2={MARKER_ROW_Y - MARKER_RADIUS}
              />
              <polygon
                data-testid="own-marker"
                className={
                  geometry.marker.passed === true
                    ? 'trial-dist-marker-pass'
                    : geometry.marker.passed === false
                      ? 'trial-dist-marker-fail'
                      : 'trial-dist-marker-unknown'
                }
                points={[
                  `${geometry.marker.cx},${geometry.marker.cy - MARKER_RADIUS}`,
                  `${geometry.marker.cx + MARKER_RADIUS},${geometry.marker.cy}`,
                  `${geometry.marker.cx},${geometry.marker.cy + MARKER_RADIUS}`,
                  `${geometry.marker.cx - MARKER_RADIUS},${geometry.marker.cy}`,
                ].join(' ')}
              />
              <text
                data-testid="own-marker-label"
                className="trial-dist-marker-label"
                x={geometry.marker.cx}
                y={MARKER_ROW_Y + MARKER_RADIUS + 11}
                textAnchor={markerLabelAnchor}
              >
                this strategy {fmt(geometry.marker.value)} · advisory
              </text>
            </>
          )}
          <text className="trial-dist-tick" x={PLOT.left} y={AXIS_Y + 12}>
            {fmt(geometry.domain.min)}
          </text>
          <text
            className="trial-dist-tick"
            x={rightEdge}
            y={AXIS_Y + 12}
            textAnchor="end"
          >
            {fmt(geometry.domain.max)}
          </text>
          <text
            className="trial-dist-caption"
            x={(PLOT.left + rightEdge) / 2}
            y={CAPTION_Y}
            textAnchor="middle"
          >
            mean window sharpe — trials · bar & marker: holdout sharpe
          </text>
        </svg>
        {excludedNote !== null && <div className="chart-footnote">{excludedNote}</div>}
      </div>
    </ChartFrame>
  )
}
