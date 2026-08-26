import { runDetailUrl, runsUrl, useFetch } from '../api'
import { num } from '../format'
import type { ApiEnvelope, RunDetail, RunRow, RunsListPayload } from '../types'
import ChartFrame from './ChartFrame'

// ChartFrame body height — fixed whether empty or populated. Raised from 220 in fix round 3:
// both viewBoxes grew taller to unstack the axis ticks from the axis captions, and since each
// SVG is `width: 100%` inside a flex row of the body's fixed height, a taller viewBox at the
// same body height renders NARROWER (it letterboxes to fit). 240 restores the plotted width the
// card had before the spacing fix rather than paying for legibility with ink density.
const HEIGHT = 240

// The API's own max page size (see the docstring below) — `runs.length === TRIAL_LIMIT` is the
// only honest truncation signal available: the payload's `count` is just the returned row count,
// never a funnel-wide total.
const TRIAL_LIMIT = 500

// Two INDEPENDENT SVG coordinate spaces — one per mark — each scaled to its OWN data (fix round
// 2: a `sweep_trial` row never carries an OOS metric, so `mean_window_sharpe` (the trial cloud)
// and holdout-class Sharpe (the deflation strip) cannot share one axis; see this file's top
// docstring).
// Vertical budget, both marks (fix round 3): the axis ticks and the axis caption used to sit 8px
// (cloud) and 2px (strip) apart at an 8px font size — they crowded in the cloud and genuinely
// OVERLAPPED in the strip, saved only by the caption being centred while the ticks sit at the
// ends. Each viewBox is taller now so the tick row and the caption row are ~7px clear of each
// other's ink. The captions are also shorter, so a wide caption can never reach an end tick.
const CLOUD_PLOT = { width: 340, height: 142, left: 16, right: 16 }
const CLOUD_SWARM_TOP = 12
const CLOUD_SWARM_HEIGHT = 46
const CLOUD_OWN_ROW_Y = 80 // this strategy's own marker — a dedicated row, not blended into the jittered cloud
const CLOUD_AXIS_Y = 104
const CLOUD_TICK_Y = CLOUD_AXIS_Y + 10
const CLOUD_CAPTION_Y = 130
const CLOUD_POINT_RADIUS = 3.5 // trial dots — small, honest, individually plotted (N~70, not binned)
const CLOUD_OWN_RADIUS = 6 // visibly larger than a trial dot

const STRIP_PLOT = { width: 340, height: 96, left: 16, right: 16 }
const STRIP_ROW_Y = 36
const STRIP_BAR_LABEL_Y = 14
const STRIP_OWN_LABEL_Y = 58
const STRIP_AXIS_Y = 72
const STRIP_TICK_Y = STRIP_AXIS_Y + 10
const STRIP_CAPTION_Y = 94
const STRIP_MARKER_RADIUS = 7

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

export interface OwnCloudMarker {
  value: number
  cx: number
}

export interface TrialCloudGeometry {
  points: TrialPoint[]
  excludedCount: number
  domain: { min: number; max: number }
  own: OwnCloudMarker | null
}

/** Pure layout, mark 1 of 2 — the funnel-wide trial cloud on `mean_window_sharpe`, with THIS
 * strategy's own `mean_window_sharpe` (its walk-forward run's stability mean — the SAME
 * statistic every `sweep_trial` row carries) marked. Same sample class throughout, so the
 * comparison is legitimate: "how much searching happened, and where did I land in it?"
 *
 * A trial missing `mean_window_sharpe` is excluded and counted, never coerced to 0 — a missing
 * measurement plotted at zero would fabricate a data point. The domain scales to the trial
 * cloud (and the own marker, when it sits outside the cloud) ONLY — never stretched to also fit
 * a holdout-class number, which lives on its own axis in `buildHoldoutStripGeometry` below. */
export function buildTrialCloudGeometry(
  trials: RunRow[],
  ownMeanWindowSharpe: number | null | undefined,
): TrialCloudGeometry {
  const finite = trials.filter((t) => isFiniteMetric(t.mean_window_sharpe))
  const excludedCount = trials.length - finite.length
  const ownValue = isFiniteMetric(ownMeanWindowSharpe) ? ownMeanWindowSharpe : null

  if (finite.length === 0) {
    return { points: [], excludedCount, domain: { min: 0, max: 0 }, own: null }
  }

  const values = finite.map((t) => t.mean_window_sharpe as number)
  let domainMin = Math.min(...values)
  let domainMax = Math.max(...values)
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

  const plotW = CLOUD_PLOT.width - CLOUD_PLOT.left - CLOUD_PLOT.right
  const scaleX = (v: number) => CLOUD_PLOT.left + ((v - domainMin) / range) * plotW

  const points: TrialPoint[] = finite.map((t) => ({
    id: t.id,
    strategy: t.strategy_name,
    value: t.mean_window_sharpe as number,
    cx: scaleX(t.mean_window_sharpe as number),
    cy: CLOUD_SWARM_TOP + hashUnit(t.id) * CLOUD_SWARM_HEIGHT,
  }))

  const own: OwnCloudMarker | null =
    ownValue !== null ? { value: ownValue, cx: scaleX(ownValue) } : null

  return { points, excludedCount, domain: { min: domainMin, max: domainMax }, own }
}

export interface StripPoint {
  value: number
  cx: number
}

export interface HoldoutStripGeometry {
  bar: StripPoint | null
  own: (StripPoint & { passed: boolean | null }) | null
  domain: { min: number; max: number } | null
}

/** Pure layout, mark 2 of 2 — a two-value strip: the deflated holdout-Sharpe bar
 * (`effective_min_holdout_sharpe`) against this strategy's own holdout Sharpe. BOTH values are
 * holdout-class (never mixed with the trial cloud's `mean_window_sharpe`), so this is the ONE
 * legitimate place to ask "did I clear it?" Never stretched to also contain the trial cloud.
 *
 * ZERO-ANCHORED (fix round 3), exactly like `GateBulletCard`'s `buildBulletGeometry` and the
 * sparkline's baseline. A two-value strip scaled to just those two values turns ANY margin,
 * however tiny, into the FULL plot width: on real seeded data the bar was 0.6118 and the holdout
 * 0.61, and the auto-scaled strip rendered that 0.0018 miss as a chasm spanning the whole card —
 * a card whose entire job is answering "did I clear it, and by how much" answering it with a
 * gaping visual gap between two numbers that are all but identical. Including 0 in the domain
 * puts both marks on a scale with real magnitude, so a hair's-breadth miss renders as a hair and
 * a genuine miss renders as a gap. The margin itself is stated as a NUMBER in the prose summary
 * (`buildSummary`) rather than left to be eyeballed off two labels. */
export function buildHoldoutStripGeometry(
  effectiveMinHoldoutSharpe: number | null | undefined,
  ownHoldout: { value: number | null | undefined; passed: boolean | null | undefined } | null,
): HoldoutStripGeometry {
  const barValue = isFiniteMetric(effectiveMinHoldoutSharpe) ? effectiveMinHoldoutSharpe : null
  const ownValue =
    ownHoldout != null && isFiniteMetric(ownHoldout.value) ? (ownHoldout.value as number) : null

  if (barValue === null && ownValue === null) {
    return { bar: null, own: null, domain: null }
  }

  const values = [0, barValue, ownValue].filter((v): v is number => v !== null)
  let domainMin = Math.min(...values)
  let domainMax = Math.max(...values)
  if (domainMin === domainMax) {
    domainMin -= 0.5
    domainMax += 0.5
  }
  const span = domainMax - domainMin
  domainMin -= span * 0.15
  domainMax += span * 0.15
  const range = domainMax - domainMin

  const plotW = STRIP_PLOT.width - STRIP_PLOT.left - STRIP_PLOT.right
  const scaleX = (v: number) => STRIP_PLOT.left + ((v - domainMin) / range) * plotW

  const bar: StripPoint | null = barValue !== null ? { value: barValue, cx: scaleX(barValue) } : null
  const own =
    ownValue !== null
      ? {
          value: ownValue,
          cx: scaleX(ownValue),
          passed:
            ownHoldout?.passed === true ? true : ownHoldout?.passed === false ? false : null,
        }
      : null

  return { bar, own, domain: { min: domainMin, max: domainMax } }
}

/** AXIS CHROME only — the two end ticks bounding each mark's domain. Two decimals is right for a
 * domain read at a glance; the DATA labels use `fmtHoldout` below. */
function fmt(v: number): string {
  return v.toFixed(2)
}

/** Every holdout-class NUMBER the strip and its prose print — the bar, this strategy's result,
 * and the margin between them. `num(v, 4)` is character-for-character what `GateBulletCard`
 * prints for the very same `holdout_sharpe` check (`num(check.value, 4)`), so the two views on
 * one screen can no longer disagree. At `toFixed(2)` they did: a 0.6118 bar and a 0.61 holdout
 * BOTH printed "0.61" here while the bullet card printed "0.6118" and "0.61" one card away. */
function fmtHoldout(v: number): string {
  return num(v, 4)
}

/** Below this, a margin ROUNDS AWAY at the 4-decimal display precision — printing "missed by 0"
 * would be worse than saying the two are level. */
const MARGIN_EPSILON = 0.00005

/** "cleared by 0.05" / "missed by 0.0018" — the difference is the number the reader actually
 * wants, and deriving it from two rounded labels is exactly what failed in review round 2 (both
 * labels read "0.61"). Stated as a number rather than left to the eye. */
export function marginPhrase(ownValue: number, barValue: number): string {
  const margin = ownValue - barValue
  if (Math.abs(margin) < MARGIN_EPSILON) return 'level with the bar'
  return `${margin > 0 ? 'cleared' : 'missed'} by ${fmtHoldout(Math.abs(margin))}`
}

/** The one sentence tying the two marks together (fix round 2): the causal story — breadth
 * produces the bar — is the point of putting these two independently-scaled marks in one card.
 * Only rendered once the strip has a bar to name; the trial count is the cloud's OWN plotted
 * count (the funnel-wide search volume the reader just saw), not a re-derivation. */
function buildSummary(nTrials: number, strip: HoldoutStripGeometry): string | null {
  if (strip.bar === null) return null
  const trialWord = nTrials === 1 ? 'trial' : 'trials'
  const lead = `${nTrials} ${trialWord} of search raised the bar to ${fmtHoldout(strip.bar.value)}`
  if (strip.own === null) return `${lead}.`
  return (
    `${lead}; your holdout was ${fmtHoldout(strip.own.value)} — ` +
    `${marginPhrase(strip.own.value, strip.bar.value)}.`
  )
}

/**
 * View 3 — the funnel trial distribution + the deflated bar (spec §6.1). This renders the
 * argument that kills most strategies: try enough variants and the best one looks good by luck,
 * so the promotion gate's holdout-Sharpe bar is DEFLATED by how much searching the funnel has
 * done (`effective_min_holdout_sharpe`, `algua/research/gates.py`). A general-purpose experiment
 * tracker has no concept of breadth deflation — this is the most algua-specific view in the
 * slice.
 *
 * TWO STACKED, INDEPENDENT MARKS (fix round 2) — not one shared axis. A `sweep_trial` row carries
 * only the four window-stability statistics and can never carry an OOS metric (a trial never
 * burns a holdout), so there is no common scale for "every trial's mean_window_sharpe" and "the
 * holdout-Sharpe bar/result" to share. Putting them on one linear axis — even a domain-stretched
 * one — reintroduces exactly the failure the promotion gate exists to prevent: reading an
 * in-sample/search-population number and an out-of-sample/single-use number as if they were
 * comparable magnitudes. Two measures of different scale get two charts (dataviz skill), never a
 * dual-axis chart:
 *
 *   1. The TRIAL CLOUD (`buildTrialCloudGeometry`) — every funnel-wide `sweep_trial`'s
 *      `mean_window_sharpe`, with this strategy's OWN `mean_window_sharpe` (its walk-forward
 *      run's stability mean, the identical statistic) marked. Same sample class throughout:
 *      "how much searching happened, and where did I land in it?"
 *   2. The DEFLATION STRIP (`buildHoldoutStripGeometry`) — the deflated bar
 *      (`effective_min_holdout_sharpe`) against this strategy's own holdout Sharpe, both
 *      holdout-class, both direct-labelled: "did I clear it?"
 *   3. One prose caption (`buildSummary`) ties them together — the causal story (breadth
 *      produces the bar) is the point, not a decoration.
 *
 * Funnel-wide, not per-sweep: a single sweep is 5-8 combos, but the gate's breadth figure is
 * ACCUMULATED across the funnel window, so the trial source is `/api/runs?kind=sweep_trial` with
 * NO `strategy` filter — every strategy's trials, the same population the bar was computed
 * against. `limit=500` (the API's max) rather than the 100 default: silently truncating the one
 * chart whose entire point is showing accumulated breadth would misrepresent the very thing it
 * argues about — and since truncation is still possible at the max, `runs.length === TRIAL_LIMIT`
 * triggers an explicit notice rather than a partial cloud passed off as the whole population.
 *
 * At N~70 a dot/strip plot is chosen over a histogram: it shows every individual trial honestly
 * rather than imposing bins. Trial dots get a deterministic vertical jitter (`hashUnit`) purely
 * to keep same-valued trials legible — the y position carries no meaning.
 *
 * COLOUR FOLLOWS THE ENTITY, NEVER ITS ROLE (fix round 3 — the dataviz non-negotiable). This
 * strategy is Electric in BOTH marks: the identity diamond in the cloud and the holdout diamond
 * in the strip. Previously the strip painted the strategy's own diamond from the status palette
 * while Electric went to the deflated BAR — so one card apart, Electric meant "this strategy"
 * and then "the threshold", and this strategy was amber. The bar is a REFERENCE the marks are
 * judged against, not data about a strategy, so it now wears the `--slate` chrome token (the
 * same reasoning that puts `.scatter-diagonal` and `.sparkline-baseline` there).
 *
 * The advisory pass/fail semantics survive on the marker's LABEL, which is verdict-tinted
 * (pass=green, fail=amber, unknown=dim — see `theme.css`'s `.trial-dist-marker-fail` comment:
 * the underlying `holdout_sharpe` check is ALWAYS advisory, so a fail here can never be a
 * breached binding floor and must never render in the red reserved for that). Status ink on a
 * status WORD, identity colour on the mark. The verdict's primary carrier is not colour at all:
 * it is the diamond's position relative to the bar on a zero-anchored scale, plus the margin
 * spelled out as a number in the prose summary. Both marks carry direct text labels — there is
 * no hover on mobile to fall back on.
 *
 * Source for the holdout evidence: this strategy's own latest research-gate run
 * (`/api/runs?strategy=&kind=gate&limit=1` -> `/api/runs/{id}`, the SAME two-step waterfall
 * `GateBulletCard` uses), reading `gate_decision.effective_min_holdout_sharpe` for the bar and
 * the `holdout_sharpe` check's `value`/`passed` for the marker — the same check the bullet card
 * renders, so the two views can never disagree about what "the bar" or "the result" was. The
 * cloud's own marker reads `mean_window_sharpe` off the SAME gate run's list row (a run row
 * always carries that column; no extra request).
 */
export default function TrialDistribution({ strategy }: { strategy: string }) {
  const trialsFetch = useFetch<ApiEnvelope<RunsListPayload>>(
    runsUrl({ kind: 'sweep_trial', limit: TRIAL_LIMIT }),
  )
  const gateListFetch = useFetch<ApiEnvelope<RunsListPayload>>(
    runsUrl({ strategy, kind: 'gate', limit: 1 }),
  )
  const gateRow = gateListFetch.data?.data?.runs?.[0] ?? null
  const gateRunId = gateRow?.id ?? null
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
  const ownMeanWindowSharpe =
    typeof gateRow?.mean_window_sharpe === 'number' ? gateRow.mean_window_sharpe : null

  const cloud = buildTrialCloudGeometry(trials, ownMeanWindowSharpe)
  const strip = buildHoldoutStripGeometry(effectiveMinHoldoutSharpe, ownHoldout)
  const isEmpty = cloud.points.length === 0
  const showStrip = strip.bar !== null || strip.own !== null
  const summary = buildSummary(cloud.points.length, strip)
  const isTruncated = trials.length === TRIAL_LIMIT

  const emptyLabel =
    trials.length === 0
      ? 'no sweep trials recorded yet'
      : `${cloud.excludedCount} sweep trial${cloud.excludedCount === 1 ? '' : 's'} ` +
        'recorded, none with a mean-window-sharpe metric yet'

  const excludedNote =
    cloud.excludedCount > 0
      ? `${cloud.excludedCount} trial${cloud.excludedCount === 1 ? '' : 's'} excluded — ` +
        'missing a mean-window-sharpe metric'
      : null

  const cloudRightEdge = CLOUD_PLOT.width - CLOUD_PLOT.right
  const ownCloudLabelAnchor =
    cloud.own !== null && cloud.own.cx > cloudRightEdge - 100 ? 'end' : 'start'

  const stripRightEdge = STRIP_PLOT.width - STRIP_PLOT.right
  // Anchor buffers are sized to each label's OWN text ("deflated bar N.NNNN" vs the longer
  // "this strategy N.NNNN · advisory" qualifier) so a label near the right edge flips to
  // right-anchored instead of running off the viewBox. Widened in fix round 3 to cover the two
  // extra decimals `fmtHoldout` now prints.
  const barLabelAnchor = strip.bar !== null && strip.bar.cx > stripRightEdge - 105 ? 'end' : 'start'
  const ownStripLabelAnchor =
    strip.own !== null && strip.own.cx > stripRightEdge - 165 ? 'end' : 'start'
  // Push each label a few pixels clear of the mark it names, in whichever direction it runs.
  // Zero-anchoring puts the bar and a near-miss diamond almost on top of each other, so a label
  // starting exactly at `cx` butts straight into the other mark's ink.
  const LABEL_GAP = 5
  const barLabelX =
    strip.bar !== null ? strip.bar.cx + (barLabelAnchor === 'end' ? -LABEL_GAP : LABEL_GAP) : 0
  const ownStripLabelX =
    strip.own !== null
      ? strip.own.cx + (ownStripLabelAnchor === 'end' ? -LABEL_GAP : LABEL_GAP)
      : 0

  return (
    <ChartFrame title="funnel trial distribution" isEmpty={isEmpty} emptyLabel={emptyLabel} height={HEIGHT}>
      <div className="trial-dist-body">
        <svg
          className="trial-dist-cloud-svg"
          data-testid="trial-cloud-svg"
          viewBox={`0 0 ${CLOUD_PLOT.width} ${CLOUD_PLOT.height}`}
          role="img"
          aria-label={
            `funnel-wide search: ${cloud.points.length} trials plotted on mean window sharpe` +
            (cloud.own !== null
              ? `; this strategy's own mean window sharpe ${fmt(cloud.own.value)}`
              : '')
          }
        >
          <rect
            className="trial-dist-plot-area"
            x={CLOUD_PLOT.left}
            y={0}
            width={CLOUD_PLOT.width - CLOUD_PLOT.left - CLOUD_PLOT.right}
            height={CLOUD_AXIS_Y}
          />
          {cloud.points.map((p) => (
            <circle
              key={p.id}
              data-testid="trial-point"
              className="trial-dist-point"
              cx={p.cx}
              cy={p.cy}
              r={CLOUD_POINT_RADIUS}
            />
          ))}
          {cloud.own !== null && (
            <>
              <line
                className="trial-dist-guide"
                x1={cloud.own.cx}
                x2={cloud.own.cx}
                y1={CLOUD_SWARM_TOP}
                y2={CLOUD_OWN_ROW_Y - CLOUD_OWN_RADIUS}
              />
              <polygon
                data-testid="own-cloud-marker"
                className="trial-dist-own-cloud-marker"
                points={[
                  `${cloud.own.cx},${CLOUD_OWN_ROW_Y - CLOUD_OWN_RADIUS}`,
                  `${cloud.own.cx + CLOUD_OWN_RADIUS},${CLOUD_OWN_ROW_Y}`,
                  `${cloud.own.cx},${CLOUD_OWN_ROW_Y + CLOUD_OWN_RADIUS}`,
                  `${cloud.own.cx - CLOUD_OWN_RADIUS},${CLOUD_OWN_ROW_Y}`,
                ].join(' ')}
              />
              <text
                data-testid="own-cloud-marker-label"
                className="trial-dist-own-cloud-marker-label"
                x={cloud.own.cx}
                y={CLOUD_OWN_ROW_Y + CLOUD_OWN_RADIUS + 11}
                textAnchor={ownCloudLabelAnchor}
              >
                this strategy {fmt(cloud.own.value)}
              </text>
            </>
          )}
          <text className="trial-dist-tick" x={CLOUD_PLOT.left} y={CLOUD_TICK_Y}>
            {fmt(cloud.domain.min)}
          </text>
          <text
            className="trial-dist-tick"
            x={cloudRightEdge}
            y={CLOUD_TICK_Y}
            textAnchor="end"
          >
            {fmt(cloud.domain.max)}
          </text>
          <text
            className="trial-dist-caption"
            x={(CLOUD_PLOT.left + cloudRightEdge) / 2}
            y={CLOUD_CAPTION_Y}
            textAnchor="middle"
          >
            mean window sharpe · search population
          </text>
        </svg>
        {showStrip && (
          <svg
            className="trial-dist-strip-svg"
            data-testid="trial-strip-svg"
            viewBox={`0 0 ${STRIP_PLOT.width} ${STRIP_PLOT.height}`}
            role="img"
            aria-label={
              (strip.bar !== null
                ? `deflated bar ${fmtHoldout(strip.bar.value)}`
                : 'no deflated bar recorded') +
              (strip.own !== null
                ? `; this strategy's holdout result ${fmtHoldout(strip.own.value)}, ` +
                  `${strip.own.passed === true ? 'clears the deflated bar' : strip.own.passed === false ? 'below the deflated bar' : 'verdict unknown'}` +
                  // The margin, spoken: a screen-reader listener has even less chance than a
                  // sighted one of subtracting two read-aloud numbers in their head.
                  (strip.bar !== null ? `, ${marginPhrase(strip.own.value, strip.bar.value)}` : '') +
                  ' (advisory check, does not veto the gate)'
                : '')
            }
          >
            <rect
              className="trial-dist-plot-area"
              x={STRIP_PLOT.left}
              y={0}
              width={STRIP_PLOT.width - STRIP_PLOT.left - STRIP_PLOT.right}
              height={STRIP_AXIS_Y}
            />
            {strip.bar !== null && (
              <>
                <line
                  data-testid="bar-line"
                  className="trial-dist-threshold"
                  x1={strip.bar.cx}
                  x2={strip.bar.cx}
                  y1={0}
                  y2={STRIP_AXIS_Y}
                />
                <text
                  data-testid="bar-label"
                  className="trial-dist-threshold-label"
                  x={barLabelX}
                  y={STRIP_BAR_LABEL_Y}
                  textAnchor={barLabelAnchor}
                >
                  deflated bar {fmtHoldout(strip.bar.value)}
                </text>
              </>
            )}
            {strip.own !== null && (
              <>
                <polygon
                  data-testid="own-marker"
                  className="trial-dist-own-strip-marker"
                  points={[
                    `${strip.own.cx},${STRIP_ROW_Y - STRIP_MARKER_RADIUS}`,
                    `${strip.own.cx + STRIP_MARKER_RADIUS},${STRIP_ROW_Y}`,
                    `${strip.own.cx},${STRIP_ROW_Y + STRIP_MARKER_RADIUS}`,
                    `${strip.own.cx - STRIP_MARKER_RADIUS},${STRIP_ROW_Y}`,
                  ].join(' ')}
                />
                <text
                  data-testid="own-marker-label"
                  className={`trial-dist-marker-label ${
                    strip.own.passed === true
                      ? 'trial-dist-marker-pass'
                      : strip.own.passed === false
                        ? 'trial-dist-marker-fail'
                        : 'trial-dist-marker-unknown'
                  }`}
                  x={ownStripLabelX}
                  y={STRIP_OWN_LABEL_Y}
                  textAnchor={ownStripLabelAnchor}
                >
                  this strategy {fmtHoldout(strip.own.value)} · advisory
                </text>
              </>
            )}
            <text className="trial-dist-tick" x={STRIP_PLOT.left} y={STRIP_TICK_Y}>
              {strip.domain !== null ? fmt(strip.domain.min) : ''}
            </text>
            <text
              className="trial-dist-tick"
              x={stripRightEdge}
              y={STRIP_TICK_Y}
              textAnchor="end"
            >
              {strip.domain !== null ? fmt(strip.domain.max) : ''}
            </text>
            <text
              className="trial-dist-caption"
              x={(STRIP_PLOT.left + stripRightEdge) / 2}
              y={STRIP_CAPTION_Y}
              textAnchor="middle"
            >
              holdout sharpe · bar vs result
            </text>
          </svg>
        )}
        {summary !== null && (
          <div className="trial-dist-summary" data-testid="trial-dist-summary">
            {summary}
          </div>
        )}
        {isTruncated && (
          <div className="chart-footnote" data-testid="truncation-notice">
            showing the first {TRIAL_LIMIT} trials — the funnel may hold more; this view can
            understate total breadth
          </div>
        )}
        {excludedNote !== null && <div className="chart-footnote">{excludedNote}</div>}
      </div>
    </ChartFrame>
  )
}
