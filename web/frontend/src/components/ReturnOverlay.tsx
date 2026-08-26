import { useEffect, useMemo, useRef, useState } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { runDetailUrl, runSeriesUrl, runsUrl, useFetch } from '../api'
import { cssColor } from '../chartColors'
import { num, parseUtc, pct } from '../format'
import type {
  ApiEnvelope,
  RunDetail,
  RunRow,
  RunSeriesEntry,
  RunSeriesPayload,
  RunsListPayload,
} from '../types'
import ChartFrame from './ChartFrame'

// ChartFrame's fixed height for THIS chart, whether empty, focus+context, or small multiples —
// small multiples divides this same budget across panels rather than growing the frame, so the
// page never jumps regardless of how many runs a strategy happens to have (see ChartFrame's
// no-layout-shift contract; this is a PLOT, so `variableHeight` is not an option here).
const HEIGHT = 220
const PANEL_GAP = 6
// Spec: "up to ~4 runs overlaid" (2026-08-23 design doc §6). 1-2 render as focus+context on one
// plot; 3+ render as small multiples, capped here so the panel budget never shrinks past
// legibility.
export const MAX_CURVES = 4
const SMALL_MULTIPLES_MIN = 3

// Small-multiples axis budget (fix round 2). uPlot's calcPlotRect only reserves space for an
// axis when `axis.show` is true (uPlot source: `if (axis.show && axis._show)`) — hiding an axis
// fully returns its space, it doesn't just suppress its ticks. uPlot's DEFAULT axis size is 50px
// (both x and y) when not overridden; at MAX_CURVES the per-panel height budget
// (`computeSmallMultiplesPanelHeight`) is far too tight for that default PLUS the explicit 44px
// y-axis this component already used — the arithmetic went negative and the preview's SVG
// stand-in concealed it. Small multiples therefore (a) shows the x-axis on ONLY the last panel —
// the shared time domain makes a per-panel x-axis redundant anyway — and (b) uses compact,
// EXPLICIT sizes for both axes rather than uPlot's 50px default.
export const SMALL_MULT_Y_AXIS_SIZE = 22
export const SMALL_MULT_X_AXIS_SIZE = 16
// The overlay (1-2 curve) case keeps the original, roomier y-axis size — one full-height panel
// has plenty of budget and always shows its own x-axis.
const OVERLAY_Y_AXIS_SIZE = 44

// Fix round 2 (review FIX 7): a fixed strip carved out of the SAME `HEIGHT` budget (never added
// on top of it — the frame's fixed height, and therefore "no layout shift", is unaffected) for
// the caption naming these curves as family siblings. Only reserved when a sibling curve is
// actually plotted (`hasSiblings` below) — this is a per-mount-static condition (a strategy's
// family doesn't change while its detail page is open), not a value that flips after first
// render, so it never reintroduces the shift the fixed `HEIGHT` guards against.
const FAMILY_CAPTION_HEIGHT = 16

function toEpochSeconds(iso: string): number | null {
  const d = parseUtc(iso)
  return d === null ? null : Math.floor(d.getTime() / 1000)
}

function asNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

export interface OverlayCurve {
  runId: number
  /** `'focus'` for the strategy this page is showing, `'context'` for every other curve
   * (siblings in the same family — see `buildReturnOverlayGeometry`'s docstring). This is a
   * per-run IDENTITY, not a mode-derived label: it stays set in small-multiples mode too, so the
   * viewed strategy's own panel keeps reading as `--series-focus` while sibling panels read as
   * `--series-context` — still exactly ONE series per panel (monochrome), just picking between
   * the same two validated tokens rather than inventing a third hue. */
  role: 'focus' | 'context'
  label: string
  /** Epoch seconds, ascending, one per `values` entry. */
  ts: number[]
  /** Cumulative growth from the run's own daily returns, normalized to start at 1.0 — this is
   * the strategy's IN-SAMPLE backtest leg only (`runs series`'s `kind: 'backtest'` entry). */
  values: number[]
}

export interface OverlayRegion {
  /** The run (a `gate`-kind run, in practice) whose holdout leg this region shades. */
  runId: number
  startTs: number
  endTs: number
  /** Composed from the run's OWN scalar OOS metrics (`/api/runs/{id}`) — THE HARD RULE: a
   * shaded interval plus a scalar label is the honest ceiling. Never a plotted per-bar curve. */
  label: string
}

export type OverlayMode = 'empty' | 'overlay' | 'small-multiples'

export interface OverlayGeometry {
  mode: OverlayMode
  curves: OverlayCurve[]
  region: OverlayRegion | null
  /** Backtest runs that were requested but had no plottable series — reported, never plotted
   * as a flat/zero line (the same honesty convention as the scatter's excluded-NULL count). */
  excludedCurveCount: number
}

/** One run's in-sample curve from its `runs series` backtest entry: cumulative growth (start =
 * 1.0) walked forward over the entry's own `[iso_date, daily_return]` pairs, in order — these
 * are REAL per-bar dates (`persist_backtest_returns` stamps the actual index date on every bar),
 * not an interpolation, so the x-axis is exact. A pair whose date does not parse is skipped
 * (that one bar drops out; the rest of the curve still plots) rather than aborting the whole
 * curve. Returns `null` (excluded, not plotted) when the entry is missing, is not a backtest
 * leg, or fewer than 2 bars survive parsing.
 *
 * Labelled by the run's OWN strategy name, not a date: a family has one backtest per member (the
 * common real-world shape — see `buildReturnOverlayGeometry`'s docstring), so a date tells the
 * reader nothing distinguishing while the name is exactly the identity being compared. */
function buildCurve(
  run: RunRow,
  entry: RunSeriesEntry | undefined,
  viewedStrategy: string,
): OverlayCurve | null {
  if (entry === null || entry === undefined || entry.kind !== 'backtest') return null
  const pairs = entry.returns
  if (!Array.isArray(pairs)) return null

  const ts: number[] = []
  const values: number[] = []
  let growth = 1
  for (const pair of pairs) {
    if (!Array.isArray(pair) || pair.length !== 2) continue
    const [dateStr, r] = pair
    const t = toEpochSeconds(dateStr)
    if (t === null) continue
    growth *= 1 + (typeof r === 'number' && Number.isFinite(r) ? r : 0)
    ts.push(t)
    values.push(growth)
  }
  if (ts.length < 2) return null
  return {
    runId: run.id,
    role: run.strategy_name === viewedStrategy ? 'focus' : 'context',
    label: run.strategy_name,
    ts,
    values,
  }
}

/** The gate run's holdout leg -> a shaded interval + scalar label. Reads ONLY `holdout_start`/
 * `holdout_end`/`n_bars` from the series entry (never a `returns` field, which this entry never
 * carries per the backend contract) and the run's own already-fetched scalar metrics for the
 * label. Returns `null` (no region drawn) when there is no gate run, no holdout leg, or the
 * interval does not parse.
 *
 * `viewedStrategy` is named IN the label (fix round 2) — the region always belongs to the page's
 * own strategy (its gate run is fetched via `?strategy=&kind=gate`), but in overlay mode the band
 * is drawn full-width across a plot that ALSO contains a family sibling's curve. A bare metrics
 * string ("63 bars OOS · sharpe 0.5") reads as if it could apply to either line; naming the
 * strategy closes that. */
function buildRegion(
  gateRun: RunRow | null,
  entry: RunSeriesEntry | undefined,
  detail: RunDetail | null,
  viewedStrategy: string,
): OverlayRegion | null {
  if (gateRun === null || entry === null || entry === undefined || entry.kind !== 'holdout') {
    return null
  }
  const startTs = toEpochSeconds(entry.holdout_start)
  const endTs = toEpochSeconds(entry.holdout_end)
  if (startTs === null || endTs === null || endTs <= startTs) return null

  const bits: string[] = [`${entry.n_bars} bars OOS`]
  const sharpeOos = detail !== null ? asNum(detail.sharpe_oos) : null
  const totalReturnOos = detail !== null ? asNum(detail.total_return_oos) : null
  if (sharpeOos !== null) bits.push(`sharpe ${num(sharpeOos, 3)}`)
  if (totalReturnOos !== null) bits.push(`return ${pct(totalReturnOos)}`)

  return { runId: gateRun.id, startTs, endTs, label: `${viewedStrategy} holdout: ${bits.join(' · ')}` }
}

/**
 * Pure layout: a set of `backtest`-kind runs (this strategy's own PLUS its family siblings',
 * when it has a family — see `ReturnOverlay`'s docstring for why family membership is the source
 * of "up to ~4 runs" rather than one strategy's own history) + their `runs series` entries + this
 * strategy's latest gate run's holdout leg -> the geometry `ReturnOverlay` renders. Isolated from
 * fetching (like `buildTrialCloudGeometry`/`buildBulletGeometry`) so THE HARD RULE — a
 * holdout leg becomes an interval, never an array — is a property of this one function, directly
 * unit-testable without a DOM or a canvas.
 *
 * `viewedStrategy` decides ROLE, not count: the curve whose run belongs to the page's own
 * strategy is `'focus'`; every other curve (a family sibling) is `'context'`. Mode is decided
 * purely by how many plottable curves there are — 1-2 -> `'overlay'` (both direct-labelled at
 * the line end), 3+ -> `'small-multiples'` (one series per panel, still identity-colored by
 * role) — independent of which curve happens to be the focus.
 */
export function buildReturnOverlayGeometry(
  backtestRuns: RunRow[],
  seriesById: Record<string, RunSeriesEntry>,
  gateRun: RunRow | null,
  gateDetail: RunDetail | null,
  viewedStrategy: string,
): OverlayGeometry {
  const built = backtestRuns.map((r) => buildCurve(r, seriesById[String(r.id)], viewedStrategy))
  const curves = built.filter((c): c is OverlayCurve => c !== null)
  const excludedCurveCount = backtestRuns.length - curves.length

  const region = buildRegion(
    gateRun,
    gateRun !== null ? seriesById[String(gateRun.id)] : undefined,
    gateDetail,
    viewedStrategy,
  )

  if (curves.length === 0) {
    return { mode: 'empty', curves: [], region: null, excludedCurveCount }
  }
  const mode: OverlayMode = curves.length >= SMALL_MULTIPLES_MIN ? 'small-multiples' : 'overlay'
  return { mode, curves, region, excludedCurveCount }
}

/** uPlot requires every series to share ONE x array. Curves come from independently-dated
 * backtest runs, so this builds the UNION of every curve's timestamps and maps each curve onto
 * it, leaving `null` (a real gap uPlot draws as a break) wherever a curve has no point at that
 * timestamp — never a fabricated 0, which would draw a false crash to the axis. Exported for a
 * direct unit test of the null-gap (never zero-fill) property. */
export function buildAlignedData(curves: OverlayCurve[]): uPlot.AlignedData {
  const tsSet = new Set<number>()
  for (const c of curves) for (const t of c.ts) tsSet.add(t)
  const xs = [...tsSet].sort((a, b) => a - b)
  const series = curves.map((c) => {
    const m = new Map(c.ts.map((t, i) => [t, c.values[i]]))
    return xs.map((t) => m.get(t) ?? null)
  })
  return [xs, ...series] as uPlot.AlignedData
}

function fracOf(v: number, min: number, max: number): number {
  return max === min ? 0.5 : (v - min) / (max - min)
}

/** Pure arithmetic (fix round 2): the small-multiples per-panel height, given `n` curves and the
 * plot budget available (defaults to the full `HEIGHT` — the value every existing call site and
 * test uses; `ReturnOverlay` passes a slightly smaller budget when the family caption is shown,
 * see its docstring) — isolated from `ReturnOverlay` so the "does the axis budget still leave
 * usable plot area at the cap" question is directly unit-testable without a canvas (uPlot
 * doesn't render in jsdom). */
export function computeSmallMultiplesPanelHeight(n: number, budget: number = HEIGHT): number {
  return Math.max(40, Math.floor((budget - PANEL_GAP * (n - 1)) / Math.max(n, 1)))
}

/** One plot: 1-2 curves (the overlay case) or exactly 1 curve (one small-multiples panel).
 * Draws the real uPlot canvas line(s) PLUS a CSS-positioned end-label per curve and, when a
 * region is given, a CSS-positioned shaded band + label — all computed from the SAME domain
 * uPlot is told to use (`scales.x/y.range` fixed to `xDomain`/`yDomain` below), so the DOM
 * overlay and the canvas line always agree on where things sit. */
function Panel({
  curves,
  region,
  height,
  showXAxis = true,
  yAxisSize = OVERLAY_Y_AXIS_SIZE,
  xAxisSize,
}: {
  curves: OverlayCurve[]
  region: OverlayRegion | null
  height: number
  /** Small multiples shows the x-axis on only the LAST panel (fix round 2) — see the module
   * docstring on `SMALL_MULT_Y_AXIS_SIZE`. The single-panel overlay case always shows it. */
  showXAxis?: boolean
  yAxisSize?: number
  /** `undefined` keeps uPlot's own default (50px) — correct for the overlay case's one
   * full-height panel. Small multiples passes `SMALL_MULT_X_AXIS_SIZE` explicitly. */
  xAxisSize?: number
}) {
  const allValues = curves.flatMap((c) => c.values)
  let yMin = Math.min(...allValues)
  let yMax = Math.max(...allValues)
  if (yMin === yMax) {
    yMin -= 0.01
    yMax += 0.01
  }
  const ySpan = yMax - yMin
  yMin -= ySpan * 0.08
  yMax += ySpan * 0.08

  const allTs = curves.flatMap((c) => c.ts)
  let xMin = Math.min(...allTs)
  let xMax = Math.max(...allTs)
  if (region !== null) {
    xMin = Math.min(xMin, region.startTs)
    xMax = Math.max(xMax, region.endTs)
  }

  return (
    <div className="overlay-panel-inner" style={{ height }}>
      <Plot
        curves={curves}
        xDomain={[xMin, xMax]}
        yDomain={[yMin, yMax]}
        height={height}
        showXAxis={showXAxis}
        yAxisSize={yAxisSize}
        xAxisSize={xAxisSize}
      />
      {region !== null && (
        <div
          className="overlay-region-band"
          data-testid="overlay-region"
          style={{
            left: `${fracOf(region.startTs, xMin, xMax) * 100}%`,
            width: `${(fracOf(region.endTs, xMin, xMax) - fracOf(region.startTs, xMin, xMax)) * 100}%`,
          }}
        >
          <span className="overlay-region-label" data-testid="overlay-region-label">
            {region.label}
          </span>
        </div>
      )}
      {curves.map((c) => (
        <div
          key={c.runId}
          className="overlay-end-label"
          data-testid="overlay-end-label"
          data-role={c.role ?? undefined}
          style={{ top: `${(1 - fracOf(c.values[c.values.length - 1], yMin, yMax)) * 100}%` }}
        >
          {c.label}
        </div>
      ))}
    </div>
  )
}

function Plot({
  curves,
  xDomain,
  yDomain,
  height,
  showXAxis = true,
  yAxisSize = OVERLAY_Y_AXIS_SIZE,
  xAxisSize,
}: {
  curves: OverlayCurve[]
  xDomain: [number, number]
  yDomain: [number, number]
  height: number
  showXAxis?: boolean
  yAxisSize?: number
  xAxisSize?: number
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const el = hostRef.current
    if (!el) return
    setWidth(Math.floor(el.clientWidth))
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver((entries) => {
      const w = Math.floor(entries[0].contentRect.width)
      setWidth((prev) => (w !== prev ? w : prev))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const el = hostRef.current
    if (!el || width < 40 || curves.length === 0) return
    const text = cssColor('--text-dim', '#a9b4c5')
    const line = cssColor('--line', 'rgba(220,228,240,0.08)')
    const axis = {
      stroke: text,
      grid: { stroke: line, width: 1 },
      ticks: { stroke: line, width: 1 },
      font: '10px "IBM Plex Mono", monospace',
    }
    const [xMin, xMax] = xDomain
    const [yMin, yMax] = yDomain
    const u = new uPlot(
      {
        width,
        height,
        tzDate: (ts) => uPlot.tzDate(new Date(ts * 1000), 'Etc/UTC'),
        legend: { show: false },
        cursor: { drag: { x: false, y: false } },
        scales: {
          x: { range: () => [xMin, xMax] },
          y: { range: () => [yMin, yMax] },
        },
        series: [
          {},
          ...curves.map((c) => ({
            label: c.label,
            // `--series-focus`/`--series-context`: the validated 2-token palette (theme.css).
            // Small multiples pass exactly ONE curve per `Plot` call, so each panel is still
            // genuinely monochrome — the token picked just depends on whose panel it is.
            stroke:
              c.role === 'context'
                ? cssColor('--series-context', '#727c8b')
                : cssColor('--series-focus', '#3982ff'),
            width: c.role === 'context' ? 1 : 1.5,
            points: { show: false },
          })),
        ],
        axes: [
          { ...axis, show: showXAxis, ...(xAxisSize !== undefined ? { size: xAxisSize } : {}) },
          { ...axis, size: yAxisSize },
        ],
      },
      buildAlignedData(curves),
      el,
    )
    return () => u.destroy()
  }, [curves, xDomain, yDomain, width, height, showXAxis, yAxisSize, xAxisSize])

  return <div ref={hostRef} className="chart-host overlay-plot-host" style={{ minHeight: height }} />
}

/**
 * View 4 — the return overlay (spec §6.1, task 7). THE HARD RULE this view carries: `runs
 * series` returns a holdout leg's INTERVAL and `n_bars` only, never a per-bar vector
 * (`holdout_returns.returns_blob` is SENSITIVE — see `algua/registry/run_views.py`'s docstring
 * and `algua/registry/db/holdout.py`'s DDL comment). A run with a holdout leg therefore gets a
 * shaded region + its own scalar OOS metrics (`/api/runs/{id}`) — region + scalar label is the
 * honest ceiling, never a plotted OOS curve.
 *
 * WHERE "up to ~4 runs to overlay" comes from: a single strategy typically has exactly ONE
 * persisted backtest curve (`record_backtest_run` is called once per evaluation, not on a
 * schedule) — there is nothing of its own to compare against. What the spec's plural "runs"
 * means in practice is this strategy set against its FAMILY: `registry list --family` groups
 * strategies exploring the same hypothesis space, so overlaying their backtest curves is the one
 * multi-run comparison that is both available on a single-strategy page and actually meaningful.
 * `family` is passed in from `StrategyDetail`'s already-fetched registry record (never a second
 * lookup). With no family, this strategy's own curve renders alone (0-1 curves, no context line).
 *
 * Data: `/api/runs?family=&kind=backtest&limit=4` (or `?strategy=&kind=backtest&limit=1` with no
 * family) for the curve(s), `/api/runs/series` for their series (ONE batched request for every
 * id), and THIS strategy's own latest `gate`-kind run for the holdout region — a sibling's gate
 * evidence is never shown on another strategy's page. 1-2 curves render focus+context on one
 * uPlot plot, both direct-labelled at the line end (no legend — the mobile rule forbids hover).
 * 3+ render as small multiples, stacked (never side by side), one monochrome series per panel —
 * the ONLY legal way to show more than 2 series on this brand (the dataviz validator fails a
 * 3rd/4th neutral step; see theme.css's series-palette comment).
 */
export default function ReturnOverlay({
  strategy,
  family,
}: {
  strategy: string
  family: string | null
}) {
  const backtestFetch = useFetch<ApiEnvelope<RunsListPayload>>(
    family !== null
      ? runsUrl({ family, kind: 'backtest', limit: MAX_CURVES })
      : runsUrl({ strategy, kind: 'backtest', limit: 1 }),
  )
  const gateListFetch = useFetch<ApiEnvelope<RunsListPayload>>(
    runsUrl({ strategy, kind: 'gate', limit: 1 }),
  )
  const backtestRuns = useMemo(
    () => backtestFetch.data?.data?.runs ?? [],
    [backtestFetch.data],
  )
  const gateRun = gateListFetch.data?.data?.runs?.[0] ?? null

  const seriesIds = useMemo(() => {
    const ids = backtestRuns.map((r) => r.id)
    if (gateRun !== null) ids.push(gateRun.id)
    return ids
  }, [backtestRuns, gateRun])

  const seriesFetch = useFetch<ApiEnvelope<RunSeriesPayload>>(
    seriesIds.length > 0 ? runSeriesUrl(seriesIds) : null,
  )
  const gateDetailFetch = useFetch<ApiEnvelope<RunDetail>>(
    gateRun !== null ? runDetailUrl(gateRun.id) : null,
  )

  const seriesById = seriesFetch.data?.data?.series ?? {}
  const gateDetail = gateDetailFetch.data?.data ?? null

  const geometry = useMemo(
    () => buildReturnOverlayGeometry(backtestRuns, seriesById, gateRun, gateDetail, strategy),
    [backtestRuns, seriesById, gateRun, gateDetail, strategy],
  )

  const isEmpty = geometry.mode === 'empty'
  const emptyLabel =
    backtestRuns.length === 0
      ? 'no backtest runs recorded yet'
      : `${geometry.excludedCurveCount} backtest run${geometry.excludedCurveCount === 1 ? '' : 's'} ` +
        'recorded, none with a persisted return series yet'

  // Fix round 2: nothing on screen said these were FAMILY SIBLINGS, not this strategy's own
  // history — the title just says "return overlay" and the only signal was a strategy name at
  // each line end. `hasSiblings` is a real 'context'-role curve, not just `family !== null`
  // (a lone strategy in its family still renders with zero siblings plotted).
  const hasSiblings = geometry.curves.some((c) => c.role === 'context')
  const plotBudget = hasSiblings ? HEIGHT - FAMILY_CAPTION_HEIGHT : HEIGHT

  const n = geometry.curves.length
  const panelHeight = computeSmallMultiplesPanelHeight(n, plotBudget)

  return (
    <ChartFrame title="return overlay" isEmpty={isEmpty} emptyLabel={emptyLabel} height={HEIGHT}>
      {hasSiblings && (
        <div className="overlay-family-caption" data-testid="overlay-family-caption">
          {family} family — {n - 1} sibling curve{n - 1 === 1 ? '' : 's'} alongside {strategy}&apos;s own
        </div>
      )}
      {geometry.mode === 'overlay' ? (
        <Panel curves={geometry.curves} region={geometry.region} height={plotBudget} />
      ) : (
        <div className="overlay-small-multiples">
          {geometry.curves.map((c, i) => (
            <div
              key={c.runId}
              data-testid="overlay-panel"
              style={{ height: panelHeight, marginBottom: i < n - 1 ? PANEL_GAP : 0 }}
            >
              {/* The region belongs to the VIEWED strategy's own gate evaluation, so it must
                  land on the FOCUS panel specifically — not whichever curve the family query
                  happened to list first (newest-first order, unrelated to role). */}
              <Panel
                curves={[c]}
                region={c.role === 'focus' ? geometry.region : null}
                height={panelHeight}
                showXAxis={i === n - 1}
                yAxisSize={SMALL_MULT_Y_AXIS_SIZE}
                xAxisSize={SMALL_MULT_X_AXIS_SIZE}
              />
            </div>
          ))}
        </div>
      )}
    </ChartFrame>
  )
}
