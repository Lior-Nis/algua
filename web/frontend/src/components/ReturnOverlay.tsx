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
// Raised from 220 in fix round 3 (review FIX E). At the `MAX_CURVES = 4` cap the old budget left
// each panel 46px, and 46 minus the two explicit axis sizes is 8px of plot — a sliver, below the
// "usable" floor the small-multiples test asserts. Fixing the test to exercise the PRODUCTION
// budget (`SMALL_MULTIPLES_PLOT_BUDGET`, not the 220 default it used to pass) made that
// arithmetic fail honestly, so the budget itself had to grow. 260 leaves 53px panels and 15px of
// plot area on the one panel carrying both axes. Deliberately ONE height for every mode —
// empty, overlay and small multiples alike — because mode is only known after the fetch
// resolves, so a mode-dependent frame height would reintroduce exactly the layout shift
// `ChartFrame`'s fixed-height contract exists to prevent.
export const HEIGHT = 260
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

// A fixed strip carved out of the SAME `HEIGHT` budget (never added on top of it — the frame's
// fixed height, and therefore "no layout shift", is unaffected) for the caption naming these
// curves as family siblings and attributing the shaded holdout band. Only reserved when a
// sibling curve is actually plotted — a per-mount-static condition (a strategy's family doesn't
// change while its detail page is open), not a value that flips after first render.
//
// TWO LINES, and CSS ENFORCES IT (fix round 3, review FIX F). This was 16px — one line — and
// nothing in `theme.css` backed that number: the caption was free-flowing text that wrapped to
// two lines at 414px, pushing the panels past the fixed `HEIGHT` and breaking the very
// no-layout-shift contract the arithmetic was written to honour. `.overlay-family-caption` now
// carries `height: 30px; line-height: 15px` with a 2-line clamp, so the rendered height and this
// constant are the same number by construction rather than by hope. KEEP THE TWO IN SYNC.
export const FAMILY_CAPTION_HEIGHT = 30

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
 * THE LABEL IS SCALARS ONLY. Fix round 2 prefixed it with the viewed strategy's name to stop the
 * band reading as if it covered a sibling's curve; fix round 3 takes that back out, because the
 * label lives inside a NARROW band in a `nowrap` span — at 390/414px "trend_breakout_v1" filled
 * the whole band and clipped at the card edge, so "63 bars OOS · sharpe …" — the numbers the
 * label exists to show — became invisible. The band is attributed in `.overlay-family-caption`
 * instead ("shaded band = its holdout"), which has the full card width to say it in. */
function buildRegion(
  gateRun: RunRow | null,
  entry: RunSeriesEntry | undefined,
  detail: RunDetail | null,
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

  return { runId: gateRun.id, startTs, endTs, label: bits.join(' · ') }
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

/** THE budget small multiples actually runs on, and the only one it ever runs on.
 *
 * Small multiples needs 3+ curves, and at most one curve can be the focus — so 3+ curves ALWAYS
 * means 2+ siblings, which always means the family caption is shown, which always means the
 * caption's strip is carved out of `HEIGHT`. Exported so the panel-arithmetic test consumes the
 * production number rather than inventing its own. */
export const SMALL_MULTIPLES_PLOT_BUDGET = HEIGHT - FAMILY_CAPTION_HEIGHT

/** Pure arithmetic (fix round 2): the small-multiples per-panel height, given `n` curves and the
 * plot budget available — isolated from `ReturnOverlay` so the "does the axis budget still leave
 * usable plot area at the cap" question is directly unit-testable without a canvas (uPlot doesn't
 * render in jsdom).
 *
 * `budget` is REQUIRED (fix round 3, review FIX E). It used to default to the full `HEIGHT`, a
 * value production never passes here — and the test that was supposed to prove the axis
 * arithmetic leaves usable plot area at `MAX_CURVES` took the default, so it passed against a
 * budget 40px larger than the real one while the real one left an 8px sliver. A default nobody
 * uses is a trap that makes a test agree with itself; there is now no default to take. */
export function computeSmallMultiplesPanelHeight(n: number, budget: number): number {
  return Math.max(40, Math.floor((budget - PANEL_GAP * (n - 1)) / Math.max(n, 1)))
}

/** The caption naming these curves as family siblings and attributing the shaded holdout band.
 *
 * EVERY INPUT IS DERIVED FROM WHAT IS ACTUALLY PLOTTED (fix round 3, review FIX F), never from
 * the fetched run list. `focusPlotted` and `siblingCount` used to be `curves.length - 1` and a
 * bare `family !== null`, both of which quietly assume the viewed strategy contributed a curve.
 * When its backtest has no persisted series while its siblings' do, it contributes none — and
 * the old caption then claimed the siblings ran "alongside {strategy}'s own" curve that is not
 * on the plot, and undercounted the siblings by one. Same for the band: it is only drawn in
 * small-multiples mode when there IS a focus panel to draw it on, so the caption must not
 * promise it otherwise. */
export function buildFamilyCaption(
  family: string,
  siblingCount: number,
  viewedStrategy: string,
  focusPlotted: boolean,
  bandDrawn: boolean,
): string {
  const curveWord = `${siblingCount} sibling curve${siblingCount === 1 ? '' : 's'}`
  const parts = [
    `${family} family`,
    focusPlotted
      ? `${curveWord} alongside ${viewedStrategy}`
      : `${curveWord}; ${viewedStrategy} has no plotted curve`,
  ]
  // "its" — the viewed strategy was just named in the preceding clause either way. Naming it a
  // second time is what pushed this caption past two lines.
  if (bandDrawn) parts.push('shaded band = its holdout')
  return parts.join(' · ')
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
          {/* A holdout band is by construction the TAIL of the period, so it sits at the right of
              the plot — and a `nowrap` label pinned to the band's left edge then grows straight
              off the card. It hangs off the band either way (the band is far narrower than the
              label), so the only question is WHICH way: past the right edge into nothing, or
              leftward across the plot the reader is already looking at. Anchored to whichever
              side leaves room. */}
          <span
            className="overlay-region-label"
            data-testid="overlay-region-label"
            style={
              fracOf(region.endTs, xMin, xMax) > 0.5
                ? { left: 'auto', right: 4 }
                : undefined
            }
          >
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
  // each line end. Fix round 3: every one of these is read off the PLOTTED curves, never off the
  // fetched run list — see `buildFamilyCaption`.
  const focusPlotted = geometry.curves.some((c) => c.role === 'focus')
  const siblingCount = geometry.curves.filter((c) => c.role === 'context').length
  const showCaption = family !== null && siblingCount > 0
  // In overlay mode the band spans the whole plot; in small multiples it is handed only to the
  // focus panel, so with no focus curve plotted it is never drawn at all.
  const bandDrawn =
    geometry.region !== null && (geometry.mode === 'overlay' || focusPlotted)

  const n = geometry.curves.length
  const overlayBudget = showCaption ? HEIGHT - FAMILY_CAPTION_HEIGHT : HEIGHT
  // 3+ curves implies 2+ siblings implies the caption is shown, so this IS `overlayBudget` in
  // small-multiples mode — spelled as the exported constant so production and the panel-
  // arithmetic test provably consume the same number.
  const panelHeight = computeSmallMultiplesPanelHeight(n, SMALL_MULTIPLES_PLOT_BUDGET)

  return (
    <ChartFrame title="return overlay" isEmpty={isEmpty} emptyLabel={emptyLabel} height={HEIGHT}>
      {showCaption && (
        <div className="overlay-family-caption" data-testid="overlay-family-caption">
          {buildFamilyCaption(family, siblingCount, strategy, focusPlotted, bandDrawn)}
        </div>
      )}
      {geometry.mode === 'overlay' ? (
        <Panel curves={geometry.curves} region={geometry.region} height={overlayBudget} />
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
