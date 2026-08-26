const DEFAULT_WIDTH = 56
const DEFAULT_HEIGHT = 20
const PADDING = 2

export interface SparklinePoint {
  x: number
  y: number
}

export interface SparklineGeometry {
  /** One array per CONTINUOUS run of finite values. A NULL/undefined entry breaks the run — it
   * is a gap, never interpolated through as a fabricated value (e.g. 0). */
  segments: SparklinePoint[][]
  /** True when fewer than two finite values exist — a single point cannot show a trend, so
   * nothing is drawn rather than a misleading dot or flat line. */
  isEmpty: boolean
  /** Y coordinate of the zero baseline in this geometry's coordinate space. Meaningless when
   * `isEmpty` is true (no baseline is drawn in that case) — otherwise always defined, because
   * the y-domain always includes zero (see `buildSparklineGeometry`). */
  zeroY: number
}

function isFiniteMetric(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

/** Pure layout: a scalar sequence -> SVG-space points, gapped at NULLs. Mirrors
 * `buildScatterGeometry`'s discipline (ScatterISOOS.tsx) — a missing measurement is excluded,
 * never coerced to a plottable number.
 *
 * The y-domain ALWAYS includes zero. Self-scaling each row independently (min/max of ITS OWN
 * finite values) is deliberate — a shared cross-row scale would flatten every other row into a
 * flat line the moment one row has a wide range, losing the one thing this mark is for: shape.
 * But self-scaling alone normalises away magnitude entirely, so `1.0 -> 0.9 -> 0.85` (a mild
 * decline) and `1.0 -> 0.9 -> -3.0` (a crater through the one threshold that actually matters
 * for a Sharpe figure — crossing zero) would draw IDENTICALLY. Anchoring the domain to zero
 * fixes that without reintroducing a shared scale: a mild decline that never reaches zero stays
 * high in the band, while a row that actually craters visibly crosses the baseline drawn at
 * `zeroY`. */
export function buildSparklineGeometry(
  values: ReadonlyArray<number | null | undefined>,
  width: number = DEFAULT_WIDTH,
  height: number = DEFAULT_HEIGHT,
  padding: number = PADDING,
): SparklineGeometry {
  const finiteValues = values.filter(isFiniteMetric)
  if (finiteValues.length < 2) return { segments: [], isEmpty: true, zeroY: 0 }

  let min = Math.min(...finiteValues)
  let max = Math.max(...finiteValues)
  // Always include zero in the domain — see the function docstring.
  min = Math.min(min, 0)
  max = Math.max(max, 0)
  if (min === max) {
    // Every finite value (and zero) coincide — pad symmetrically so the scale never divides by
    // zero; the line renders as a flat horizontal run through the baseline, the honest shape.
    min -= 0.5
    max += 0.5
  }
  const range = max - min
  const n = values.length
  const xStep = n > 1 ? (width - padding * 2) / (n - 1) : 0
  const scaleY = (v: number) => padding + (1 - (v - min) / range) * (height - padding * 2)

  const segments: SparklinePoint[][] = []
  let current: SparklinePoint[] = []
  values.forEach((v, i) => {
    if (isFiniteMetric(v)) {
      current.push({ x: padding + i * xStep, y: scaleY(v) })
    } else if (current.length > 0) {
      segments.push(current)
      current = []
    }
  })
  if (current.length > 0) segments.push(current)
  return { segments, isEmpty: false, zeroY: scaleY(0) }
}

/**
 * One-series inline SVG micro-chart for a run-list row. Deliberately monochrome — task 4's
 * brief: "one series per row needs no palette, and no categorical hue may be introduced." Uses
 * `--series-context`, the neutral series token (theme.css) — never `--series-focus`/Electric,
 * which is reserved for the ONE active thing on a screen; with one sparkline per row, none of
 * them is that. The zero baseline is chrome, not a series — `--line-2`, the same token
 * `ScatterISOOS`'s reference diagonal uses, for the same reason (a reference the marks are
 * judged against, not data itself).
 *
 * NOT a time series (see `RunList.tsx`'s caption, and this component's `label` prop) — the three
 * points are `[in-sample worst window, in-sample mean window, out-of-sample result]`, a
 * degradation profile, not evenly-spaced observations over time. The `aria-label` says so
 * explicitly rather than defaulting to "trend", which would assert exactly the reading this
 * component exists to avoid.
 *
 * Renders NOTHING drawn (no polyline, no baseline) when fewer than two finite values exist — the
 * honest-empty discipline applied to a per-row mark rather than a full chart panel (ChartFrame
 * governs the panel-level case; this is the equivalent for a decoration inside a row). The
 * `<svg>` box itself is always emitted at a fixed size so a row's height never shifts between a
 * populated and an empty sparkline.
 */
export default function Sparkline({
  values,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  label,
}: {
  values: ReadonlyArray<number | null | undefined>
  width?: number
  height?: number
  /** Accessible description of what this specific sparkline plots. Callers that give the mark a
   * specific meaning (e.g. RunList's degradation profile) should pass one; the default is
   * generic but still never claims a time series. */
  label?: string
}) {
  const { segments, isEmpty, zeroY } = buildSparklineGeometry(values, width, height)
  return (
    <svg
      data-testid="sparkline"
      data-empty={isEmpty ? 'true' : 'false'}
      className="sparkline"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={
        isEmpty ? 'not enough data to plot' : (label ?? 'sequence of values, relative to zero')
      }
    >
      {!isEmpty && (
        <line
          className="sparkline-baseline"
          x1={PADDING}
          x2={width - PADDING}
          y1={zeroY}
          y2={zeroY}
        />
      )}
      {segments.map((seg, i) => (
        <polyline
          key={i}
          className="sparkline-line"
          points={seg.map((p) => `${p.x},${p.y}`).join(' ')}
        />
      ))}
    </svg>
  )
}
