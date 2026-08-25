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
}

function isFiniteMetric(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

/** Pure layout: a scalar sequence -> SVG-space points, gapped at NULLs. Mirrors
 * `buildScatterGeometry`'s discipline (ScatterISOOS.tsx) — a missing measurement is excluded,
 * never coerced to a plottable number. */
export function buildSparklineGeometry(
  values: ReadonlyArray<number | null | undefined>,
  width: number = DEFAULT_WIDTH,
  height: number = DEFAULT_HEIGHT,
  padding: number = PADDING,
): SparklineGeometry {
  const finiteValues = values.filter(isFiniteMetric)
  if (finiteValues.length < 2) return { segments: [], isEmpty: true }

  let min = Math.min(...finiteValues)
  let max = Math.max(...finiteValues)
  if (min === max) {
    // A flat series (every finite value tied) — pad symmetrically so the scale never divides
    // by zero; the line renders as a flat horizontal run, which is the honest shape.
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
  return { segments, isEmpty: false }
}

/**
 * One-series inline SVG micro-chart for a run-list row. Deliberately monochrome — task 4's
 * brief: "one series per row needs no palette, and no categorical hue may be introduced." Uses
 * `--series-context`, the neutral series token (theme.css) — never `--series-focus`/Electric,
 * which is reserved for the ONE active thing on a screen; with one sparkline per row, none of
 * them is that.
 *
 * Renders NOTHING drawn (no polyline) when fewer than two finite values exist — the honest-empty
 * discipline applied to a per-row mark rather than a full chart panel (ChartFrame governs the
 * panel-level case; this is the equivalent for a decoration inside a row). The `<svg>` box
 * itself is always emitted at a fixed size so a row's height never shifts between a populated
 * and an empty sparkline.
 */
export default function Sparkline({
  values,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
}: {
  values: ReadonlyArray<number | null | undefined>
  width?: number
  height?: number
}) {
  const { segments, isEmpty } = buildSparklineGeometry(values, width, height)
  return (
    <svg
      data-testid="sparkline"
      data-empty={isEmpty ? 'true' : 'false'}
      className="sparkline"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={isEmpty ? 'not enough data for a trend' : 'trend'}
    >
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
