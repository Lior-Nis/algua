import { useEffect, useMemo, useRef, useState } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { cssColor } from '../chartColors'
import { parseUtc } from '../format'
import type { Lane, SeriesPayload, SeriesRow } from '../types'
import EvidenceStatus from './EvidenceStatus'

const HEIGHT = 220
const LANES: Lane[] = ['live', 'paper']

interface Normalized {
  ts: number[]
  equity: (number | null)[]
  peak: (number | null)[]
}

/** Parse→sort→dedupe one lane's rows into aligned uPlot arrays.
 * - tick_ts parsed as UTC epoch seconds (unparseable rows skipped)
 * - sorted ascending by (ts, id)
 * - repeated timestamps deduped keeping the LAST by id
 * - peak_equity may be null (uPlot accepts null gaps) */
function normalize(rows: SeriesRow[]): Normalized {
  const parsed: { t: number; row: SeriesRow }[] = []
  for (const row of rows) {
    const d = parseUtc(row.tick_ts)
    if (d === null) continue
    parsed.push({ t: Math.floor(d.getTime() / 1000), row })
  }
  parsed.sort((a, b) => a.t - b.t || a.row.id - b.row.id)
  const byTs = new Map<number, SeriesRow>()
  for (const e of parsed) {
    const prev = byTs.get(e.t)
    if (prev === undefined || e.row.id >= prev.id) byTs.set(e.t, e.row)
  }
  const ts = [...byTs.keys()].sort((a, b) => a - b)
  return {
    ts,
    equity: ts.map((t) => byTs.get(t)!.equity ?? null),
    peak: ts.map((t) => byTs.get(t)!.peak_equity ?? null),
  }
}

export function drawableSeriesPointCount(rows: SeriesRow[] | null): number {
  return rows === null ? 0 : normalize(rows).ts.length
}

export default function EquityChart({
  payload,
  loading = false,
  error = null,
  stale = false,
  onRetry,
}: {
  payload?: SeriesPayload
  loading?: boolean
  error?: Error | null
  stale?: boolean
  onRetry?: () => void
}) {
  const series = payload?.series ?? { paper: null, live: null }
  const normalized = useMemo(() => ({
    live: normalize(series.live ?? []),
    paper: normalize(series.paper ?? []),
  }), [series.live, series.paper])
  // A lane is available only when it has at least one point that can actually be plotted. Raw
  // rows with invalid timestamps must not steal selection from a drawable lane.
  const available = LANES.filter((l) => normalized[l].ts.length > 0)
  const showPicker = available.length > 1

  const [lane, setLane] = useState<Lane | null>(null)
  const firstPresentLane = LANES.find((candidate) => series[candidate] !== null) ?? 'paper'
  const activeLane: Lane = lane !== null && available.includes(lane)
    ? lane
    : (available[0] ?? firstPresentLane)

  const data = normalized[activeLane]
  const cached = payload !== undefined && (stale || error !== null)

  const footnotes: string[] = []
  // Truncation MUST be visible: `fleet series` keeps only the newest N per lane, so a
  // truncated curve's left edge is a cut, not the strategy's first tick. Silently
  // clipping it reads as "this is the whole history".
  if (payload?.truncated[activeLane] === true) footnotes.push('newest ticks only (truncated)')
  if ((payload?.n_legacy_excluded ?? 0) > 0) footnotes.push(`${payload!.n_legacy_excluded} legacy excluded`)
  if ((payload?.n_unparseable ?? 0) > 0) footnotes.push(`${payload!.n_unparseable} unparseable`)
  if ((payload?.n_invalid_lane ?? 0) > 0) footnotes.push(`${payload!.n_invalid_lane} invalid lane`)

  let status: React.ReactNode = null
  if (payload === undefined) {
    status = loading
      ? <EvidenceStatus title="equity" state="loading" detail="loading tick history" />
      : <EvidenceStatus title="equity" state="failed" detail={error?.message ?? 'tick history unavailable'} onRetry={onRetry} />
  } else if ((series.paper?.length ?? 0) + (series.live?.length ?? 0) === 0) {
    status = (
      <EvidenceStatus
        title="equity"
        state={cached ? 'stale' : 'empty'}
        detail={cached ? 'cached tick history is empty — no ticks were recorded' : 'no tick history recorded'}
        onRetry={cached ? onRetry : undefined}
      />
    )
  } else if (data.ts.length < 2) {
    status = (
      <EvidenceStatus
        title={`equity — ${activeLane}`}
        state={cached ? 'stale' : 'incomplete'}
        detail={`${cached ? 'cached tick history is incomplete — ' : ''}fewer than two valid ticks; no honest curve can be drawn`}
        onRetry={cached ? onRetry : undefined}
      />
    )
  }

  return (
    <section className="panel chart-panel">
      <div className="chart-head">
        <span className="micro-label">equity{showPicker ? '' : ` — ${activeLane}`}</span>
        {showPicker && (
          <div className="seg-control" role="group" aria-label="lane">
            {available.map((l) => (
              <button
                key={l}
                type="button"
                className={l === activeLane ? 'seg active' : 'seg'}
                onClick={() => setLane(l)}
              >
                {l}
              </button>
            ))}
          </div>
        )}
      </div>
      {cached && status === null && (
        <EvidenceStatus title="equity" state="stale" detail="showing cached tick history while refresh is unavailable" onRetry={onRetry} />
      )}
      {status ?? <Plot data={data} />}
      {footnotes.length > 0 && <div className="chart-footnote">{footnotes.join(' · ')}</div>}
    </section>
  )
}

function Plot({ data }: { data: Normalized }) {
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

  // Destroy/recreate on data or width change; cleanup on unmount.
  useEffect(() => {
    const el = hostRef.current
    if (!el || width < 40 || data.ts.length < 2) return
    const text = cssColor('--text-dim', '#a9b4c5')
    const line = cssColor('--line', 'rgba(220,228,240,0.08)')
    const axis = {
      stroke: text,
      grid: { stroke: line, width: 1 },
      ticks: { stroke: line, width: 1 },
      font: '10px "IBM Plex Mono", monospace',
    }
    const u = new uPlot(
      {
        width,
        height: HEIGHT,
        // tick_ts is UTC — render axis labels in UTC so an after-close tick
        // (00:30 UTC) never drifts onto the wrong local date.
        tzDate: (ts) => uPlot.tzDate(new Date(ts * 1000), 'Etc/UTC'),
        legend: { show: false },
        cursor: { drag: { x: false, y: false } },
        series: [
          {},
          {
            // `--series-focus` / `--series-context` (theme.css): the validated dataviz-slice
            // palette, shared with `ReturnOverlay` — this chart plays the same focus/context
            // role (the live equity line vs. its own peak) and must not hardcode a second
            // source of truth for the same two colors.
            label: 'equity',
            stroke: cssColor('--series-focus', '#3982ff'),
            width: 1.5,
            points: { show: false },
          },
          {
            label: 'peak',
            stroke: cssColor('--series-context', '#727c8b'),
            width: 1,
            points: { show: false },
          },
        ],
        axes: [axis, { ...axis, size: 58 }],
      },
      [data.ts, data.equity, data.peak],
      el,
    )
    return () => u.destroy()
  }, [data, width])

  return <div ref={hostRef} className="chart-host" style={{ minHeight: HEIGHT }} />
}
