import { useEffect, useMemo, useRef, useState } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { parseUtc } from '../format'
import type { Lane, SeriesPayload, SeriesRow } from '../types'

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

function cssColor(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

export default function EquityChart({ payload }: { payload: SeriesPayload }) {
  // Lanes the backend actually returned (null = not requested / not present).
  const available = LANES.filter((l) => {
    const rows = payload.series[l]
    return rows !== null && rows.length > 0
  })
  const showPicker = payload.series.paper !== null && payload.series.live !== null
    && (payload.series.paper?.length ?? 0) > 0 && (payload.series.live?.length ?? 0) > 0

  const [lane, setLane] = useState<Lane>(available[0] ?? 'paper')
  const activeLane: Lane = available.includes(lane) ? lane : (available[0] ?? 'paper')

  const rows = payload.series[activeLane] ?? []
  const data = useMemo(() => normalize(rows), [rows])

  const footnotes: string[] = []
  // Truncation MUST be visible: `fleet series` keeps only the newest N per lane, so a
  // truncated curve's left edge is a cut, not the strategy's first tick. Silently
  // clipping it reads as "this is the whole history".
  if (payload.truncated[activeLane] === true) footnotes.push('newest ticks only (truncated)')
  if (payload.n_legacy_excluded > 0) footnotes.push(`${payload.n_legacy_excluded} legacy excluded`)
  if (payload.n_unparseable > 0) footnotes.push(`${payload.n_unparseable} unparseable`)
  if (payload.n_invalid_lane > 0) footnotes.push(`${payload.n_invalid_lane} invalid lane`)

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
      {data.ts.length < 2 ? (
        <div className="chart-placeholder">awaiting tick history</div>
      ) : (
        <Plot data={data} />
      )}
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
    const text = cssColor('--text-dim', '#6c7888')
    const line = cssColor('--line', 'rgba(255,255,255,0.075)')
    const axis = {
      stroke: text,
      grid: { stroke: line, width: 1 },
      ticks: { stroke: line, width: 1 },
      font: '10px "JetBrains Mono", monospace',
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
            label: 'equity',
            stroke: cssColor('--cyan', '#5cc8ff'),
            width: 1.5,
            points: { show: false },
          },
          {
            label: 'peak',
            stroke: cssColor('--text-fade', '#4a5462'),
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
