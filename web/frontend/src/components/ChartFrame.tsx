import type { ReactNode } from 'react'

/** Shell every preset-view chart mounts inside. Exists to enforce, in ONE place,
 * two rules every chart in this slice must follow:
 *
 * - The honest empty state: when `isEmpty`, render only the label — never an
 *   `<svg>`/`<canvas>` drawn around nothing (axes with no data, a zero baseline
 *   that reads as "flat performance"). The `runs` table on `main` has 0 rows and
 *   the design forbids backfill, so this is what the operator sees on day one —
 *   it is the default state, not an edge case. `children` is therefore NEVER
 *   rendered while `isEmpty` is true, so a chart can't bypass the rule by drawing
 *   its own placeholder.
 * - No layout shift: the body is a fixed `height` whether empty or populated, so
 *   the page never jumps when data arrives.
 */
export default function ChartFrame({
  title,
  isEmpty,
  emptyLabel,
  height,
  children,
}: {
  title: string
  isEmpty: boolean
  emptyLabel: string
  height: number
  children: ReactNode
}) {
  return (
    <section className="panel chart-frame">
      <div className="chart-frame-head">
        <span className="micro-label">{title}</span>
      </div>
      <div className="chart-frame-body" style={{ height }}>
        {isEmpty ? <div className="chart-frame-empty">{emptyLabel}</div> : children}
      </div>
    </section>
  )
}
