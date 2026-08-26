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
 *   the page never jumps when data arrives. This is the DEFAULT for every consumer —
 *   `variableHeight` is opt-in, so existing charts keep the guarantee with no edit.
 *
 * `variableHeight` (default `false`): a genuine PLOT has a natural fixed plot area, but a
 * LIST-shaped chart (e.g. the gate bullet card: one row per check, count varies with the gate)
 * does not — forcing a fixed height on it either clips rows or nests a scroll region inside a
 * page that already scrolls, which hides content behind an affordance the reader has to
 * discover. A list-shaped consumer opts out explicitly instead of quietly computing a height
 * from its payload (which defeats the no-layout-shift contract by construction: the empty state
 * can never know the future populated row count). When `variableHeight` is true, `height` sets
 * ONLY the empty state's height (so the honest-empty box has a stable, non-collapsing size);
 * once populated, the body is left unconstrained and sizes to its content naturally.
 */
export default function ChartFrame({
  title,
  isEmpty,
  emptyLabel,
  height,
  variableHeight = false,
  children,
}: {
  title: string
  isEmpty: boolean
  emptyLabel: string
  height: number
  variableHeight?: boolean
  children: ReactNode
}) {
  const bodyStyle = variableHeight && !isEmpty ? undefined : { height }
  return (
    <section className="panel chart-frame">
      <div className="chart-frame-head">
        <span className="micro-label">{title}</span>
      </div>
      <div className="chart-frame-body" style={bodyStyle}>
        {isEmpty ? <div className="chart-frame-empty">{emptyLabel}</div> : children}
      </div>
    </section>
  )
}
