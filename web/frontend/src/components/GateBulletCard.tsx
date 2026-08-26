import { runDetailUrl, runsUrl, useFetch } from '../api'
import { num } from '../format'
import type { ApiEnvelope, GateCheck, RunDetail, RunsListPayload } from '../types'
import ChartFrame from './ChartFrame'
import PassMark from './PassMark'

// Fixed internal SVG coordinate space per row — the viewBox scales to fit the real
// container width, so no ResizeObserver dance is needed (same convention as ScatterISOOS).
const PLOT = { width: 280, height: 20, left: 2, right: 2 }

// The one non-color weight cue: a BINDING check's fill is visibly thicker than an ADVISORY
// check's — a real geometry attribute (readable with color entirely stripped), not a class name.
const BINDING_BAR_HEIGHT = 12
const ADVISORY_BAR_HEIGHT = 7

// The frame's height is a FIXED constant, not derived from `checks.length` — a list-shaped
// card has no natural plot area to size a fixed height from (see `ChartFrame`'s
// `variableHeight` doc), and computing one from the payload would only recreate the very
// layout-shift the fixed-height contract exists to prevent (empty state can never know the
// future populated row count). This sizes the honest-empty box only; once populated,
// `ChartFrame`'s `variableHeight` leaves the body unconstrained and it grows with content.
const FRAME_HEIGHT = 120

export interface BulletGeometry {
  /** Zero-anchored bar span (see the sparkline/scatter precedent in this slice: the domain
   * always includes 0, so "crossed zero" or "past the threshold" reads as real geometry, not
   * a self-scaled illusion). */
  barX0: number
  barX1: number
  thresholdX: number | null
  domainMin: number
  domainMax: number
}

/** Pure per-row layout: one check's value/threshold -> a zero-anchored bar span + threshold
 * tick, all in one row-local domain (each check has its own units/scale — a Sharpe ratio and a
 * combo count cannot share an axis, so every row is self-scaled).
 *
 * Returns `null` when `value` is missing (`null`/`undefined`) — a check that never ran has no
 * geometry to draw, and must NEVER be rendered as a bar reaching zero. The caller renders
 * "not evaluated" instead of calling this. */
export function buildBulletGeometry(check: GateCheck): BulletGeometry | null {
  const value = check.value
  if (value === null || value === undefined || !Number.isFinite(value)) return null
  const threshold =
    typeof check.threshold === 'number' && Number.isFinite(check.threshold)
      ? check.threshold
      : null

  const candidates = threshold !== null ? [0, value, threshold] : [0, value]
  let domainMin = Math.min(...candidates)
  let domainMax = Math.max(...candidates)
  if (domainMin === domainMax) {
    domainMin -= 1
    domainMax += 1
  }
  const span = domainMax - domainMin
  domainMin -= span * 0.1
  domainMax += span * 0.1
  const range = domainMax - domainMin

  const plotW = PLOT.width - PLOT.left - PLOT.right
  const scaleX = (v: number) => PLOT.left + ((v - domainMin) / range) * plotW

  return {
    barX0: scaleX(Math.min(0, value)),
    barX1: scaleX(Math.max(0, value)),
    thresholdX: threshold !== null ? scaleX(threshold) : null,
    domainMin,
    domainMax,
  }
}

function isAdvisory(check: GateCheck): boolean {
  return check.advisory === true
}

export interface SplitChecks {
  binding: GateCheck[]
  advisory: GateCheck[]
}

/** Pure grouping: partitions checks into binding (decides pass/fail) vs advisory (recorded,
 * never vetoes) — order-preserving within each group. The card's POSITION encoding (two
 * separate groups, binding first) reduces to this one partition; kept here, beside
 * `buildBulletGeometry`, as an isolated + unit-tested piece rather than inline `.filter` calls
 * in the component, so any future change to the split (or to how it interacts with layout) is
 * caught by a test rather than reaching review — the height-derivation defect this replaced
 * would have surfaced immediately had the equivalent arithmetic been isolated like this from
 * the start. */
export function splitChecks(checks: GateCheck[]): SplitChecks {
  const binding: GateCheck[] = []
  const advisory: GateCheck[] = []
  for (const c of checks) {
    ;(isAdvisory(c) ? advisory : binding).push(c)
  }
  return { binding, advisory }
}

function CheckRow({ check, index }: { check: GateCheck; index: number }) {
  const advisory = isAdvisory(check)
  const geometry = buildBulletGeometry(check)
  const name = check.name ?? `check ${index + 1}`
  const barHeight = advisory ? ADVISORY_BAR_HEIGHT : BINDING_BAR_HEIGHT
  const barY = (PLOT.height - barHeight) / 2

  return (
    <div className="bullet-row" data-testid="gate-check-row" data-kind={advisory ? 'advisory' : 'binding'}>
      <div className="bullet-row-head">
        <span className="bullet-check-name">{name}</span>
        <span className="bullet-check-kind">{advisory ? 'advisory' : 'binding'}</span>
        <PassMark passed={check.passed} advisory={advisory} />
      </div>
      {geometry === null ? (
        <div className="bullet-not-evaluated" data-testid="not-evaluated">
          not evaluated
        </div>
      ) : (
        <div className="bullet-bar-row">
          <svg
            className="bullet-svg"
            viewBox={`0 0 ${PLOT.width} ${PLOT.height}`}
            role="img"
            aria-label={`${name}: ${num(check.value, 4)}${
              check.threshold != null ? ` against threshold ${num(check.threshold, 4)}` : ''
            }, ${advisory ? 'advisory — never vetoes' : 'binding'}`}
          >
            <rect
              className="bullet-track"
              x={PLOT.left}
              y={(PLOT.height - BINDING_BAR_HEIGHT) / 2}
              width={PLOT.width - PLOT.left - PLOT.right}
              height={BINDING_BAR_HEIGHT}
            />
            <rect
              className={
                check.passed === false
                  ? advisory
                    ? 'bullet-fill-advisory-fail'
                    : 'bullet-fill-fail'
                  : check.passed === true
                    ? advisory
                      ? 'bullet-fill-advisory-pass'
                      : 'bullet-fill-pass'
                    : 'bullet-fill-unknown'
              }
              data-testid="bullet-fill"
              x={Math.min(geometry.barX0, geometry.barX1)}
              y={barY}
              width={Math.max(1, Math.abs(geometry.barX1 - geometry.barX0))}
              height={barHeight}
            />
            {geometry.thresholdX !== null && (
              <line
                className="bullet-threshold"
                data-testid="bullet-threshold"
                x1={geometry.thresholdX}
                x2={geometry.thresholdX}
                y1={0}
                y2={PLOT.height}
              />
            )}
          </svg>
          <div className="bullet-values">
            <span className="dim-note num">
              {check.op ?? ''} {num(check.threshold, 4)}
            </span>
            <span className="num">{num(check.value, 4)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * View 5 — the gate bullet card (spec §6.1). Replaces the per-check `<table>` that used to sit
 * inside `StrategyDetail`'s `LatestGate` — the densest text dump in the app.
 *
 * Source is deliberately `/api/runs/{id}` for a `gate` run, NOT the already-fetched
 * `StrategyDetailResponse.gates` composite: this is a two-step read over the SAME run-ledger
 * surface every other view in this slice uses (`/api/runs?strategy=&kind=gate&limit=1` to find
 * the latest gate run's id, then `/api/runs/{id}` for its allow-list-projected `gate_decision`).
 * Renders exactly what the allowlist gives it — never requests a wider payload.
 *
 * THE DISTINCTION THIS CARD EXISTS TO MAKE: binding checks decide pass/fail; advisory checks
 * compute and are recorded but have NO veto power (`algua/research/gates.py`). Conflating them
 * misreads the whole gate. Encoded three ways, none of them color: POSITION (two separate
 * groups, binding first), LABEL (each row says "binding" or "advisory" in text), and WEIGHT (the
 * advisory bar is visibly thinner — a real SVG height attribute). Status color (pass=green,
 * binding fail=red, advisory fail=amber) is reinforcement only, exactly PassMark's existing rule.
 *
 * A check whose `value` is NULL (never ran, or ran but the shape guard dropped it) renders
 * "not evaluated" — never a bar drawn at zero, which would fabricate a data point on the one
 * card whose entire job is explaining a rejection.
 */
export default function GateBulletCard({ strategy }: { strategy: string }) {
  const listUrl = runsUrl({ strategy, kind: 'gate', limit: 1 })
  const list = useFetch<ApiEnvelope<RunsListPayload>>(listUrl)
  const runId = list.data?.data?.runs?.[0]?.id ?? null
  const detail = useFetch<ApiEnvelope<RunDetail>>(runId !== null ? runDetailUrl(runId) : null)

  const checks = detail.data?.data?.gate_decision?.checks ?? []
  const { binding, advisory: advisoryChecks } = splitChecks(checks)
  const isEmpty = checks.length === 0

  return (
    <ChartFrame
      title="gate checks"
      isEmpty={isEmpty}
      emptyLabel="no gate checks recorded yet — the ledger fills once a gate run exists"
      height={FRAME_HEIGHT}
      variableHeight
    >
      <div className="bullet-card">
        {binding.length > 0 && (
          <div className="bullet-group" data-testid="bullet-group-binding">
            <div className="micro-label">binding — decides pass / fail</div>
            {binding.map((c, i) => (
              <CheckRow key={`${c.name ?? 'check'}-${i}`} check={c} index={i} />
            ))}
          </div>
        )}
        {advisoryChecks.length > 0 && (
          <div className="bullet-group" data-testid="bullet-group-advisory">
            <div className="micro-label">advisory — recorded, never vetoes</div>
            {advisoryChecks.map((c, i) => (
              <CheckRow key={`${c.name ?? 'check'}-${i}`} check={c} index={i} />
            ))}
          </div>
        )}
      </div>
    </ChartFrame>
  )
}
