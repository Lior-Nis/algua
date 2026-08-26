import { readFileSync } from 'node:fs'
import path from 'node:path'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
import ScatterISOOS, { buildScatterGeometry } from './ScatterISOOS'

// ---- deferred item, promoted: the diagonal's contrast against pure black ------------------

function srgbToLinear(c: number): number {
  const v = c / 255
  return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
}

function relativeLuminance(hex: string): number {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b)
}

function contrastVsBlack(hex: string): number {
  // WCAG contrast ratio, black's luminance is exactly 0.
  return (relativeLuminance(hex) + 0.05) / 0.05
}

describe('.scatter-diagonal contrast (deferred item, promoted)', () => {
  it("the diagonal's stroke token clears the 3:1 floor for a meaningful graphical object on " +
    'pure black — `--line-2` (the previous token) computed to ~1.53:1', () => {
    const css = readFileSync(path.resolve(__dirname, '../theme.css'), 'utf-8')
    const rule = css.match(/\.scatter-diagonal\s*\{([^}]*)\}/)
    expect(rule).toBeTruthy()
    expect(rule![1]).not.toMatch(/--line-2\b/)
    const strokeVarMatch = rule![1].match(/stroke:\s*var\((--[\w-]+)/)
    expect(strokeVarMatch).toBeTruthy()
    const tokenName = strokeVarMatch![1]
    const tokenMatch = css.match(new RegExp(`${tokenName}:\\s*(#[0-9a-fA-F]{6})`))
    expect(tokenMatch).toBeTruthy()
    const contrast = contrastVsBlack(tokenMatch![1])
    expect(contrast).toBeGreaterThanOrEqual(3)
  })
})

function row(overrides: Partial<RunRow> & { id: number; strategy_name: string }): RunRow {
  return {
    kind: 'gate',
    strategy_id: 1,
    created_at: '2026-08-25T00:00:00+00:00',
    passed: 1,
    mean_window_sharpe: null,
    sharpe_oos: null,
    ...overrides,
  }
}

describe('buildScatterGeometry (pure layout)', () => {
  it('excludes a run missing either metric and counts it — never plots it at 0', () => {
    const runs = [
      row({ id: 1, strategy_name: 'has_both', mean_window_sharpe: 0.3, sharpe_oos: 0.4 }),
      row({ id: 2, strategy_name: 'missing_oos', mean_window_sharpe: 0.3, sharpe_oos: null }),
      row({ id: 3, strategy_name: 'missing_is', mean_window_sharpe: null, sharpe_oos: 0.4 }),
      row({ id: 4, strategy_name: 'missing_both', mean_window_sharpe: null, sharpe_oos: null }),
    ]
    const geo = buildScatterGeometry(runs)
    expect(geo.points.map((p) => p.id)).toEqual([1])
    expect(geo.excludedCount).toBe(3)
    // The excluded runs must never surface as a plotted (0, y) or (x, 0) point.
    expect(geo.points.some((p) => p.x === 0 || p.y === 0)).toBe(false)
  })

  it('places a point with equal IS/OOS values exactly on the y=x diagonal', () => {
    const runs = [
      row({ id: 1, strategy_name: 'on_diagonal', mean_window_sharpe: 0.5, sharpe_oos: 0.5 }),
      row({ id: 2, strategy_name: 'anchor_low', mean_window_sharpe: -0.2, sharpe_oos: -0.2 }),
      row({ id: 3, strategy_name: 'anchor_high', mean_window_sharpe: 1.5, sharpe_oos: 1.5 }),
    ]
    const geo = buildScatterGeometry(runs)
    const { min: domainMin, max: domainMax } = geo.domain
    const p = geo.points.find((pt) => pt.id === 1)
    expect(p).toBeTruthy()
    // Interpolate the diagonal's own endpoints at this point's data value — if the diagonal is a
    // real y=x mapping (not merely a line that happens to look 45 degrees), the point's rendered
    // (cx, cy) must land exactly on that interpolation.
    const t = ((p!.x as number) - domainMin) / (domainMax - domainMin)
    const expectedCx = geo.diagonal.x1 + t * (geo.diagonal.x2 - geo.diagonal.x1)
    const expectedCy = geo.diagonal.y1 + t * (geo.diagonal.y2 - geo.diagonal.y1)
    expect(p!.cx).toBeCloseTo(expectedCx, 6)
    expect(p!.cy).toBeCloseTo(expectedCy, 6)

    // Domain/padding-agnostic check: three DIFFERENT on-diagonal data points must render
    // perfectly collinear with each other. If x and y were ever scaled against different
    // domains, three points with x===y would NOT line up — this would catch that regardless of
    // what the padding constant happens to be.
    const [a, b, c] = [1, 2, 3].map((id) => geo.points.find((pt) => pt.id === id)!)
    const cross = (b.cx - a.cx) * (c.cy - a.cy) - (b.cy - a.cy) * (c.cx - a.cx)
    expect(cross).toBeCloseTo(0, 6)
  })

  it('marks a point above the diagonal (oos > is) and one below distinctly', () => {
    const runs = [
      row({ id: 1, strategy_name: 'mined', mean_window_sharpe: 0.1, sharpe_oos: 1.4 }),
      row({ id: 2, strategy_name: 'overfit', mean_window_sharpe: 1.3, sharpe_oos: 0.1 }),
    ]
    const geo = buildScatterGeometry(runs)
    const mined = geo.points.find((p) => p.id === 1)!
    const overfit = geo.points.find((p) => p.id === 2)!
    expect(mined.above).toBe(true)
    expect(overfit.above).toBe(false)
    // Above a diagonal drawn top-left-to-bottom-right in SVG space (y grows downward) means a
    // SMALLER cy for the same-or-lower x — sanity-check the pixel geometry agrees with the
    // semantic "above" flag, not just the raw data comparison.
    expect(mined.cy).toBeLessThan(overfit.cy)
  })

  it('an empty run list produces zero points and zero exclusions', () => {
    const geo = buildScatterGeometry([])
    expect(geo.points).toEqual([])
    expect(geo.excludedCount).toBe(0)
  })

  // Fix round 3 (review FIX G): `outlier` used to be assigned from the label set, which is capped
  // at MAX_OUTLIER_LABELS = 3 — so a 4th run genuinely past the gap threshold was painted in the
  // neutral CONTEXT colour. The chart said "unremarkable" about a run that had cleared the very
  // threshold the colour exists to mark. Predicate and label budget are now separate.
  it('a 4th run past the gap threshold is still coloured as an outlier — the cap governs ' +
    'LABELS only', () => {
    const runs = [
      // Anchors: fix the domain so the gap threshold is a stable fraction of a known range.
      row({ id: 90, strategy_name: 'anchor_low', mean_window_sharpe: 0, sharpe_oos: 0 }),
      row({ id: 91, strategy_name: 'anchor_high', mean_window_sharpe: 2, sharpe_oos: 2 }),
      // Four runs, all far above the diagonal, in descending gap order.
      row({ id: 1, strategy_name: 'mined_a', mean_window_sharpe: 0.0, sharpe_oos: 1.9 }),
      row({ id: 2, strategy_name: 'mined_b', mean_window_sharpe: 0.0, sharpe_oos: 1.8 }),
      row({ id: 3, strategy_name: 'mined_c', mean_window_sharpe: 0.0, sharpe_oos: 1.7 }),
      row({ id: 4, strategy_name: 'mined_d', mean_window_sharpe: 0.0, sharpe_oos: 1.6 }),
    ]
    const geo = buildScatterGeometry(runs)
    const mined = geo.points.filter((p) => p.strategy.startsWith('mined_'))
    expect(mined).toHaveLength(4)

    // Every one of them cleared the threshold, so every one of them is an outlier.
    expect(mined.every((p) => p.outlier)).toBe(true)
    // But only three carry a direct text label — labels collide, fills do not.
    expect(mined.filter((p) => p.labelled)).toHaveLength(3)
    // The three labelled ones are the widest gaps, not an arbitrary three.
    expect(mined.filter((p) => p.labelled).map((p) => p.strategy).sort()).toEqual([
      'mined_a',
      'mined_b',
      'mined_c',
    ])
    // And the anchors, which sit exactly on the diagonal, are neither.
    const anchors = geo.points.filter((p) => p.strategy.startsWith('anchor_'))
    expect(anchors.every((p) => !p.outlier && !p.labelled)).toBe(true)
  })
})

const runsEnvelope: ApiEnvelope<RunsListPayload> = {
  ok: true,
  fetched_at: '2026-08-26T00:00:00Z',
  stale: false,
  data: {
    count: 6,
    runs: [
      row({ id: 1, strategy_name: 'honest_pos', mean_window_sharpe: 0.52, sharpe_oos: 0.61 }),
      row({ id: 2, strategy_name: 'honest_neg', mean_window_sharpe: -0.18, sharpe_oos: -0.24 }),
      // Far above the diagonal — the mined case this chart exists to catch.
      row({ id: 3, strategy_name: 'mined_above_1', mean_window_sharpe: 0.08, sharpe_oos: 1.42 }),
      row({ id: 4, strategy_name: 'overfit_below', mean_window_sharpe: 1.35, sharpe_oos: 0.11 }),
      // Missing OOS entirely — must be excluded, never plotted at (x, 0).
      row({ id: 5, strategy_name: 'null_metrics', mean_window_sharpe: null, sharpe_oos: null }),
      row({ id: 6, strategy_name: 'honest_pos_2', mean_window_sharpe: 0.29, sharpe_oos: 0.24 }),
    ],
  },
}

function stubRunsFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => runsEnvelope,
    })) as unknown as typeof fetch,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('draws the y=x diagonal', async () => {
  stubRunsFetch()
  render(<ScatterISOOS />)
  const diagonal = await screen.findByTestId('diagonal')
  expect(diagonal.tagName.toLowerCase()).toBe('line')
  expect(diagonal.getAttribute('x1')).not.toBe(diagonal.getAttribute('x2'))
})

it('renders above- and below-diagonal points as DOM-distinguishable regions', async () => {
  stubRunsFetch()
  render(<ScatterISOOS />)
  await screen.findByTestId('diagonal')
  const points = screen.getAllByTestId('scatter-point')
  expect(points.length).toBe(5) // 6 rows, 1 excluded (null_metrics)
  const above = points.filter((p) => p.getAttribute('data-region') === 'above')
  const below = points.filter((p) => p.getAttribute('data-region') === 'below')
  expect(above.length).toBeGreaterThan(0)
  expect(below.length).toBeGreaterThan(0)
  expect(above.length + below.length).toBe(points.length)
})

it('excludes the NULL-metric run and reports the exclusion count — never plots it at zero', async () => {
  stubRunsFetch()
  render(<ScatterISOOS />)
  await screen.findByTestId('diagonal')
  expect(screen.getByText(/1 run.*excluded/i)).toBeTruthy()
  expect(screen.queryByText('null_metrics')).toBeNull()
})

it('direct-labels only the far outlier, never every point', async () => {
  stubRunsFetch()
  render(<ScatterISOOS />)
  await screen.findByTestId('diagonal')
  // mined_above_1 (gap 1.34) is the clear outlier; honest_pos (gap 0.09, barely above) must
  // NOT get a label — direct-label is reserved for the runs that matter.
  expect(screen.getByText('mined_above_1')).toBeTruthy()
  expect(screen.queryByText('honest_pos')).toBeNull()
  expect(screen.queryByText('overfit_below')).toBeNull()
  expect(screen.queryByText('honest_neg')).toBeNull()
})

it('colours a point Electric ONLY when it is a true outlier — never merely "above the ' +
  'diagonal" (fix round 2: honest_pos sits above by a hair (gap 0.09) and must read as ' +
  'context, exactly like the label rule)', async () => {
  stubRunsFetch()
  render(<ScatterISOOS />)
  await screen.findByTestId('diagonal')
  const points = screen.getAllByTestId('scatter-point')
  const above = points.filter((p) => p.getAttribute('data-region') === 'above')
  // Two points sit above the diagonal (honest_pos, mined_above_1) but only ONE is a real
  // outlier — the colour encoding must agree with the label rule, not with "above" alone.
  expect(above.length).toBe(2)
  const electricAbove = above.filter(
    (p) => p.querySelector('.scatter-point-outlier') !== null,
  )
  expect(electricAbove.length).toBe(1)
  const neutralAbove = above.filter(
    (p) => p.querySelector('.scatter-point-normal') !== null,
  )
  expect(neutralAbove.length).toBe(1)
})
