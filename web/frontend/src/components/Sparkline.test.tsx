import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import Sparkline, { buildSparklineGeometry } from './Sparkline'

afterEach(cleanup)

describe('buildSparklineGeometry (pure layout)', () => {
  it('draws one continuous segment through three finite values', () => {
    const geo = buildSparklineGeometry([0.2, 0.5, 0.1], 60, 20)
    expect(geo.isEmpty).toBe(false)
    expect(geo.segments.length).toBe(1)
    expect(geo.segments[0].length).toBe(3)
  })

  it('treats a NULL entry as a gap — never a fabricated zero interpolated into the line', () => {
    const geo = buildSparklineGeometry([0.2, null, 0.4], 60, 20)
    // A real zero value and a missing value must not produce the same geometry: the middle
    // index is skipped entirely rather than plotted at height/2 (which is what an interpolated
    // "0" would look like on this scale).
    expect(geo.segments.length).toBe(2)
    expect(geo.segments[0].length).toBe(1)
    expect(geo.segments[1].length).toBe(1)
  })

  it('reports empty geometry with fewer than two finite values — a single point cannot show a trend', () => {
    expect(buildSparklineGeometry([0.4], 60, 20).isEmpty).toBe(true)
    expect(buildSparklineGeometry([null, null, null], 60, 20).isEmpty).toBe(true)
    expect(buildSparklineGeometry([], 60, 20).isEmpty).toBe(true)
  })

  it('places the higher value at the smaller y (SVG y grows downward)', () => {
    const geo = buildSparklineGeometry([0.1, 0.9], 60, 20)
    const [a, b] = geo.segments[0]
    expect(b.y).toBeLessThan(a.y)
  })

  it('pads a flat-at-zero series symmetrically — never divides by zero', () => {
    const geo = buildSparklineGeometry([0, 0, 0], 60, 20)
    expect(geo.isEmpty).toBe(false)
    for (const p of geo.segments[0]) expect(Number.isFinite(p.y)).toBe(true)
    // Padded symmetrically around zero, so the (degenerate) baseline sits centered.
    expect(geo.zeroY).toBeCloseTo(10, 5)
  })

  it('the y-domain always includes zero, even when every finite value is positive', () => {
    // All-positive values: zero is the domain FLOOR, so the baseline sits at the bottom edge —
    // never fabricated mid-band the way padding an arbitrary range would place it.
    const geo = buildSparklineGeometry([0.2, 0.5, 0.1], 60, 20, 2)
    expect(geo.zeroY).toBeCloseTo(18, 5) // height(20) - padding(2)
  })

  it('the y-domain always includes zero, even when every finite value is negative', () => {
    // All-negative values: zero is the domain CEILING, so the baseline sits at the top edge.
    const geo = buildSparklineGeometry([-0.2, -0.5, -0.1], 60, 20, 2)
    expect(geo.zeroY).toBeCloseTo(2, 5) // == padding
  })

  it('a mild decline and a crater through zero are now visibly distinguishable relative to the baseline', () => {
    // Under PURE self-scaling (each row's own min/max, no zero anchor), these two series draw
    // IDENTICALLY: both are "high, slightly lower, lower still" — three points strictly
    // monotonic down, same relative shape, regardless of whether the last point is 0.85 or
    // -3.0. That was the bug this test guards against.
    const mildDecline = buildSparklineGeometry([1.0, 0.9, 0.85], 60, 20)
    const crater = buildSparklineGeometry([1.0, 0.9, -3.0], 60, 20)

    const mildLast = mildDecline.segments[0][2]
    const craterLast = crater.segments[0][2]

    // The mild decline never reaches zero: its last point stays ABOVE (smaller y than) its own
    // baseline — still high in the band.
    expect(mildLast.y).toBeLessThan(mildDecline.zeroY)
    // The crater actually crosses zero: its last point sits BELOW (larger y than) its own
    // baseline — the one distinction that matters is now visually real.
    expect(craterLast.y).toBeGreaterThan(crater.zeroY)
  })
})

describe('Sparkline (component)', () => {
  it('renders a polyline for a real trio of values', () => {
    const { container } = render(<Sparkline values={[0.3, 0.52, 0.61]} />)
    const svg = container.querySelector('[data-testid="sparkline"]')
    expect(svg).toBeTruthy()
    expect(svg?.getAttribute('data-empty')).toBe('false')
    expect(container.querySelectorAll('polyline').length).toBeGreaterThan(0)
  })

  it('draws a zero baseline alongside the data when populated, and none when empty', () => {
    const { container } = render(<Sparkline values={[0.3, 0.52, 0.61]} />)
    expect(container.querySelector('.sparkline-baseline')).toBeTruthy()
    cleanup()
    const { container: emptyContainer } = render(<Sparkline values={[null, null, null]} />)
    expect(emptyContainer.querySelector('.sparkline-baseline')).toBeNull()
  })

  it('renders no polyline — honest, not a fabricated flat line — when data is insufficient', () => {
    const { container } = render(<Sparkline values={[null, null, null]} />)
    const svg = container.querySelector('[data-testid="sparkline"]')
    expect(svg).toBeTruthy()
    expect(svg?.getAttribute('data-empty')).toBe('true')
    expect(container.querySelector('polyline')).toBeNull()
  })

  it('is monochrome: every polyline shares the same stroke class, never a categorical hue', () => {
    const { container } = render(<Sparkline values={[0.08, null, 0.98]} />)
    const lines = container.querySelectorAll('polyline')
    expect(lines.length).toBe(2)
    for (const line of lines) expect(line.getAttribute('class')).toBe('sparkline-line')
  })

  it('never claims to be a "trend" — the aria-label describes the mark honestly', () => {
    const { container } = render(<Sparkline values={[0.3, 0.52, 0.61]} />)
    const svg = container.querySelector('[data-testid="sparkline"]')
    expect(svg?.getAttribute('aria-label')).not.toMatch(/^trend$/i)
    cleanup()
    const { container: withLabel } = render(
      <Sparkline values={[0.3, 0.52, 0.61]} label="worst window to holdout, relative to zero" />,
    )
    expect(withLabel.querySelector('[data-testid="sparkline"]')?.getAttribute('aria-label')).toBe(
      'worst window to holdout, relative to zero',
    )
  })
})
