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

  it('centers a flat line when every finite value is identical — never divides by zero', () => {
    const geo = buildSparklineGeometry([0.3, 0.3, 0.3], 60, 20)
    expect(geo.isEmpty).toBe(false)
    for (const p of geo.segments[0]) expect(Number.isFinite(p.y)).toBe(true)
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
})
