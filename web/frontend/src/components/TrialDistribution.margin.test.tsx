/** The deflation strip must answer "did I clear it, and BY HOW MUCH" — honestly.
 *
 * The defect: with a real seeded bar of 0.6118 and a real holdout of 0.61, the strip auto-scaled
 * to that 0.0018 span, so the two marks sat at opposite ends of the plot — the full card width of
 * visual daylight — while every label on the card, both axis ticks included, printed "0.61". A
 * near-miss rendered as a chasm between two identical numbers, and `GateBulletCard` printed the
 * same check at 4 decimals one card away, so the two views on one screen disagreed.
 *
 * Two properties, both tested here: the domain includes ZERO (so magnitude survives and a tiny
 * margin renders tiny), and the numbers print at a precision that can tell the two values apart.
 */
import { render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { num } from '../format'
import type { ApiEnvelope, RunDetail, RunRow, RunsListPayload } from '../types'
import TrialDistribution, { buildHoldoutStripGeometry, marginPhrase } from './TrialDistribution'

// The real numbers off the seeded scratch database that produced the defect.
const REAL_BAR = 0.6118
const REAL_HOLDOUT = 0.61

afterEach(() => {
  vi.unstubAllGlobals()
})

it('the strip domain is zero-anchored, so a 0.0018 miss renders as a hair, not a chasm', () => {
  const geometry = buildHoldoutStripGeometry(REAL_BAR, {
    value: REAL_HOLDOUT,
    passed: false,
  })
  expect(geometry.domain).not.toBeNull()
  const domain = geometry.domain as { min: number; max: number }
  // Zero is inside the plotted domain — the whole point.
  expect(domain.min).toBeLessThanOrEqual(0)
  expect(domain.max).toBeGreaterThan(REAL_BAR)

  const bar = geometry.bar as { cx: number }
  const own = geometry.own as { cx: number }
  const plotWidth = 340 - 16 - 16
  const separation = Math.abs(bar.cx - own.cx)
  // Auto-scaled, these two sat ~76% of the plot width apart (a fail diamond at 12% and a bar at
  // 88%). Zero-anchored, the 0.0018 margin is worth well under 1% of the plot.
  expect(separation / plotWidth).toBeLessThan(0.01)
})

it('a genuine miss still renders as a genuine gap — the fix must not flatten every margin', () => {
  const near = buildHoldoutStripGeometry(REAL_BAR, { value: REAL_HOLDOUT, passed: false })
  const far = buildHoldoutStripGeometry(REAL_BAR, { value: 0.05, passed: false })
  const nearGap = Math.abs((near.bar as { cx: number }).cx - (near.own as { cx: number }).cx)
  const farGap = Math.abs((far.bar as { cx: number }).cx - (far.own as { cx: number }).cx)
  expect(farGap).toBeGreaterThan(nearGap * 50)
})

it('prints the bar, the result and the margin at GateBulletCard precision, and states the ' +
  'margin as a number rather than leaving it to be eyeballed', async () => {
  const trials: RunRow[] = [
    { id: 1, kind: 'sweep_trial', strategy_name: 's', mean_window_sharpe: 0.2 } as RunRow,
    { id: 2, kind: 'sweep_trial', strategy_name: 's', mean_window_sharpe: 0.5 } as RunRow,
  ]
  const row = { id: 77, kind: 'gate', strategy_name: 'near_miss', mean_window_sharpe: 0.5 } as RunRow
  const listEnvelope = (runs: RunRow[]): ApiEnvelope<RunsListPayload> => ({
    ok: true,
    fetched_at: '2026-08-26T00:00:00Z',
    stale: false,
    data: { runs, count: runs.length } as RunsListPayload,
  })
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            ok: true,
            fetched_at: '2026-08-26T00:00:00Z',
            stale: false,
            data: {
              ...row,
              extra_metrics: {},
              gate_decision: {
                effective_min_holdout_sharpe: REAL_BAR,
                checks: [
                  {
                    name: 'holdout_sharpe',
                    value: REAL_HOLDOUT,
                    threshold: REAL_BAR,
                    passed: false,
                    advisory: true,
                  },
                ],
              },
            } as unknown as RunDetail,
          }),
        }
      }
      if (url.includes('kind=sweep_trial')) {
        return { ok: true, status: 200, json: async () => listEnvelope(trials) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([row]) }
    }) as unknown as typeof fetch,
  )

  render(<TrialDistribution strategy="near_miss" />)

  // The two values no longer print the same string — this is exactly what `GateBulletCard`
  // renders for the identical `holdout_sharpe` check (`num(check.value, 4)`).
  const barLabel = await screen.findByTestId('bar-label')
  expect(barLabel.textContent).toContain(num(REAL_BAR, 4))
  expect(screen.getByTestId('own-marker-label').textContent).toContain(num(REAL_HOLDOUT, 4))
  expect(num(REAL_BAR, 4)).not.toBe(num(REAL_HOLDOUT, 4))

  // And the difference — the number the reader actually wants — is stated outright.
  const summary = screen.getByTestId('trial-dist-summary').textContent ?? ''
  expect(summary).toContain('missed by 0.0018')
})

it('names the direction of the margin, and never prints "missed by 0"', () => {
  expect(marginPhrase(REAL_HOLDOUT, REAL_BAR)).toBe('missed by 0.0018')
  expect(marginPhrase(0.75, 0.5)).toBe('cleared by 0.25')
  // A difference that rounds away at the 4-decimal display precision must say so in words
  // rather than print a margin of zero, which would read as a bug.
  expect(marginPhrase(0.500001, 0.5)).toBe('level with the bar')
  expect(marginPhrase(0.5, 0.5)).toBe('level with the bar')
})
