import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { SeriesPayload } from '../types'
import EquityChart from './EquityChart'

afterEach(cleanup)

function row(id: number, tick_ts: string) {
  return {
    id,
    tick_ts,
    recorded_at: tick_ts,
    equity: 100_000 + id,
    peak_equity: 100_000 + id,
    reconcile_ok: true,
  }
}

function payload(series: SeriesPayload['series']): SeriesPayload {
  return {
    strategy: 'mom_breakout',
    lane_filter: null,
    series,
    truncated: { paper: false, live: false },
    n_legacy_excluded: 0,
    n_unparseable: 0,
    n_invalid_lane: 0,
  }
}

it('renders a compact genuine-empty state without drawing a chart', () => {
  const payload: SeriesPayload = {
    strategy: 'mom_breakout',
    lane_filter: null,
    series: { paper: [], live: null },
    truncated: { paper: false, live: null },
    n_legacy_excluded: 2,
    n_unparseable: 1,
    n_invalid_lane: 0,
  }
  const { container } = render(<EquityChart payload={payload} />)
  expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('empty')
  expect(container.querySelector('canvas, svg')).toBeNull()
  // dim footnote surfaces the excluded/unparseable counts
  expect(screen.getByText(/2 legacy excluded · 1 unparseable/)).toBeTruthy()
})

it('flags a truncated lane so the clipped left edge is not read as the full history', () => {
  const payload: SeriesPayload = {
    strategy: 'mom_breakout',
    lane_filter: 'paper',
    series: {
      paper: [
        {
          id: 1,
          tick_ts: '2026-08-08T20:30:00+00:00',
          recorded_at: '2026-08-08T20:30:01+00:00',
          equity: 100_000,
          peak_equity: 100_000,
          reconcile_ok: true,
        },
      ],
      live: null,
    },
    truncated: { paper: true, live: null },
    n_legacy_excluded: 0,
    n_unparseable: 0,
    n_invalid_lane: 3,
  }
  render(<EquityChart payload={payload} />)
  expect(screen.getByText(/newest ticks only \(truncated\) · 3 invalid lane/)).toBeTruthy()
})

it('renders a single-point lane as incomplete rather than empty', () => {
  const payload: SeriesPayload = {
    strategy: 'mom_breakout',
    lane_filter: 'paper',
    series: {
      paper: [
        {
          id: 1,
          tick_ts: '2026-08-08T20:30:00+00:00',
          recorded_at: '2026-08-08T20:30:01+00:00',
          equity: 100_000,
          peak_equity: 100_000,
          reconcile_ok: true,
        },
      ],
      live: null,
    },
    truncated: { paper: false, live: null },
    n_legacy_excluded: 0,
    n_unparseable: 0,
    n_invalid_lane: 0,
  }
  render(<EquityChart payload={payload} />)
  expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('incomplete')
})

it('distinguishes loading and failure before a payload exists', () => {
  const loading = render(<EquityChart loading />)
  expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('loading')
  loading.unmount()
  render(<EquityChart error={new Error('offline')} onRetry={vi.fn()} />)
  expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('failed')
  expect(screen.getByRole('button', { name: 'retry' })).toBeTruthy()
})

it.each([
  ['empty', payload({ paper: [], live: null }), /cached tick history is empty/i],
  ['incomplete', payload({ paper: [row(1, '2026-08-08T20:30:00Z')], live: null }), /cached tick history is incomplete/i],
] as const)('preserves stale freshness when cached equity is %s', (_, value, detail) => {
  render(<EquityChart payload={value} stale onRetry={vi.fn()} />)

  expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
  expect(screen.getByText(detail)).toBeTruthy()
})

it('treats a failed refresh with cached drawable equity as stale without hiding the curve host', () => {
  const value = payload({
    paper: [row(1, '2026-08-08T20:30:00Z'), row(2, '2026-08-09T20:30:00Z')],
    live: null,
  })
  const { container } = render(
    <EquityChart payload={value} error={new Error('refresh failed')} onRetry={vi.fn()} />,
  )

  expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
  expect(container.querySelector('.chart-host')).toBeTruthy()
})

it('derives availability from valid normalized points and defaults to live after async data arrives', () => {
  const view = render(<EquityChart loading />)
  view.rerender(<EquityChart payload={payload({
    paper: [row(1, 'not-a-timestamp')],
    live: [row(2, '2026-08-08T20:30:00Z'), row(3, '2026-08-09T20:30:00Z')],
  })} />)

  expect(screen.getByText('equity — live')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'paper' })).toBeNull()
  expect(view.container.querySelector('.chart-host')).toBeTruthy()
})
