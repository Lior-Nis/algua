import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import type { SeriesPayload } from '../types'
import EquityChart from './EquityChart'

afterEach(cleanup)

it('renders the awaiting-tick-history placeholder for an empty series without crashing', () => {
  const payload: SeriesPayload = {
    strategy: 'mom_breakout',
    lane_filter: null,
    series: { paper: [], live: null },
    truncated: { paper: false, live: null },
    n_legacy_excluded: 2,
    n_unparseable: 1,
    n_invalid_lane: 0,
  }
  render(<EquityChart payload={payload} />)
  expect(screen.getByText(/awaiting tick history/i)).toBeTruthy()
  // dim footnote surfaces the excluded/unparseable counts
  expect(screen.getByText(/2 legacy excluded · 1 unparseable/)).toBeTruthy()
})

it('renders a single-point lane as the placeholder too (uPlot needs >=2 points)', () => {
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
  expect(screen.getByText(/awaiting tick history/i)).toBeTruthy()
})
