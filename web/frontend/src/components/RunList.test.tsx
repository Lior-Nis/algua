import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
import RunList from './RunList'

function row(overrides: Partial<RunRow> & { id: number; strategy_name: string }): RunRow {
  return {
    kind: 'gate',
    strategy_id: 1,
    created_at: '2026-08-25T00:00:00+00:00',
    passed: 1,
    mean_window_sharpe: null,
    sharpe_oos: null,
    min_window_sharpe: null,
    ...overrides,
  }
}

// Server order for the default sort (`sharpe_oos`): DESC with NULLs last, exactly as
// `list_runs` produces it (`ORDER BY sharpe_oos IS NULL, sharpe_oos DESC`). The component must
// render this order VERBATIM — a real negative number (honest_neg, -0.24) legitimately sits
// above the NULL row, which a client-side resort that coalesces null to 0 would get backwards.
const defaultSortRuns: RunRow[] = [
  row({ id: 3, strategy_name: 'mined_above_1', sharpe_oos: 1.42, mean_window_sharpe: 0.08,
        min_window_sharpe: -0.1, passed: 0 }),
  row({ id: 1, strategy_name: 'trend_breakout_v1', sharpe_oos: 0.61, mean_window_sharpe: 0.52,
        min_window_sharpe: 0.2, passed: 1 }),
  row({ id: 5, strategy_name: 'overfit_below', sharpe_oos: 0.11, mean_window_sharpe: 1.35,
        min_window_sharpe: 0.9, passed: 0 }),
  row({ id: 2, strategy_name: 'honest_neg', sharpe_oos: -0.24, mean_window_sharpe: -0.18,
        min_window_sharpe: -0.4, passed: 0 }),
  row({ id: 4, strategy_name: 'vol_carry_v2', sharpe_oos: null, mean_window_sharpe: null,
        min_window_sharpe: null, passed: null }),
]

// Server order for the alternate sort (`mean_window_sharpe`) — genuinely different content and
// order from the default, so a test can prove the chip actually re-requested rather than
// re-sorting the same payload client-side.
const altSortRuns: RunRow[] = [
  row({ id: 5, strategy_name: 'overfit_below', mean_window_sharpe: 1.35, sharpe_oos: 0.11,
        min_window_sharpe: 0.9, passed: 0 }),
  row({ id: 1, strategy_name: 'trend_breakout_v1', mean_window_sharpe: 0.52, sharpe_oos: 0.61,
        min_window_sharpe: 0.2, passed: 1 }),
  row({ id: 3, strategy_name: 'mined_above_1', mean_window_sharpe: 0.08, sharpe_oos: 1.42,
        min_window_sharpe: -0.1, passed: 0 }),
  row({ id: 2, strategy_name: 'honest_neg', mean_window_sharpe: -0.18, sharpe_oos: -0.24,
        min_window_sharpe: -0.4, passed: 0 }),
  row({ id: 4, strategy_name: 'vol_carry_v2', mean_window_sharpe: null, sharpe_oos: null,
        min_window_sharpe: null, passed: null }),
]

function envelope(runs: RunRow[]): ApiEnvelope<RunsListPayload> {
  return {
    ok: true,
    fetched_at: '2026-08-26T00:00:00Z',
    stale: false,
    data: { count: runs.length, runs },
  }
}

function stubRunsFetch(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const sort = new URL(url, 'http://x').searchParams.get('sort')
      const body = sort === 'mean_window_sharpe' ? envelope(altSortRuns) : envelope(defaultSortRuns)
      return { ok: true, status: 200, json: async () => body }
    }) as unknown as typeof fetch,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function renderRunList() {
  return render(
    <MemoryRouter>
      <RunList />
    </MemoryRouter>,
  )
}

it('renders rows in the exact server order — NULL sort-metric row sorts last, never outranking a real negative number', async () => {
  stubRunsFetch()
  renderRunList()
  const rows = await screen.findAllByTestId('run-row')
  expect(rows.length).toBe(5)
  const names = rows.map((r) => within(r).getByTestId('run-row-name').textContent)
  expect(names).toEqual([
    'mined_above_1',
    'trend_breakout_v1',
    'overfit_below',
    'honest_neg',
    'vol_carry_v2',
  ])
})

it('a chip changes sort order — re-requests the ledger rather than re-sorting client-side', async () => {
  stubRunsFetch()
  renderRunList()
  await screen.findByText('mined_above_1')
  const rowOrder = () => screen.getAllByTestId('run-row').map((r) => r.getAttribute('data-strategy'))
  expect(rowOrder()[0]).toBe('mined_above_1')

  fireEvent.click(screen.getByRole('button', { name: /is sharpe/i }))

  await screen.findByText('overfit_below')
  expect(rowOrder()).toEqual([
    'overfit_below',
    'trend_breakout_v1',
    'mined_above_1',
    'honest_neg',
    'vol_carry_v2',
  ])
})

it('tapping a row navigates to the strategy detail route — never a tooltip', async () => {
  stubRunsFetch()
  renderRunList()
  const link = (await screen.findByText('mined_above_1')).closest('a')
  expect(link?.getAttribute('href')).toBe('/s/mined_above_1')
})

it('renders a sparkline per row, including an honest empty one for the NULL-metric row', async () => {
  stubRunsFetch()
  renderRunList()
  const rows = await screen.findAllByTestId('run-row')
  expect(rows.length).toBe(5)
  for (const r of rows) {
    expect(within(r).getByTestId('sparkline')).toBeTruthy()
  }
  const nullRow = rows.find((r) => r.getAttribute('data-strategy') === 'vol_carry_v2')!
  expect(within(nullRow).getByTestId('sparkline').getAttribute('data-empty')).toBe('true')
})

it('renders a pass/fail mark from the row, never fabricating a verdict for the NULL run', async () => {
  stubRunsFetch()
  renderRunList()
  const rows = await screen.findAllByTestId('run-row')
  const passed = rows.find((r) => r.getAttribute('data-strategy') === 'trend_breakout_v1')!
  expect(within(passed).getByText('pass')).toBeTruthy()
  const failed = rows.find((r) => r.getAttribute('data-strategy') === 'honest_neg')!
  expect(within(failed).getByText('fail')).toBeTruthy()
  const unknown = rows.find((r) => r.getAttribute('data-strategy') === 'vol_carry_v2')!
  expect(within(unknown).getByText('unknown')).toBeTruthy()
})
