import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { clearFetchCache } from '../api'
import type { ApiEnvelope, StrategyBoardGate, StrategyBoardPayload, StrategyBoardRow } from '../types'
import StrategyBoard, { SORT_OPTIONS, orderBoardRows } from './StrategyBoard'

function gate(overrides: Partial<StrategyBoardGate> & { run_id: number }): StrategyBoardGate {
  return {
    created_at: '2026-10-15T12:00:00+00:00',
    passed: true,
    sharpe_oos: null,
    total_return_oos: null,
    max_drawdown_oos: null,
    n_obs_oos: null,
    mean_window_sharpe: null,
    min_window_sharpe: null,
    pct_positive_windows: null,
    ...overrides,
  }
}

function row(
  id: number,
  name: string,
  stage: string,
  latest_gate: StrategyBoardGate | null,
): StrategyBoardRow {
  return { strategy_id: id, name, family: null, stage, latest_gate }
}

// Lifecycle order, as the server returns it.
const rows: StrategyBoardRow[] = [
  row(1, 'bare_idea', 'idea', null),
  row(2, 'mined_above', 'paper', gate({ run_id: 902, sharpe_oos: 1.42, mean_window_sharpe: 0.08,
                                         min_window_sharpe: -0.1, total_return_oos: 0.2,
                                         max_drawdown_oos: -0.3, pct_positive_windows: 0.25,
                                         n_obs_oos: 126, passed: false })),
  row(3, 'honest_neg', 'paper', gate({ run_id: 903, sharpe_oos: -0.24, mean_window_sharpe: -0.18,
                                        min_window_sharpe: -0.4, total_return_oos: -0.02,
                                        max_drawdown_oos: -0.05, pct_positive_windows: 0.5,
                                        passed: false })),
  row(4, 'unmeasured', 'paper', gate({ run_id: 904, passed: null })),
  row(5, 'steady', 'live', gate({ run_id: 905, sharpe_oos: 0.61, mean_window_sharpe: 0.52,
                                   min_window_sharpe: 0.2, total_return_oos: 0.1,
                                   max_drawdown_oos: -0.08, pct_positive_windows: 0.75 })),
  row(6, 'tied_a', 'live', gate({ run_id: 906, sharpe_oos: 0.61, mean_window_sharpe: 0.3 })),
]

function env(payload: StrategyBoardPayload): ApiEnvelope<StrategyBoardPayload> {
  return { ok: true, fetched_at: '2026-10-19T00:00:00Z', stale: false, data: payload }
}

const populated = env({ schema_version: 'strategy-board/v1', count: rows.length, rows })

// `useFetch` keeps a module-level cache keyed by URL; every test brings its own payload.
beforeEach(() => clearFetchCache())

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function mockBoard(body: unknown, status = 200) {
  const fetchMock = vi.fn(async () => ({ ok: status < 400, status, json: async () => body }))
  vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch)
  return fetchMock
}

function renderBoard() {
  return render(
    <MemoryRouter>
      <StrategyBoard />
    </MemoryRouter>,
  )
}

const names = () => screen.getAllByTestId('board-row-name').map((el) => el.textContent)

it('renders exactly one row per registered strategy, including ones with no gate evidence', async () => {
  mockBoard(populated)
  renderBoard()
  expect(await screen.findAllByTestId('board-row')).toHaveLength(rows.length)
  expect(names()).toEqual(rows.map((r) => r.name))
  const bare = screen.getAllByTestId('board-row')[0]
  expect(within(bare).getByText(/no usable gate evidence/i)).toBeTruthy()
  expect(bare.querySelector('svg')).toBeNull()
})

it('keeps a missing metric unavailable instead of rendering it as zero', async () => {
  mockBoard(populated)
  renderBoard()
  await screen.findAllByTestId('board-row')
  const unmeasured = screen.getAllByTestId('board-row')[3]
  expect(unmeasured.textContent).not.toMatch(/\b0(\.0+)?\b/)
  expect(unmeasured.textContent).toContain('—')
})

it('links each row to the existing read-only strategy drill-down', async () => {
  mockBoard(populated)
  renderBoard()
  const first = (await screen.findAllByTestId('board-row'))[1]
  expect(first.tagName).toBe('A')
  expect(first.getAttribute('href')).toBe('/s/mined_above')
})

it('filters by stage without fetching again or touching representative runs', async () => {
  const fetchMock = mockBoard(populated)
  renderBoard()
  await screen.findAllByTestId('board-row')
  fireEvent.click(screen.getByRole('button', { name: /^live/ }))
  expect(screen.getByRole('button', { name: /^live/ }).getAttribute('aria-pressed')).toBe('true')
  expect(names()).toEqual(['steady', 'tied_a'])
  fireEvent.click(screen.getByRole('button', { name: /^all/ }))
  expect(names()).toEqual(rows.map((r) => r.name))
  expect(fetchMock).toHaveBeenCalledTimes(1)
})

it('orders by a chosen metric higher-first, nulls last, ties kept in lifecycle order', async () => {
  const fetchMock = mockBoard(populated)
  renderBoard()
  await screen.findAllByTestId('board-row')
  fireEvent.click(screen.getByRole('button', { name: 'oos sharpe' }))
  expect(names()).toEqual([
    'mined_above', 'steady', 'tied_a', 'honest_neg', 'bare_idea', 'unmeasured',
  ])
  expect(fetchMock).toHaveBeenCalledTimes(1)
})

it('never changes which run represents a strategy when the comparison order changes', async () => {
  mockBoard(populated)
  renderBoard()
  await screen.findAllByTestId('board-row')
  const runIds = () =>
    Object.fromEntries(
      screen.getAllByTestId('board-row').map((el) => [el.getAttribute('data-strategy'), el.getAttribute('data-run')]),
    )
  const before = runIds()
  for (const opt of SORT_OPTIONS) fireEvent.click(screen.getByRole('button', { name: opt.label }))
  expect(runIds()).toEqual(before)
  expect(before.mined_above).toBe('902')
  expect(before.bare_idea).toBe('')
})

it('states that comparison ordering is exploratory only, and returns to lifecycle order', async () => {
  mockBoard(populated)
  renderBoard()
  await screen.findAllByTestId('board-row')
  expect(screen.queryByText(/exploratory/i)).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'oos return' }))
  expect(screen.getByRole('button', { name: 'oos return' }).getAttribute('aria-pressed')).toBe('true')
  expect(screen.getByText(/exploratory/i)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'lifecycle' }))
  expect(names()).toEqual(rows.map((r) => r.name))
  expect(screen.queryByText(/exploratory/i)).toBeNull()
})

it('labels the oos and walk-forward samples', async () => {
  mockBoard(populated)
  renderBoard()
  await screen.findAllByTestId('board-row')
  expect(screen.getByText(/oos = holdout/i)).toBeTruthy()
  expect(screen.getByText(/wf = walk-forward/i)).toBeTruthy()
  expect(within(screen.getAllByTestId('board-row')[1]).getByText(/n=126/)).toBeTruthy()
})

it('shows a directed empty state for an empty registry, not an error', async () => {
  mockBoard(env({ schema_version: 'strategy-board/v1', count: 0, rows: [] }))
  renderBoard()
  expect(await screen.findByText(/no strategies registered/i)).toBeTruthy()
  expect(screen.getByText(/registry add/i)).toBeTruthy()
  expect(screen.queryByText(/fetch failed/i)).toBeNull()
})

it('never presents a stale empty registry as current', async () => {
  mockBoard({
    ...env({ schema_version: 'strategy-board/v1', count: 0, rows: [] }),
    stale: true,
    last_error_code: 'cli_timeout',
  })
  renderBoard()
  expect(await screen.findByText('stale data')).toBeTruthy()
  expect(screen.getByText(/last error cli_timeout/i)).toBeTruthy()
  expect(screen.getByText(/no strategies registered/i)).toBeTruthy()
})

it('shows the error treatment with retry on a backend failure — never a false empty board', async () => {
  const fetchMock = mockBoard({ ok: false, error: 'boom', code: 'cli_timeout' }, 502)
  renderBoard()
  expect(await screen.findByText('fetch failed')).toBeTruthy()
  expect(screen.getByText('cli_timeout')).toBeTruthy()
  expect(screen.queryByText(/no strategies registered/i)).toBeNull()
  expect(screen.queryByTestId('board-row')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'retry' }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
})

it('refuses an unknown schema version rather than guessing at its shape', async () => {
  mockBoard(env({ schema_version: 'strategy-board/v2', count: 1, rows } as unknown as StrategyBoardPayload))
  renderBoard()
  expect(await screen.findByText('fetch failed')).toBeTruthy()
  expect(screen.queryByTestId('board-row')).toBeNull()
})

it('refuses null or malformed versioned data instead of throwing while rendering', async () => {
  mockBoard({ ...populated, data: null })
  renderBoard()
  expect(await screen.findByText('unsupported_schema')).toBeTruthy()
  cleanup()
  clearFetchCache()
  mockBoard({ ...populated, data: { ...populated.data, count: 1, rows: [null] } })
  renderBoard()
  expect(await screen.findByText('unsupported_schema')).toBeTruthy()
})

it('orderBoardRows is a pure stable ordering that leaves the input untouched', () => {
  const copy = [...rows]
  expect(orderBoardRows(rows, null)).toEqual(rows)
  const ordered = orderBoardRows(rows, 'max_drawdown_oos')
  // Higher (shallower) drawdown first; nulls last.
  expect(ordered.slice(0, 3).map((r) => r.name)).toEqual(['honest_neg', 'steady', 'mined_above'])
  expect(ordered.slice(3).every((r) => r.latest_gate?.max_drawdown_oos == null)).toBe(true)
  expect(rows).toEqual(copy)
})

it('exposes no action that mutates state', async () => {
  mockBoard(populated)
  const { container } = renderBoard()
  await screen.findAllByTestId('board-row')
  expect(container.querySelector('form, input, select, textarea')).toBeNull()
  const labels = screen.getAllByRole('button').map((b) => b.textContent ?? '')
  expect(labels.join(' ')).not.toMatch(/promote|retire|allocate|trade|approve/i)
})
