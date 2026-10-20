import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { UseFetchResult } from '../api'
import type { ApiEnvelope, RunsListPayload } from '../types'
import { ScatterISOOSView } from './ScatterISOOS'

function query(
  data: ApiEnvelope<RunsListPayload> | undefined,
  overrides: Partial<UseFetchResult<ApiEnvelope<RunsListPayload>>> = {},
): UseFetchResult<ApiEnvelope<RunsListPayload>> {
  return {
    data,
    error: null,
    loading: false,
    refetch: vi.fn(),
    ...overrides,
  }
}

function envelope(runs: RunsListPayload['runs'], stale = false): ApiEnvelope<RunsListPayload> {
  return {
    ok: true,
    fetched_at: '2026-10-19T00:00:00Z',
    stale,
    data: { count: runs.length, runs },
  }
}

afterEach(cleanup)

describe('ScatterISOOS source states', () => {
  it('distinguishes loading, failed, incomplete and genuinely empty without a false chart', () => {
    const cases = [
      ['loading', query(undefined, { loading: true })],
      ['failed', query(undefined, { error: new Error('offline') })],
      ['empty', query(envelope([]))],
      ['incomplete', query(envelope([{
        id: 1,
        kind: 'gate',
        strategy_name: 'thin',
        strategy_id: 1,
        created_at: '2026-10-19T00:00:00Z',
        passed: null,
        mean_window_sharpe: null,
        sharpe_oos: null,
      }]))],
    ] as const

    for (const [state, value] of cases) {
      const view = render(<ScatterISOOSView query={value} />)
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe(state)
      expect(view.container.querySelector('svg')).toBeNull()
      view.unmount()
    }
  })

  it('labels stale populated evidence without hiding its trustworthy marks', () => {
    render(<ScatterISOOSView query={query(envelope([{
      id: 1,
      kind: 'gate',
      strategy_name: 'measured',
      strategy_id: 1,
      created_at: '2026-10-19T00:00:00Z',
      passed: true,
      mean_window_sharpe: 0.4,
      sharpe_oos: 0.5,
    }], true))} />)
    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
    expect(screen.getByTestId('diagonal')).toBeTruthy()
  })

  it.each([
    ['empty', [], /cached gate evidence is empty/i],
    ['incomplete', [{
      id: 1,
      kind: 'gate',
      strategy_name: 'thin',
      strategy_id: 1,
      created_at: '2026-10-19T00:00:00Z',
      passed: null,
      mean_window_sharpe: null,
      sharpe_oos: null,
    }], /cached gate evidence is incomplete/i],
  ] as const)('preserves stale freshness when cached evidence is %s', (_, runs, detail) => {
    render(<ScatterISOOSView query={query(envelope([...runs], true))} />)

    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
    expect(screen.getByText(detail)).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('treats a failed refresh with cached marks as stale and keeps the marks visible', () => {
    render(<ScatterISOOSView query={query(envelope([{
      id: 1,
      kind: 'gate',
      strategy_name: 'measured',
      strategy_id: 1,
      created_at: '2026-10-19T00:00:00Z',
      passed: true,
      mean_window_sharpe: 0.4,
      sharpe_oos: 0.5,
    }]), { error: new Error('refresh failed') })} />)

    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
    expect(screen.getByTestId('diagonal')).toBeTruthy()
  })
})
