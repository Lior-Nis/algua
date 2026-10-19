import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearFetchCache } from '../api'
import type { ApiEnvelope, RunRow, RunSeriesEntry, RunSeriesPayload, RunsListPayload } from '../types'
import ReturnOverlay from './ReturnOverlay'

vi.mock('uplot', () => {
  class FakeUPlot { destroy(): void {} }
  return { default: FakeUPlot }
})

function envelope<T>(data: T, stale = false): ApiEnvelope<T> {
  return { ok: true, fetched_at: '2026-10-19T00:00:00Z', stale, data }
}

function run(id: number): RunRow {
  return {
    id,
    kind: 'backtest',
    strategy_name: `state_${id}`,
    strategy_id: id,
    created_at: '2026-10-19T00:00:00Z',
    passed: null,
    mean_window_sharpe: null,
    sharpe_oos: null,
  }
}

function curveSeries(): RunSeriesEntry {
  return {
    kind: 'backtest',
    period_start: '2026-01-01',
    period_end: '2026-01-02',
    returns: [['2026-01-01', 0.01], ['2026-01-02', -0.005]],
  }
}

function stubData(opts: { runs: RunRow[]; series?: Record<string, RunSeriesEntry>; stale?: boolean }) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.startsWith('/api/runs/series')) {
      const data: RunSeriesPayload = { series: opts.series ?? {} }
      return { ok: true, status: 200, json: async () => envelope(data, opts.stale) }
    }
    if (url.includes('kind=gate')) {
      const data: RunsListPayload = { count: 0, runs: [] }
      return { ok: true, status: 200, json: async () => envelope(data, opts.stale) }
    }
    const data: RunsListPayload = { count: opts.runs.length, runs: opts.runs }
    return { ok: true, status: 200, json: async () => envelope(data, opts.stale) }
  }) as unknown as typeof fetch)
}

afterEach(() => {
  cleanup()
  clearFetchCache()
  vi.unstubAllGlobals()
})

describe('ReturnOverlay source states', () => {
  it('distinguishes loading, failure, genuine empty and incomplete evidence', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})) as unknown as typeof fetch)
    const loading = render(<ReturnOverlay strategy="loading_state" family={null} />)
    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('loading')
    loading.unmount()
    clearFetchCache()

    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false, status: 503, json: async () => ({ error: 'offline' }),
    })) as unknown as typeof fetch)
    const failed = render(<ReturnOverlay strategy="failed_state" family={null} />)
    expect((await screen.findByTestId('evidence-status')).getAttribute('data-state')).toBe('failed')
    failed.unmount()
    clearFetchCache()

    stubData({ runs: [] })
    const empty = render(<ReturnOverlay strategy="empty_state" family={null} />)
    await waitFor(() => {
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('empty')
    })
    empty.unmount()
    clearFetchCache()

    stubData({ runs: [run(91)] })
    render(<ReturnOverlay strategy="state_91" family={null} />)
    await waitFor(() => {
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('incomplete')
    })
  })

  it('labels stale populated evidence while retaining the plot', async () => {
    const series = curveSeries()
    stubData({ runs: [run(92)], series: { '92': series }, stale: true })
    render(<ReturnOverlay strategy="state_92" family={null} />)
    await waitFor(() => {
      expect(screen.getByText('showing cached return evidence')).toBeTruthy()
    })
    expect(screen.getByTestId('overlay-end-label')).toBeTruthy()
  })

  it.each([
    ['empty', [], {}, /cached return evidence is empty/i],
    ['incomplete', [run(93)], {}, /cached return evidence is incomplete/i],
  ] as const)('preserves stale freshness when cached primary evidence is %s', async (_, runs, series, detail) => {
    stubData({ runs: [...runs], series, stale: true })
    render(<ReturnOverlay strategy="state_93" family={null} />)

    await waitFor(() => {
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
    })
    expect(screen.getByText(detail)).toBeTruthy()
    expect(screen.queryByTestId('overlay-end-label')).toBeNull()
  })

  it('shows gate-context loading without hiding a current primary curve', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.includes('kind=gate')) return new Promise<Response>(() => {})
      if (url.startsWith('/api/runs/series')) {
        return { ok: true, status: 200, json: async () => envelope({ series: { '94': curveSeries() } }) }
      }
      return { ok: true, status: 200, json: async () => envelope({ count: 1, runs: [run(94)] }) }
    }) as unknown as typeof fetch)

    render(<ReturnOverlay strategy="state_94" family={null} />)
    expect(await screen.findByTestId('overlay-end-label')).toBeTruthy()
    expect(screen.getByText('holdout context')).toBeTruthy()
    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('loading')
  })

  it('retries the failed gate detail source without refetching the healthy run lists', async () => {
    const counts = new Map<string, number>()
    const gate = { ...run(195), kind: 'gate' }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      counts.set(url, (counts.get(url) ?? 0) + 1)
      if (url === '/api/runs/195') {
        return { ok: false, status: 503, json: async () => ({ error: 'detail offline' }) }
      }
      if (url.includes('kind=gate')) {
        return { ok: true, status: 200, json: async () => envelope({ count: 1, runs: [gate] }) }
      }
      if (url.startsWith('/api/runs/series')) {
        return { ok: true, status: 200, json: async () => envelope({ series: { '95': curveSeries() } }) }
      }
      return { ok: true, status: 200, json: async () => envelope({ count: 1, runs: [run(95)] }) }
    }) as unknown as typeof fetch)

    render(<ReturnOverlay strategy="state_95" family={null} />)
    expect(await screen.findByText('detail offline')).toBeTruthy()
    expect(screen.getByTestId('overlay-end-label')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'retry' }))
    await waitFor(() => expect(counts.get('/api/runs/195')).toBe(2))
    expect([...counts.entries()].filter(([url]) => url.includes('kind=gate'))[0]?.[1]).toBe(1)
  })

  it('keeps gate staleness auxiliary when the primary return curve is current', async () => {
    const gate = { ...run(196), kind: 'gate' }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/runs/196') {
        return { ok: true, status: 200, json: async () => envelope({ sharpe_oos: 0.4 }, true) }
      }
      if (url.includes('kind=gate')) {
        return { ok: true, status: 200, json: async () => envelope({ count: 1, runs: [gate] }) }
      }
      if (url.startsWith('/api/runs/series')) {
        return { ok: true, status: 200, json: async () => envelope({ series: { '96': curveSeries() } }) }
      }
      return { ok: true, status: 200, json: async () => envelope({ count: 1, runs: [run(96)] }) }
    }) as unknown as typeof fetch)

    render(<ReturnOverlay strategy="state_96" family={null} />)
    expect(await screen.findByTestId('overlay-end-label')).toBeTruthy()
    expect(screen.getByText('holdout context')).toBeTruthy()
    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
    expect(screen.queryByText('showing cached return evidence')).toBeNull()
  })

  it('treats a failed backtest refresh as stale while retaining cached primary evidence', async () => {
    let failRefresh = false
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.includes('kind=gate')) {
        return { ok: true, status: 200, json: async () => envelope({ count: 0, runs: [] }) }
      }
      if (url.startsWith('/api/runs/series')) {
        return { ok: true, status: 200, json: async () => envelope({ series: { '97': curveSeries() } }) }
      }
      if (failRefresh) return { ok: false, status: 503, json: async () => ({ error: 'refresh offline' }) }
      return { ok: true, status: 200, json: async () => envelope({ count: 1, runs: [run(97)] }) }
    }) as unknown as typeof fetch)

    render(<ReturnOverlay strategy="state_97" family={null} />)
    expect(await screen.findByTestId('overlay-end-label')).toBeTruthy()
    failRefresh = true
    document.dispatchEvent(new Event('visibilitychange'))

    await waitFor(() => {
      expect(screen.getByText('showing cached return evidence')).toBeTruthy()
    })
    expect(screen.getByTestId('overlay-end-label')).toBeTruthy()
  })
})
