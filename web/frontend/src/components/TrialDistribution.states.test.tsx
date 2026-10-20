import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearFetchCache } from '../api'
import type { ApiEnvelope, RunRow, RunsListPayload } from '../types'
import TrialDistribution from './TrialDistribution'

function envelope(runs: RunRow[], stale = false): ApiEnvelope<RunsListPayload> {
  return {
    ok: true,
    fetched_at: '2026-10-19T00:00:00Z',
    stale,
    data: { count: runs.length, runs },
  }
}

function trial(id: number, metric: number | null): RunRow {
  return {
    id,
    kind: 'sweep_trial',
    strategy_name: `trial_${id}`,
    strategy_id: id,
    created_at: '2026-10-19T00:00:00Z',
    passed: null,
    mean_window_sharpe: metric,
    sharpe_oos: null,
  }
}

function stubLists(trials: RunRow[], stale = false) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const rows = url.includes('kind=sweep_trial') ? trials : []
    return { ok: true, status: 200, json: async () => envelope(rows, stale) }
  }) as unknown as typeof fetch)
}

afterEach(() => {
  cleanup()
  clearFetchCache()
  vi.unstubAllGlobals()
})

describe('TrialDistribution source states', () => {
  it('distinguishes loading, failure, genuine empty and incomplete evidence', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})) as unknown as typeof fetch)
    const loading = render(<TrialDistribution strategy="loading_state" />)
    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('loading')
    loading.unmount()
    clearFetchCache()

    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false, status: 503, json: async () => ({ error: 'offline' }),
    })) as unknown as typeof fetch)
    const failed = render(<TrialDistribution strategy="failed_state" />)
    expect((await screen.findByTestId('evidence-status')).getAttribute('data-state')).toBe('failed')
    failed.unmount()
    clearFetchCache()

    stubLists([])
    const empty = render(<TrialDistribution strategy="empty_state" />)
    await waitFor(() => {
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('empty')
    })
    empty.unmount()
    clearFetchCache()

    stubLists([trial(81, null)])
    render(<TrialDistribution strategy="incomplete_state" />)
    await waitFor(() => {
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('incomplete')
    })
  })

  it('labels stale populated evidence without hiding its marks', async () => {
    stubLists([trial(82, 0.4)], true)
    render(<TrialDistribution strategy="stale_state" />)
    await waitFor(() => {
      expect(screen.getByText('showing cached trial evidence')).toBeTruthy()
    })
    expect(screen.getByTestId('trial-cloud-svg')).toBeTruthy()
  })

  it.each([
    ['empty', [], /cached trial evidence is empty/i],
    ['incomplete', [trial(83, null)], /cached trial evidence is incomplete/i],
  ] as const)('preserves stale freshness when cached trial evidence is %s', async (_, trials, detail) => {
    stubLists([...trials], true)
    render(<TrialDistribution strategy="thin_state" />)

    await waitFor(() => {
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
    })
    expect(screen.getByText(detail)).toBeTruthy()
    expect(screen.queryByTestId('trial-cloud-svg')).toBeNull()
  })

  it('shows gate-comparison loading without hiding a current trial cloud', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.includes('kind=gate')) return new Promise<Response>(() => {})
      return { ok: true, status: 200, json: async () => envelope([trial(84, 0.4)]) }
    }) as unknown as typeof fetch)

    render(<TrialDistribution strategy="loading_gate" />)
    expect(await screen.findByTestId('trial-cloud-svg')).toBeTruthy()
    expect(screen.getByText('gate comparison')).toBeTruthy()
    expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('loading')
  })

  it('retries the failed gate detail source without refetching the healthy lists', async () => {
    const counts = new Map<string, number>()
    const gate = { ...trial(185, 0.3), kind: 'gate' }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      counts.set(url, (counts.get(url) ?? 0) + 1)
      if (url === '/api/runs/185') {
        return { ok: false, status: 503, json: async () => ({ error: 'gate detail offline' }) }
      }
      if (url.includes('kind=gate')) {
        return { ok: true, status: 200, json: async () => envelope([gate as RunRow]) }
      }
      return { ok: true, status: 200, json: async () => envelope([trial(85, 0.4)]) }
    }) as unknown as typeof fetch)

    render(<TrialDistribution strategy="detail_failure" />)
    expect(await screen.findByText('gate detail offline')).toBeTruthy()
    expect(screen.getByTestId('trial-cloud-svg')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'retry' }))

    await waitFor(() => expect(counts.get('/api/runs/185')).toBe(2))
    expect([...counts.entries()].filter(([url]) => url.includes('kind=gate'))[0]?.[1]).toBe(1)
  })

  it('keeps cached gate comparison auxiliary to a current trial cloud', async () => {
    const gate = { ...trial(186, 0.3), kind: 'gate' }
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/runs/186') {
        return { ok: true, status: 200, json: async () => ({
          ok: true, fetched_at: '2026-10-19T00:00:00Z', stale: true,
          data: { ...gate, gate_decision: null },
        }) }
      }
      if (url.includes('kind=gate')) {
        return { ok: true, status: 200, json: async () => envelope([gate as RunRow]) }
      }
      return { ok: true, status: 200, json: async () => envelope([trial(86, 0.4)]) }
    }) as unknown as typeof fetch)

    render(<TrialDistribution strategy="stale_gate" />)
    expect(await screen.findByTestId('trial-cloud-svg')).toBeTruthy()
    expect(screen.getByText('gate comparison')).toBeTruthy()
    await waitFor(() => {
      expect(screen.getByTestId('evidence-status').getAttribute('data-state')).toBe('stale')
    })
    expect(screen.queryByText('showing cached trial evidence')).toBeNull()
  })

  it('treats a failed trial refresh as stale while retaining cached marks', async () => {
    let failRefresh = false
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.includes('kind=gate')) return { ok: true, status: 200, json: async () => envelope([]) }
      if (failRefresh) return { ok: false, status: 503, json: async () => ({ error: 'refresh offline' }) }
      return { ok: true, status: 200, json: async () => envelope([trial(87, 0.4)]) }
    }) as unknown as typeof fetch)

    render(<TrialDistribution strategy="refresh_failure" />)
    expect(await screen.findByTestId('trial-cloud-svg')).toBeTruthy()
    failRefresh = true
    document.dispatchEvent(new Event('visibilitychange'))

    expect(await screen.findByText('showing cached trial evidence')).toBeTruthy()
    expect(screen.getByTestId('trial-cloud-svg')).toBeTruthy()
  })
})
