import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, RunDetail, RunRow, RunSeriesEntry, RunSeriesPayload, RunsListPayload } from '../types'
import ReturnOverlay, { buildAlignedData, buildReturnOverlayGeometry } from './ReturnOverlay'

// uPlot renders to <canvas>, which this jsdom environment cannot inspect (no `canvas` package
// installed — see EquityChart.test.tsx's precedent of never mounting `<Plot>` with real data).
// Stub the whole module so the component tree renders fully and every DOM-visible thing this
// slice actually cares about (end labels, region band, panel count) is assertable, without ever
// touching real canvas.
vi.mock('uplot', () => {
  class FakeUPlot {
    destroy(): void {}
  }
  return { default: FakeUPlot }
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function listEnvelope(runs: RunRow[]): ApiEnvelope<RunsListPayload> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: { count: runs.length, runs } }
}

function seriesEnvelope(series: Record<string, RunSeriesEntry>): ApiEnvelope<RunSeriesPayload> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: { series } }
}

function detailEnvelope(detail: RunDetail): ApiEnvelope<RunDetail> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: detail }
}

function backtestRun(id: number, strategy: string, createdAt: string): RunRow {
  return {
    id,
    kind: 'backtest',
    strategy_name: strategy,
    strategy_id: 1,
    created_at: createdAt,
    passed: null,
    mean_window_sharpe: null,
    sharpe_oos: null,
  }
}

function gateRun(id: number, strategy: string): RunRow {
  return {
    id,
    kind: 'gate',
    strategy_name: strategy,
    strategy_id: 1,
    created_at: '2026-08-20T00:00:00+00:00',
    passed: 1,
    mean_window_sharpe: 0.4,
    sharpe_oos: 0.42,
  }
}

/** Real per-bar `[iso_date, daily_return]` pairs, one per day starting at `periodStart` — mirrors
 * the actual backend shape (`persist_backtest_returns`), not a bare number array. */
function backtestSeries(periodStart: string, periodEnd: string, n: number): RunSeriesEntry {
  const start = new Date(`${periodStart}T00:00:00Z`)
  const returns: [string, number][] = Array.from({ length: n }, (_, i) => {
    const d = new Date(start.getTime() + i * 86_400_000)
    return [d.toISOString().slice(0, 10), i % 2 === 0 ? 0.01 : -0.005]
  })
  return { kind: 'backtest', period_start: periodStart, period_end: periodEnd, returns }
}

function holdoutSeries(start: string, end: string, nBars: number): RunSeriesEntry {
  return { kind: 'holdout', holdout_start: start, holdout_end: end, n_bars: nBars }
}

/** Stubs the full waterfall: family/strategy backtest list, gate list, batched series, gate
 * detail. Routes on substring so callers only need to supply the parts a given test cares
 * about. */
function stubFetch(opts: {
  backtests?: RunRow[]
  gate?: RunRow | null
  series?: Record<string, RunSeriesEntry>
  gateDetail?: RunDetail | null
}): { calls: string[] } {
  const calls: string[] = []
  const backtests = opts.backtests ?? []
  const gate = opts.gate ?? null
  const series = opts.series ?? {}
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      calls.push(url)
      if (url.startsWith('/api/runs/series')) {
        return { ok: true, status: 200, json: async () => seriesEnvelope(series) }
      }
      if (/^\/api\/runs\/\d+$/.test(url)) {
        const detail = opts.gateDetail ?? (gate ? { ...gate, extra_metrics: {} } : null)
        return { ok: true, status: 200, json: async () => detailEnvelope(detail as RunDetail) }
      }
      if (url.includes('kind=gate')) {
        return { ok: true, status: 200, json: async () => listEnvelope(gate ? [gate] : []) }
      }
      // kind=backtest (family- or strategy-scoped) falls through to the backtest list
      return { ok: true, status: 200, json: async () => listEnvelope(backtests) }
    }) as unknown as typeof fetch,
  )
  return { calls }
}

// ---- pure geometry builder -------------------------------------------------------------------

it('two family backtest runs produce two prepared curves: the viewed strategy is focus, the sibling is context', () => {
  const runs = [backtestRun(1, 'mom_pullback', '2026-08-01T00:00:00+00:00'), backtestRun(2, 'mom', '2026-08-20T00:00:00+00:00')]
  const seriesById: Record<string, RunSeriesEntry> = {
    '1': backtestSeries('2023-01-02', '2023-12-29', 8),
    '2': backtestSeries('2024-01-02', '2024-12-30', 10),
  }
  const geo = buildReturnOverlayGeometry(runs, seriesById, null, null, 'mom')
  expect(geo.mode).toBe('overlay')
  expect(geo.curves).toHaveLength(2)
  const byRunId = new Map(geo.curves.map((c) => [c.runId, c]))
  expect(byRunId.get(2)?.role).toBe('focus') // run 2 belongs to 'mom', the viewed strategy
  expect(byRunId.get(1)?.role).toBe('context') // run 1 belongs to its sibling 'mom_pullback'
  expect(geo.excludedCurveCount).toBe(0)
})

it('three family backtest runs trigger small-multiples mode, not a 3-series overlay', () => {
  const runs = [
    backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00'),
    backtestRun(2, 'mom_b', '2026-08-19T00:00:00+00:00'),
    backtestRun(3, 'mom_c', '2026-08-18T00:00:00+00:00'),
  ]
  const seriesById: Record<string, RunSeriesEntry> = {
    '1': backtestSeries('2024-01-02', '2024-12-30', 6),
    '2': backtestSeries('2023-01-02', '2023-12-29', 6),
    '3': backtestSeries('2022-01-03', '2022-12-30', 6),
  }
  const geo = buildReturnOverlayGeometry(runs, seriesById, null, null, 'mom')
  expect(geo.mode).toBe('small-multiples')
  expect(geo.curves).toHaveLength(3)
  // The viewed strategy's own curve still carries identity even in small multiples — it just
  // means "which panel gets --series-focus", not a second series sharing its panel.
  const byRunId = new Map(geo.curves.map((c) => [c.runId, c]))
  expect(byRunId.get(1)?.role).toBe('focus')
  expect(byRunId.get(2)?.role).toBe('context')
  expect(byRunId.get(3)?.role).toBe('context')
})

it('a run with no persisted series is excluded and counted, never plotted as a flat/zero curve', () => {
  const runs = [backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00'), backtestRun(2, 'mom_b', '2026-08-01T00:00:00+00:00')]
  const seriesById: Record<string, RunSeriesEntry> = {
    '1': backtestSeries('2024-01-02', '2024-12-30', 10),
    '2': null, // unregistered-at-the-time backtest: no series was ever persisted
  }
  const geo = buildReturnOverlayGeometry(runs, seriesById, null, null, 'mom')
  expect(geo.curves).toHaveLength(1)
  expect(geo.excludedCurveCount).toBe(1)
})

it('the holdout leg is prepared as an interval (start/end), never a per-bar array', () => {
  const runs = [backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00')]
  const gate = gateRun(9, 'mom')
  const seriesById: Record<string, RunSeriesEntry> = {
    '1': backtestSeries('2024-01-02', '2024-12-30', 10),
    '9': holdoutSeries('2025-01-06', '2025-04-01', 63),
  }
  const detail: RunDetail = { ...gate, extra_metrics: {}, sharpe_oos: 0.42, total_return_oos: 0.081 } as RunDetail
  const geo = buildReturnOverlayGeometry(runs, seriesById, gate, detail, 'mom')
  expect(geo.region).not.toBeNull()
  expect(geo.region).toEqual(
    expect.objectContaining({
      runId: 9,
      startTs: expect.any(Number),
      endTs: expect.any(Number),
    }),
  )
  // The interval is a scalar pair, not a series — no array-shaped field on the region at all.
  for (const value of Object.values(geo.region as object)) {
    expect(Array.isArray(value)).toBe(false)
  }
  expect(geo.region?.label).toMatch(/63 bars/)
  expect(geo.region?.label).toMatch(/0\.42/)
})

it('THE HARD RULE: no array of per-bar OOS values reaches the geometry or the component at all', () => {
  const runs = [backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00')]
  const gate = gateRun(9, 'mom')
  // Simulates a backend that (incorrectly) leaked a holdout returns array — the geometry
  // builder must never read or forward such a field even if it were present on the entry.
  const leaky = {
    kind: 'holdout',
    holdout_start: '2025-01-06',
    holdout_end: '2025-04-01',
    n_bars: 63,
    returns: [0.001, -0.002, 0.003], // must never be read
  } as unknown as RunSeriesEntry
  const seriesById: Record<string, RunSeriesEntry> = {
    '1': backtestSeries('2024-01-02', '2024-12-30', 10),
    '9': leaky,
  }
  const geo = buildReturnOverlayGeometry(runs, seriesById, gate, null, 'mom')
  const serialized = JSON.stringify(geo)
  expect(serialized).not.toContain('0.001')
  expect(serialized).not.toContain('-0.002')
  // The region object itself carries only scalars.
  expect(geo.region).not.toHaveProperty('returns')
})

it('a curve with fewer than 2 parseable per-bar dates is excluded rather than plotting a fabricated axis', () => {
  const runs = [backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00')]
  const seriesById: Record<string, RunSeriesEntry> = {
    '1': { kind: 'backtest', period_start: '2024-01-02', period_end: '2024-01-03', returns: [['not-a-date', 0.01]] },
  }
  const geo = buildReturnOverlayGeometry(runs, seriesById, null, null, 'mom')
  expect(geo.curves).toHaveLength(0)
  expect(geo.excludedCurveCount).toBe(1)
  expect(geo.mode).toBe('empty')
})

it('one unparseable bar within an otherwise-good series drops only that bar, not the whole curve', () => {
  const runs = [backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00')]
  const good = backtestSeries('2024-01-02', '2024-01-05', 4) as Extract<RunSeriesEntry, { kind: 'backtest' }>
  const withOneBadBar: RunSeriesEntry = {
    ...good,
    returns: [['garbage-date', 0.01], ...good.returns],
  }
  const seriesById: Record<string, RunSeriesEntry> = { '1': withOneBadBar }
  const geo = buildReturnOverlayGeometry(runs, seriesById, null, null, 'mom')
  expect(geo.curves).toHaveLength(1)
  expect(geo.curves[0].ts).toHaveLength(4) // the 4 good bars, not 5
  expect(geo.excludedCurveCount).toBe(0)
})

it('buildAlignedData null-gaps non-overlapping curves rather than zero-filling them', () => {
  const runs = [backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00'), backtestRun(2, 'mom_b', '2026-08-01T00:00:00+00:00')]
  const seriesById: Record<string, RunSeriesEntry> = {
    '1': backtestSeries('2024-06-01', '2024-06-10', 4),
    '2': backtestSeries('2023-01-01', '2023-01-10', 4),
  }
  const geo = buildReturnOverlayGeometry(runs, seriesById, null, null, 'mom')
  const aligned = buildAlignedData(geo.curves) as unknown as [number[], ...(number | null)[][]]
  const [xs, ...ySeries] = aligned
  expect(xs.length).toBeGreaterThan(4) // union of two non-overlapping date ranges
  for (const ys of ySeries) {
    expect(ys.some((v) => v === null)).toBe(true) // gaps, not fabricated zeros
    expect(ys.every((v) => v !== 0)).toBe(true)
  }
})

// ---- component: DOM-visible structure (never rendered canvas marks) -----------------------

it('renders the honest empty state before any backtest run has been recorded', async () => {
  stubFetch({ backtests: [] })
  render(<ReturnOverlay strategy="never_backtested" family={null} />)
  expect(await screen.findByText(/no backtest runs recorded yet/i)).toBeTruthy()
  expect(document.querySelector('canvas')).toBeNull()
})

it('two family runs render two direct end-labels, focus and context distinguishable without color', async () => {
  const runs = [backtestRun(1, 'mom', '2026-08-20T00:00:00+00:00'), backtestRun(2, 'mom_sibling', '2026-08-01T00:00:00+00:00')]
  stubFetch({
    backtests: runs,
    series: {
      '1': backtestSeries('2024-01-02', '2024-12-30', 10),
      '2': backtestSeries('2023-01-02', '2023-12-29', 8),
    },
  })
  render(<ReturnOverlay strategy="mom" family="momentum" />)
  const labels = await screen.findAllByTestId('overlay-end-label')
  expect(labels).toHaveLength(2)
  const roles = labels.map((l) => l.getAttribute('data-role')).sort()
  expect(roles).toEqual(['context', 'focus'])
  expect(document.querySelectorAll('[data-testid="overlay-panel"]')).toHaveLength(0) // single-panel mode, not small multiples
})

it('three family runs render three separate panels, not one chart with three series', async () => {
  const runs = [
    backtestRun(1, 'mom3', '2026-08-20T00:00:00+00:00'),
    backtestRun(2, 'mom3_b', '2026-08-19T00:00:00+00:00'),
    backtestRun(3, 'mom3_c', '2026-08-18T00:00:00+00:00'),
  ]
  stubFetch({
    backtests: runs,
    series: {
      '1': backtestSeries('2024-01-02', '2024-12-30', 6),
      '2': backtestSeries('2023-01-02', '2023-12-29', 6),
      '3': backtestSeries('2022-01-03', '2022-12-30', 6),
    },
  })
  render(<ReturnOverlay strategy="mom3" family="momentum3" />)
  const panels = await screen.findAllByTestId('overlay-panel')
  expect(panels).toHaveLength(3)
})

it('in small multiples, the OOS region lands on the FOCUS panel, not whichever panel is listed first', async () => {
  // The family query is newest-first and a sibling ('mom_older') is newest here — a
  // position-based ("always panel 0") region placement would land on the WRONG strategy's panel.
  const runs = [
    backtestRun(1, 'mom_older', '2026-08-20T00:00:00+00:00'),
    backtestRun(2, 'mom_focus', '2026-08-19T00:00:00+00:00'),
    backtestRun(3, 'mom_third', '2026-08-18T00:00:00+00:00'),
  ]
  // Gate id 19 — distinct from every other gate id in this file (`/api/runs/{id}` is keyed by
  // this id alone, unqualified by strategy, so a reused id would hit another test's cache).
  const gate = gateRun(19, 'mom_focus')
  stubFetch({
    backtests: runs,
    gate,
    series: {
      '1': backtestSeries('2024-01-02', '2024-12-30', 6),
      '2': backtestSeries('2023-01-02', '2023-12-29', 6),
      '3': backtestSeries('2022-01-03', '2022-12-30', 6),
      '19': holdoutSeries('2025-01-06', '2025-04-01', 63),
    },
    gateDetail: { ...gate, extra_metrics: {}, sharpe_oos: 0.5, total_return_oos: 0.05 } as RunDetail,
  })
  render(<ReturnOverlay strategy="mom_focus" family="momentum_region" />)
  const panels = await screen.findAllByTestId('overlay-panel')
  expect(panels).toHaveLength(3)
  // Exactly one panel carries the region, and it is the one with the focus end-label.
  const panelsWithRegion = panels.filter((p) => p.querySelector('[data-testid="overlay-region"]') !== null)
  expect(panelsWithRegion).toHaveLength(1)
  const focusLabel = panelsWithRegion[0].querySelector('[data-testid="overlay-end-label"][data-role="focus"]')
  expect(focusLabel).not.toBeNull()
  expect(focusLabel?.textContent).toBe('mom_focus')
})

it('with no family, the strategy renders its own curve alone (no context line)', async () => {
  const runs = [backtestRun(1, 'lone_wolf', '2026-08-20T00:00:00+00:00')]
  stubFetch({
    backtests: runs,
    series: { '1': backtestSeries('2024-01-02', '2024-12-30', 10) },
  })
  render(<ReturnOverlay strategy="lone_wolf" family={null} />)
  const labels = await screen.findAllByTestId('overlay-end-label')
  expect(labels).toHaveLength(1)
  expect(labels[0].getAttribute('data-role')).toBe('focus')
})

it('a gate run with a holdout leg renders the OOS region and its scalar label, never a curve for it', async () => {
  const runs = [backtestRun(1, 'mom_oos', '2026-08-20T00:00:00+00:00')]
  const gate = gateRun(9, 'mom_oos')
  stubFetch({
    backtests: runs,
    gate,
    series: {
      '1': backtestSeries('2024-01-02', '2024-12-30', 10),
      '9': holdoutSeries('2025-01-06', '2025-04-01', 63),
    },
    gateDetail: { ...gate, extra_metrics: {}, sharpe_oos: 0.42, total_return_oos: 0.081 } as RunDetail,
  })
  render(<ReturnOverlay strategy="mom_oos" family={null} />)
  const region = await screen.findByTestId('overlay-region-label')
  expect(region.textContent).toMatch(/63 bars/)
  expect(region.textContent).toMatch(/0\.42/)
  // Exactly one curve (the backtest's own in-sample line) plus the region — never a second
  // "line" standing in for the gate run's holdout leg.
  expect(screen.getAllByTestId('overlay-end-label')).toHaveLength(1)
})

it('fetches runs/series in ONE batched request, never one call per run id', async () => {
  // Distinct run ids from every other test in this file: `useFetch`'s cache is module-level and
  // keyed by URL, and vitest shares one module graph per file — reusing another test's
  // `/api/runs/series?ids=...` URL would silently serve ITS cached response instead of hitting
  // this stub (same hazard GateBulletCard.test.tsx documents for its own waterfall).
  const runs = [
    backtestRun(101, 'mom_batch', '2026-08-20T00:00:00+00:00'),
    backtestRun(102, 'mom_batch_sibling', '2026-08-01T00:00:00+00:00'),
  ]
  const { calls } = stubFetch({
    backtests: runs,
    series: {
      '101': backtestSeries('2024-01-02', '2024-12-30', 10),
      '102': backtestSeries('2023-01-02', '2023-12-29', 8),
    },
  })
  render(<ReturnOverlay strategy="mom_batch" family="batch_family" />)
  await screen.findAllByTestId('overlay-end-label')
  const seriesCalls = calls.filter((u) => u.startsWith('/api/runs/series'))
  expect(seriesCalls).toHaveLength(1)
  expect(seriesCalls[0]).toContain('101')
  expect(seriesCalls[0]).toContain('102')
})
