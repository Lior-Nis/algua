import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import type { ApiEnvelope, GateCheck, RunDetail, RunRow, RunsListPayload } from '../types'
import GateBulletCard, { buildBulletGeometry } from './GateBulletCard'

function listEnvelope(runs: RunRow[]): ApiEnvelope<RunsListPayload> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: { count: runs.length, runs } }
}

function detailEnvelope(detail: RunDetail): ApiEnvelope<RunDetail> {
  return { ok: true, fetched_at: '2026-08-26T00:00:00Z', stale: false, data: detail }
}

function runRow(id: number, strategy: string): RunRow {
  return {
    id,
    kind: 'gate',
    strategy_name: strategy,
    strategy_id: 1,
    created_at: '2026-08-25T00:00:00+00:00',
    passed: 0,
    mean_window_sharpe: null,
    sharpe_oos: null,
  }
}

function runDetail(id: number, strategy: string, checks: GateCheck[]): RunDetail {
  return {
    ...runRow(id, strategy),
    extra_metrics: {},
    gate_decision: { passed: false, checks },
  }
}

/** Stubs the two-call waterfall the card issues: `/api/runs?...&kind=gate&limit=1` (find the
 * latest gate run id for the strategy) then `/api/runs/{id}` (its checks). */
function stubGateFetch(strategy: string, runId: number, checks: GateCheck[]): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return { ok: true, status: 200, json: async () => detailEnvelope(runDetail(runId, strategy, checks)) }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([runRow(runId, strategy)]) }
    }) as unknown as typeof fetch,
  )
}

function stubNoGateRuns(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, json: async () => listEnvelope([]) })) as unknown as typeof fetch,
  )
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('fetches the latest gate run for the strategy, then its detail — never requests more than the allowlist gives it', async () => {
  const calls: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      calls.push(url)
      if (/^\/api\/runs\/\d+$/.test(url)) {
        return {
          ok: true,
          status: 200,
          json: async () =>
            detailEnvelope(
              runDetail(42, 'mom_breakout', [
                { name: 'pit_universe', op: '==', threshold: null, value: null, passed: true },
              ]),
            ),
        }
      }
      return { ok: true, status: 200, json: async () => listEnvelope([runRow(42, 'mom_breakout')]) }
    }) as unknown as typeof fetch,
  )

  render(<GateBulletCard strategy="mom_breakout" />)
  await screen.findByText('pit_universe')

  expect(calls.some((u) => u.startsWith('/api/runs?') && u.includes('kind=gate'))).toBe(true)
  expect(calls).toContain('/api/runs/42')
})

it('a binding fail and an advisory fail are distinguishable with colour stripped', async () => {
  // Own strategy name + run id (distinct from the other tests in this file): `useFetch`'s
  // cache is module-level and keyed by URL, and vitest shares one module graph per FILE —
  // reusing another test's URL would silently serve ITS cached response instead of this stub.
  stubGateFetch('gate_probe_binding_advisory', 101, [
    { name: 'holdout_sharpe_floor', op: '>=', threshold: 0.3, value: 0.1, passed: false },
    { name: 'dsr_evidence', op: '>=', threshold: 2.0, value: 0.5, passed: false, advisory: true },
  ])

  render(<GateBulletCard strategy="gate_probe_binding_advisory" />)
  const bindingRow = (await screen.findByText('holdout_sharpe_floor')).closest(
    '[data-testid="gate-check-row"]',
  ) as HTMLElement
  const advisoryRow = screen.getByText('dsr_evidence').closest(
    '[data-testid="gate-check-row"]',
  ) as HTMLElement

  // Strip color: read only data-attributes and text content, never `fill`/`color`/`style`.
  expect(bindingRow.getAttribute('data-kind')).toBe('binding')
  expect(advisoryRow.getAttribute('data-kind')).toBe('advisory')

  // Explicit text label — survives colour being stripped entirely.
  expect(within(bindingRow).getByText('binding')).toBeTruthy()
  expect(within(advisoryRow).getByText('advisory')).toBeTruthy()

  // Position: the two kinds render in separate groups, not intermixed rows.
  const bindingGroup = screen.getByTestId('bullet-group-binding')
  const advisoryGroup = screen.getByTestId('bullet-group-advisory')
  expect(within(bindingGroup).queryByText('dsr_evidence')).toBeNull()
  expect(within(advisoryGroup).queryByText('holdout_sharpe_floor')).toBeNull()

  // Weight: the bar geometry itself (an attribute, not a fill color) differs by kind.
  const bindingFill = within(bindingRow).getByTestId('bullet-fill')
  const advisoryFill = within(advisoryRow).getByTestId('bullet-fill')
  expect(bindingFill.getAttribute('height')).not.toBe(advisoryFill.getAttribute('height'))
})

it('a NULL-value check reads "not evaluated", never 0', async () => {
  stubGateFetch('gate_probe_null_value', 102, [
    { name: 'pit_universe', op: '==', threshold: null, value: null, passed: true },
  ])

  render(<GateBulletCard strategy="gate_probe_null_value" />)
  const row = (await screen.findByText('pit_universe')).closest(
    '[data-testid="gate-check-row"]',
  ) as HTMLElement

  expect(within(row).getByText('not evaluated')).toBeTruthy()
  expect(within(row).queryByTestId('bullet-fill')).toBeNull()
  expect(within(row).queryByText('0')).toBeNull()
})

it('mounts inside ChartFrame — an honest empty state when the strategy has no gate run yet', async () => {
  stubNoGateRuns()
  render(<GateBulletCard strategy="never_gated" />)
  expect(await screen.findByText(/no gate checks recorded yet/i)).toBeTruthy()
  expect(document.querySelector('svg')).toBeNull()
})

it('buildBulletGeometry returns null for a NULL value (never scales a fabricated zero)', () => {
  expect(buildBulletGeometry({ name: 'x', value: null, threshold: 1 })).toBeNull()
  expect(buildBulletGeometry({ name: 'x', threshold: 1 })).toBeNull()
})

it('buildBulletGeometry spans from zero to the value, oriented correctly for sign', () => {
  const pos = buildBulletGeometry({ name: 'x', value: 1, threshold: 0.5 })!
  expect(pos.barX1).toBeGreaterThan(pos.barX0)
  expect(pos.domainMin).toBeLessThanOrEqual(0)
  expect(pos.domainMax).toBeGreaterThanOrEqual(1)

  const neg = buildBulletGeometry({ name: 'x', value: -1, threshold: 0.5 })!
  expect(neg.barX1).toBeGreaterThan(neg.barX0)
  expect(neg.domainMin).toBeLessThanOrEqual(-1)
  expect(neg.domainMax).toBeGreaterThanOrEqual(0)
})
