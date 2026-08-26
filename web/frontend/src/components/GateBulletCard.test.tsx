import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import type { GateCheck } from '../types'
import GateBulletCard, { buildBulletGeometry, splitChecks } from './GateBulletCard'

afterEach(() => {
  cleanup()
})

it('renders exactly the checks it is given — no fetch, no waterfall (fix round 2: the card ' +
  'takes `checks` as a prop; `StrategyDetail` feeds it `row.decision.checks`, already fetched)', () => {
  render(
    <GateBulletCard
      checks={[{ name: 'pit_universe', op: '==', threshold: null, value: null, passed: true }]}
    />,
  )
  expect(screen.getByText('pit_universe')).toBeTruthy()
})

it('a binding fail and an advisory fail are distinguishable with colour stripped', () => {
  render(
    <GateBulletCard
      checks={[
        { name: 'holdout_sharpe_floor', op: '>=', threshold: 0.3, value: 0.1, passed: false },
        { name: 'dsr_evidence', op: '>=', threshold: 2.0, value: 0.5, passed: false, advisory: true },
      ]}
    />,
  )
  const bindingRow = screen.getByText('holdout_sharpe_floor').closest(
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

it('a NULL-value check reads "no numeric value", never 0 — NOT "not evaluated", since a ' +
  'boolean/None-by-construction/fail-closed check genuinely ran (fix round 2)', () => {
  render(
    <GateBulletCard
      checks={[{ name: 'pit_universe', op: '==', threshold: null, value: null, passed: true }]}
    />,
  )
  const row = screen.getByText('pit_universe').closest(
    '[data-testid="gate-check-row"]',
  ) as HTMLElement

  expect(within(row).getByText('no numeric value')).toBeTruthy()
  expect(within(row).queryByText('not evaluated')).toBeNull()
  expect(within(row).queryByTestId('bullet-fill')).toBeNull()
  expect(within(row).queryByText('0')).toBeNull()
})

it('a FAILED verdict and a NULL value are independent: both the fail mark and "no numeric ' +
  'value" render together', () => {
  // Regression: a check with no numeric value (e.g. a fail-closed non-finite Sharpe) can still
  // carry a real `false` verdict — e.g. "returns not available" auto-fails a check with nothing
  // to compare. The fail mark must not disappear just because there is no bar to draw, and "no
  // numeric value" must not disappear just because the verdict is a real fail. If PassMark were
  // ever moved inside the `geometry === null` branch (coupling the two), this test catches it.
  render(
    <GateBulletCard
      checks={[{ name: 'holdout_sharpe_floor', op: '>', threshold: 0.0, value: null, passed: false }]}
    />,
  )
  const row = screen.getByText('holdout_sharpe_floor').closest(
    '[data-testid="gate-check-row"]',
  ) as HTMLElement

  expect(within(row).getByText('fail')).toBeTruthy()
  expect(within(row).getByText('no numeric value')).toBeTruthy()
  expect(within(row).queryByTestId('bullet-fill')).toBeNull()
})

it('an honest empty state when there are no checks (e.g. no gate run yet)', () => {
  render(<GateBulletCard checks={[]} />)
  expect(screen.getByText(/no gate checks recorded yet/i)).toBeTruthy()
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

it('splitChecks partitions binding vs advisory, preserving order within each group', () => {
  const checks: GateCheck[] = [
    { name: 'a', passed: true },
    { name: 'b', passed: false, advisory: true },
    { name: 'c', passed: true },
    { name: 'd', passed: false, advisory: true },
  ]
  const { binding, advisory } = splitChecks(checks)
  expect(binding.map((c) => c.name)).toEqual(['a', 'c'])
  expect(advisory.map((c) => c.name)).toEqual(['b', 'd'])
})

it('splitChecks handles an empty list without throwing', () => {
  expect(splitChecks([])).toEqual({ binding: [], advisory: [] })
})
