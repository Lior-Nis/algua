import { describe, expect, it } from 'vitest'
import type { StrategyDetailResponse } from '../types'
import { deriveStrategyDecision } from './strategyDecision'

function gate({
  id,
  passed,
  createdAt,
  checks = [],
  decision = true,
}: {
  id: number
  passed: boolean
  createdAt: string
  checks?: NonNullable<NonNullable<StrategyDetailResponse['gates']>['gate_evaluations'][number]['decision']>['checks']
  decision?: boolean
}): NonNullable<StrategyDetailResponse['gates']>['gate_evaluations'][number] {
  return {
    id,
    passed,
    actor: 'agent',
    created_at: createdAt,
    decision: decision ? { passed, checks } : null,
  }
}

function detail(overrides: Partial<StrategyDetailResponse> = {}): StrategyDetailResponse {
  return {
    ok: true,
    strategy: 'mom_breakout',
    registry: {
      id: 1,
      name: 'mom_breakout',
      stage: 'candidate',
      family: 'momentum',
      tags: [],
      author: 'agent',
      hypothesis_status: 'untested',
      derived_from: null,
      description: null,
      transitions: [],
    },
    paper: null,
    gates: {
      strategy: 'mom_breakout',
      gate_evaluations: [{
        id: 9,
        passed: 1,
        consumed: 0,
        actor: 'agent',
        created_at: '2026-08-02T15:00:00+00:00',
        decision: {
          passed: true,
          checks: [
            { name: 'pit_universe', passed: true },
            { name: 'holdout_sharpe', passed: false, advisory: true },
          ],
        },
      }],
      forward_gate_evaluations: [],
    },
    part_errors: {},
    fetched_at: '2026-08-09T14:30:00Z',
    stale: false,
    ...overrides,
  }
}

describe('deriveStrategyDecision', () => {
  it('keeps the passing verdict while naming a failed advisory weakness', () => {
    const decision = deriveStrategyDecision(detail(), 'empty')
    expect(decision.state).toBe('candidate · advisory')
    expect(decision.consequence).toBe('Research gate passed; advisory weakness remains.')
    expect(decision.reason).toContain('holdout_sharpe')
    expect(decision.reason).toContain('does not veto')
    expect(decision.nextCheckpoint).toBe('Paper allocation record.')
  })

  it('fails closed when a required source is unavailable without inventing a verdict', () => {
    const decision = deriveStrategyDecision(
      detail({ gates: null, part_errors: { gates: 'cli_failed' } }),
      'failed',
    )
    expect(decision.state).toBe('candidate · partial')
    expect(decision.consequence).toBe('Decision evidence is incomplete.')
    expect(decision.reason).toMatch(/gate history unavailable.*equity series unavailable/i)
  })

  it('uses the newer forward failure on paper rather than hiding it behind a research pass', () => {
    const base = detail()
    const decision = deriveStrategyDecision(detail({
      registry: { ...base.registry, stage: 'paper' },
      gates: {
        strategy: base.strategy,
        gate_evaluations: [gate({ id: 10, passed: true, createdAt: '2026-08-01T00:00:00Z' })],
        forward_gate_evaluations: [gate({
          id: 11,
          passed: false,
          createdAt: '2026-08-03T00:00:00Z',
          checks: [{ name: 'paper_sessions', passed: false }],
        })],
      },
    }), 'current')
    expect(decision.state).toBe('paper · gate fail')
    expect(decision.consequence).toBe('Latest forward gate failed.')
    expect(decision.reason).toContain('paper_sessions')
  })

  it.each(['forward_tested', 'live'])('%s requires forward evidence and never falls back to research', (stage) => {
    const base = detail()
    const decision = deriveStrategyDecision(detail({
      registry: { ...base.registry, stage },
      gates: {
        strategy: base.strategy,
        gate_evaluations: [gate({ id: 10, passed: true, createdAt: '2026-08-01T00:00:00Z' })],
        forward_gate_evaluations: [],
      },
    }), 'current')
    expect(decision.state).toBe(`${stage} · partial`)
    expect(decision.consequence).toBe('Decision evidence is incomplete.')
    expect(decision.reason).toMatch(/required forward gate.*missing/i)
  })

  it('treats a null projected decision as incomplete even when the row passed flag is true', () => {
    const base = detail()
    const decision = deriveStrategyDecision(detail({
      gates: {
        strategy: base.strategy,
        gate_evaluations: [gate({
          id: 10,
          passed: true,
          createdAt: '2026-08-01T00:00:00Z',
          decision: false,
        })],
        forward_gate_evaluations: [],
      },
    }), 'current')
    expect(decision.state).toBe('candidate · partial')
    expect(decision.reason).toMatch(/research gate decision unavailable/i)
  })

  it('treats a missing registry record as incomplete', () => {
    const decision = deriveStrategyDecision(detail({
      registry: null as unknown as StrategyDetailResponse['registry'],
    }), 'current')
    expect(decision.state).toBe('unknown · partial')
    expect(decision.reason).toMatch(/registry record unavailable/i)
  })

  it('leads with unhealthy paper operations while retaining advisory gate evidence', () => {
    const base = detail()
    const decision = deriveStrategyDecision(detail({
      registry: { ...base.registry, stage: 'paper' },
      paper: {
        strategy: base.strategy,
        stage: 'paper',
        health: 'halted',
        staleness_sessions: 2,
        stale_after_sessions: 1,
        last_tick_error: 'reconcile failed',
        kill_switch: null,
        drawdown: null,
        last_tick: null,
        positions: null,
        n_orders: null,
        recent_orders: [],
      },
    }), 'current')
    expect(decision.state).toBe('paper · halted')
    expect(decision.consequence).toBe('Operational evidence needs attention.')
    expect(decision.reason).toMatch(/^Paper health is halted\./)
    expect(decision.reason).toMatch(/research gate passed.*holdout_sharpe.*advisory/i)
  })
})
