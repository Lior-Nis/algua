import type { GateRow, StrategyDetailResponse } from '../types'

export type SeriesEvidenceState = 'loading' | 'current' | 'stale' | 'failed' | 'empty' | 'incomplete'

export interface StrategyDecision {
  state: string
  consequence: string
  reason: string
  nextCheckpoint: string
  tone: 'current' | 'warning' | 'blocked' | 'quiet'
}

type GateKind = 'research' | 'forward'

interface SelectedGate {
  kind: GateKind
  row: GateRow
}

function passed(row: GateRow): boolean {
  return row.passed === true || row.passed === 1
}

function checkpoint(stage: string): string {
  switch (stage) {
    case 'idea': return 'First reproducible backtest.'
    case 'backtested': return 'Next research gate evaluation.'
    case 'candidate': return 'Paper allocation record.'
    case 'paper': return 'Next forward-gate evaluation.'
    case 'forward_tested': return 'Human live decision.'
    case 'live': return 'Next completed-session health check.'
    case 'dormant': return 'Next lifecycle review.'
    case 'retired': return 'No further lifecycle checkpoint is scheduled.'
    default: return 'Next registry lifecycle record.'
  }
}

function compareGateRows(a: GateRow, b: GateRow): number {
  const aTime = Date.parse(a.created_at)
  const bTime = Date.parse(b.created_at)
  const timeOrder = (Number.isFinite(aTime) ? aTime : 0) - (Number.isFinite(bTime) ? bTime : 0)
  return timeOrder !== 0 ? timeOrder : a.id - b.id
}

function newest(rows: GateRow[]): GateRow | undefined {
  return rows.reduce<GateRow | undefined>(
    (latest, row) => latest === undefined || compareGateRows(row, latest) > 0 ? row : latest,
    undefined,
  )
}

function selectGate(
  stage: string,
  gates: NonNullable<StrategyDetailResponse['gates']>,
): SelectedGate | undefined {
  const research = newest(gates.gate_evaluations)
  const forward = newest(gates.forward_gate_evaluations)
  if (stage === 'forward_tested' || stage === 'live') {
    return forward === undefined ? undefined : { kind: 'forward', row: forward }
  }
  if (stage === 'paper' || stage === 'dormant' || stage === 'retired') {
    if (research === undefined) return forward === undefined ? undefined : { kind: 'forward', row: forward }
    if (forward === undefined) return { kind: 'research', row: research }
    return compareGateRows(forward, research) > 0
      ? { kind: 'forward', row: forward }
      : { kind: 'research', row: research }
  }
  return research === undefined ? undefined : { kind: 'research', row: research }
}

function gateEvidenceSummary(selected: SelectedGate): string {
  const advisory = selected.row.decision?.checks?.filter(
    (check) => check.advisory === true && check.passed === false,
  ) ?? []
  const binding = selected.row.decision?.checks?.filter(
    (check) => check.advisory !== true && check.passed === false,
  ) ?? []
  if (!passed(selected.row)) {
    const failures = binding.map((check) => check.name ?? 'unnamed check').join(', ')
    return `Latest ${selected.kind} gate failed.${failures ? ` Binding failures: ${failures}.` : ''}`
  }
  if (advisory.length > 0) {
    const failures = advisory.map((check) => check.name ?? 'unnamed check').join(', ')
    return `Latest ${selected.kind} gate passed; ${failures} failed as advisory evidence (non-vetoing).`
  }
  return `Latest ${selected.kind} gate passed.`
}

/** Presentation-only synthesis; it never changes a verdict or infers promotion eligibility. */
export function deriveStrategyDecision(
  detail: StrategyDetailResponse,
  seriesState: SeriesEvidenceState,
): StrategyDecision {
  const stage = detail.registry?.stage ?? 'unknown'
  const gates = detail.gates
  const selected = gates === null ? undefined : selectGate(stage, gates)
  const forwardRequired = stage === 'forward_tested' || stage === 'live'
  const advisoryFailures = selected?.row.decision?.checks?.filter(
    (check) => check.advisory === true && check.passed === false,
  ) ?? []
  const bindingFailures = selected?.row.decision?.checks?.filter(
    (check) => check.advisory !== true && check.passed === false,
  ) ?? []

  const sourceIssues: string[] = []
  if (detail.registry == null) sourceIssues.push('registry record unavailable')
  if (detail.stale) sourceIssues.push('strategy detail is stale')
  if (detail.part_errors?.gates !== undefined || gates === null) sourceIssues.push('gate history unavailable')
  if (detail.part_errors?.paper !== undefined) sourceIssues.push('paper state unavailable')
  if (forwardRequired && gates !== null && selected === undefined) sourceIssues.push('required forward gate is missing')
  if (selected !== undefined && selected.row.decision === null) {
    sourceIssues.push(`${selected.kind} gate decision unavailable`)
  }
  if (seriesState === 'failed') sourceIssues.push('equity series unavailable')
  if (seriesState === 'stale') sourceIssues.push('equity series stale')
  if (seriesState === 'loading') sourceIssues.push('equity series loading')

  const paperHealth = detail.paper?.health
  const paperNeedsAttention = paperHealth !== undefined && !['ok', 'idle'].includes(paperHealth)
  const hasAdvisoryConflict = selected !== undefined
    && selected.row.decision !== null
    && passed(selected.row)
    && advisoryFailures.length > 0
  const gateConclusionBlocked = detail.registry == null
    || gates === null
    || detail.part_errors?.gates !== undefined
    || (forwardRequired && selected === undefined)
    || selected?.row.decision === null

  if (paperNeedsAttention) {
    const evidence = selected !== undefined && selected.row.decision !== null
      ? ` ${gateEvidenceSummary(selected)}`
      : ''
    const incomplete = sourceIssues.length > 0 ? ` ${sourceIssues.join('; ')}.` : ''
    return {
      state: `${stage} · ${paperHealth}`,
      consequence: 'Operational evidence needs attention.',
      reason: `Paper health is ${paperHealth}.${evidence}${incomplete}`,
      nextCheckpoint: checkpoint(stage),
      tone: 'blocked',
    }
  }

  if (gateConclusionBlocked) {
    return {
      state: `${stage} · partial`,
      consequence: 'Decision evidence is incomplete.',
      reason: `${sourceIssues.join('; ')}.`,
      nextCheckpoint: checkpoint(stage),
      tone: 'blocked',
    }
  }

  if (hasAdvisoryConflict && selected !== undefined) {
    const names = advisoryFailures.map((check) => check.name ?? 'unnamed check').join(', ')
    const label = selected.kind === 'research' ? 'Research' : 'Forward'
    return {
      state: `${stage} · ${sourceIssues.length > 0 ? 'partial' : 'advisory'}`,
      consequence: `${label} gate passed; advisory weakness remains.`,
      reason: `${names} failed as advisory evidence and does not veto the gate.` +
        (sourceIssues.length > 0 ? ` ${sourceIssues.join('; ')}.` : ''),
      nextCheckpoint: checkpoint(stage),
      tone: 'warning',
    }
  }

  if (selected !== undefined && !passed(selected.row)) {
    const names = bindingFailures.map((check) => check.name ?? 'unnamed check')
    return {
      state: `${stage} · ${sourceIssues.length > 0 ? 'partial' : 'gate fail'}`,
      consequence: `Latest ${selected.kind} gate failed.`,
      reason: (names.length > 0 ? `Binding failures: ${names.join(', ')}.` : 'No passing gate verdict is recorded.') +
        (sourceIssues.length > 0 ? ` ${sourceIssues.join('; ')}.` : ''),
      nextCheckpoint: checkpoint(stage),
      tone: 'blocked',
    }
  }

  if (selected !== undefined) {
    return {
      state: `${stage} · ${sourceIssues.length > 0 ? 'partial' : 'gate pass'}`,
      consequence: `Latest ${selected.kind} gate passed.`,
      reason: sourceIssues.length > 0
        ? `No failed advisory check is recorded. ${sourceIssues.join('; ')}.`
        : 'No failed advisory check is recorded.',
      nextCheckpoint: checkpoint(stage),
      tone: 'current',
    }
  }

  if (sourceIssues.length > 0) {
    return {
      state: `${stage} · partial`,
      consequence: 'Decision evidence is incomplete.',
      reason: `${sourceIssues.join('; ')}.`,
      nextCheckpoint: checkpoint(stage),
      tone: 'blocked',
    }
  }

  return {
    state: `${stage} · unknown`,
    consequence: 'No gate verdict is available.',
    reason: 'Gate history is genuinely empty.',
    nextCheckpoint: checkpoint(stage),
    tone: 'quiet',
  }
}
