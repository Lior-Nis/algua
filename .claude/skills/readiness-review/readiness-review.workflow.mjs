export const meta = {
  name: 'readiness-review',
  description: 'Whole-repo 11-lane readiness audit: fan out lane finders, adversarially verify, dedup, auto-file GitHub issues, and commit a synthesis verdict',
  phases: [
    { title: 'Context' },
    { title: 'Find' },
    { title: 'Verify' },
    { title: 'Dedup' },
    { title: 'File' },
    { title: 'Synthesize' },
  ],
}

// ---- config (args-driven; no Date/Math.random in workflow scripts) ----
const DRY_RUN = !!(args && args.dryRun)
const LANE_SUBSET = (args && args.lanes) || null
const RID = (args && args.runId) || 'run'
const LANES_DOC = '.claude/skills/readiness-review/lanes.md'

const ALL_LANES = [
  'swe', 'mle', 'ds', 'qf', 'clean-code', 'agentic',
  'ml-dl-integration', 'risk-safe-scaling', 'model-risk-management',
  'security', 'observability',
]
const LANES = LANE_SUBSET ? ALL_LANES.filter((l) => LANE_SUBSET.includes(l)) : ALL_LANES

// ---- schemas ----
const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          lane: { type: 'string' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          files: { type: 'array', items: { type: 'string' } },
          description: { type: 'string' },
          why_it_matters: { type: 'string' },
          north_star_link: { type: 'string', enum: ['ml_readiness', 'safe_scale', 'both', 'none'] },
          recommendation: { type: 'string' },
          kb_citation: { type: 'string' },
          web_citation: { type: 'string' },
        },
        required: ['lane', 'title', 'severity', 'files', 'description', 'why_it_matters', 'north_star_link', 'recommendation', 'kb_citation', 'web_citation'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    is_real: { type: 'boolean' },
    already_mitigated: { type: 'boolean' },
    duplicate_of_open_issue: { type: ['integer', 'null'] },
    evidence_confirmed: { type: 'boolean' },
    refutation_reason: { type: 'string' },
    keep: { type: 'boolean' },
  },
  required: ['is_real', 'already_mitigated', 'duplicate_of_open_issue', 'evidence_confirmed', 'refutation_reason', 'keep'],
}

const DEDUP_SCHEMA = {
  type: 'object',
  properties: {
    kept: { type: 'array', items: FINDING_SCHEMA.properties.findings.items },
    dropped: {
      type: 'array',
      items: {
        type: 'object',
        properties: { title: { type: 'string' }, reason: { type: 'string' } },
        required: ['title', 'reason'],
      },
    },
  },
  required: ['kept', 'dropped'],
}

const FILED_SCHEMA = {
  type: 'object',
  properties: {
    filed: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          number: { type: 'integer' },
          title: { type: 'string' },
          lane: { type: 'string' },
          severity: { type: 'string' },
        },
        required: ['number', 'title', 'lane', 'severity'],
      },
    },
  },
  required: ['filed'],
}

// ---- Phase 0: shared context brief ----
phase('Context')
const context = await agent(
  `You are the context-pack agent for a readiness audit of the algua repo (cwd = repo root).
Produce a DENSE shared brief the lane finders will rely on. Include:
1. A 10-15 line summary of the architecture spec at docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md (read it).
2. The hard invariants and review mandate from AGENTS.md (read it) — list them tersely.
3. Module inventory: one line per directory under algua/ describing its responsibility.
4. Currently OPEN issue titles — run: gh issue list --state open --limit 300 --json number,title
Return ALL of this as one markdown brief. Dense, no preamble.`,
  { label: 'context-pack', phase: 'Context', model: 'sonnet' },
)
