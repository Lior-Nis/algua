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

// ---- Phases 1-2: per-lane find, then adversarially verify each finding ----
phase('Find')
const laneResults = await pipeline(
  LANES,
  // stage 1: find
  (lane) =>
    agent(
      `You are the "${lane}" readiness-review finder for the algua repo (cwd = repo root).

SHARED CONTEXT BRIEF:
${context}

STEP 1 — read your rubric: open the file ${LANES_DOC} and locate the section headed "## ${lane}".
It names your KB anchor directories under /home/liornisimov/KB, what to audit, and your north-star focus.

GROUNDING CONTRACT (mandatory):
1. Read your KB anchor notes (start with each anchor dir's CLAUDE.md, then relevant notes).
2. Run WebSearch for CURRENT best practices for your lane; cite at least one URL per finding.
3. Audit the algua repo through your lens. Every finding MUST cite concrete file:line evidence.
   If you cannot point to specific code/config, it is NOT a finding — stay silent.
4. Tag each finding's north_star_link: how it affects ML/DL-strategy integration readiness and/or
   risk management for safe scale.
5. Findings must be concrete and actionable (an implementer could act on the recommendation).

Return your findings in the required schema. Set every finding's "lane" to "${lane}".`,
      { label: `find:${lane}`, phase: 'Find', schema: FINDING_SCHEMA },
    ),
  // stage 2: adversarially verify each finding; return only kept findings for this lane
  (found, lane) => {
    const findings = (found && found.findings) || []
    if (!findings.length) return []
    return parallel(
      findings.map((f, i) => () =>
        agent(
          `You are an adversarial verifier. Try to REFUTE this readiness-review finding.
Default to keep=false when uncertain. cwd = repo root.

FINDING (lane ${f.lane}):
title: ${f.title}
severity: ${f.severity}
files: ${JSON.stringify(f.files)}
description: ${f.description}
why_it_matters: ${f.why_it_matters}
recommendation: ${f.recommendation}

Do ALL of:
1. Open each cited file:line and confirm the defect actually exists as described. If the citation is
   wrong/vague, set evidence_confirmed=false.
2. Search for guards/tests that ALREADY mitigate this concern. If mitigated, already_mitigated=true.
3. Run: gh issue list --state open --limit 300 --json number,title — if an open issue already covers
   this, set duplicate_of_open_issue to its number, else null.
4. keep = is_real AND evidence_confirmed AND NOT already_mitigated AND duplicate_of_open_issue is null.

Return the verdict schema.`,
          { label: `verify:${lane}#${i + 1}`, phase: 'Verify', schema: VERDICT_SCHEMA },
        ).then((v) => ({ finding: f, verdict: v })),
      ),
    ).then((rs) =>
      rs
        .filter(Boolean)
        .filter((r) => r.verdict && r.verdict.keep)
        .map((r) => r.finding),
    )
  },
)

const survivors = laneResults.filter(Boolean).flat()
log(`Verified survivors: ${survivors.length} across ${LANES.length} lane(s)`)

// ---- Phase 3: dedup survivors vs each other and vs open issues (barrier) ----
phase('Dedup')
let deduped = survivors
if (survivors.length > 1) {
  const dd = await agent(
    `You are the dedup agent (cwd = repo root). Below are ${survivors.length} verified
readiness-review findings as JSON. Also run: gh issue list --state open --limit 300 --json number,title,body

Tasks:
1. Collapse near-duplicate findings (same root defect) into one; keep the highest severity and merge
   their file lists.
2. Drop any finding already covered by an OPEN issue.
3. For EVERY finding you drop (duplicate of another finding or of an open issue), add a row to
   "dropped" with a one-line reason. Never silently drop.

Findings JSON:
${JSON.stringify(survivors)}

Return the dedup schema: { kept: [...findings...], dropped: [{title, reason}] }.`,
    { label: 'dedup', phase: 'Dedup', schema: DEDUP_SCHEMA },
  )
  deduped = (dd && dd.kept) || survivors
  if (dd && dd.dropped) dd.dropped.forEach((d) => log(`dropped: ${d.title} — ${d.reason}`))
}
log(`After dedup: ${deduped.length} to file`)
