# readiness-review Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a committed project skill that audits the whole `algua` repo across 11 evaluation lanes (fan-out finders → adversarial verify → dedup → auto-file GitHub issues → commit a north-star synthesis verdict), grounded per-lane in `~/KB` + web best practices.

**Architecture:** A `SKILL.md` invocable skill whose engine is a committed **Workflow script** (`.mjs`). The skill invokes `Workflow({scriptPath, args:{runId}})`. Lane rubrics live in a separate `lanes.md` the finder subagents Read at runtime (Workflow scripts have no filesystem access, so rubrics cannot be embedded via fs — finders read the file themselves). The pipeline pipelines Find→Verify per lane, barriers on Dedup, then Files and Synthesizes.

**Tech Stack:** Markdown skill + reference docs; a JavaScript Workflow script (ESM, `.mjs`); `gh` CLI for issue/label creation; `node --check` for syntax validation.

## Global Constraints

- Skill package dir: `.claude/skills/readiness-review/` (committed to the algua repo).
- Workflow script MUST begin with `export const meta = {...}` as a **pure literal** (no computed values, no interpolation).
- Workflow scripts have **no filesystem/Node API** and MUST NOT use `Date.now()`/`Math.random()`/argless `new Date()` — the run date is passed in via `args.runId`.
- Finders MUST obey the grounding contract: cite `file:line`; cite ≥1 KB note AND ≥1 web source; fail toward "not a finding" when evidence is absent.
- KB root is absolute: `/home/liornisimov/KB`.
- Auto-file has **no volume cap**; every deduped survivor is filed; every drop is logged (never silent).
- Synthesis verdict is **committed but NOT pushed**. GitHub issues are created via `gh` regardless.
- Commit trailers on any commit this skill makes:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and the `Claude-Session:` line.
- Never `git add -A` (concurrent-session WIP hazard) — add only the specific files.
- The 11 lane slugs (canonical, used everywhere): `swe`, `mle`, `ds`, `qf`, `clean-code`, `agentic`, `ml-dl-integration`, `risk-safe-scaling`, `model-risk-management`, `security`, `observability`.

---

### Task 1: Skill package + SKILL.md

**Files:**
- Create: `.claude/skills/readiness-review/SKILL.md`

**Interfaces:**
- Produces: the invocable skill. Documents the invocation
  `Workflow({ scriptPath: '.claude/skills/readiness-review/readiness-review.workflow.mjs', args: { runId: '<YYYY-MM-DD>', dryRun?: bool, lanes?: string[] } })`.

- [ ] **Step 1: Write `SKILL.md`**

```markdown
---
name: readiness-review
description: Use to run a whole-repo readiness audit of algua across 11 lanes (SWE, MLE, DS, QF, clean-code, agentic-ops, ML/DL integration readiness, risk & safe-scaling, model risk management, security, observability). Fans out one finder subagent per lane grounded in ~/KB + web best practices, adversarially verifies every finding, dedups against open issues, auto-files GitHub issues, and commits a north-star readiness verdict. Trigger on "readiness review", "audit the repo", "11-lane review", "is this ready to scale / for ML models".
---

# Readiness Review — 11-lane whole-repo audit

Audits the `algua` repo across 11 lanes, oriented by two north-star themes every lane must
address: (1) **ML/DL-model-as-strategy integration readiness** and (2) **risk management for safe
scale**. Output: adversarially-verified GitHub issues (labeled by lane + severity) + a committed
synthesis verdict.

## Preconditions
- cwd is the repo root (`/home/liornisimov/Projects/algua`).
- `gh auth status` succeeds (issues are filed via `gh`).
- The KB exists at `/home/liornisimov/KB`.

## How to run
This skill authorizes and runs a Workflow. Invoke it with today's date as `runId`:

    Workflow({
      scriptPath: '.claude/skills/readiness-review/readiness-review.workflow.mjs',
      args: { runId: '<TODAY as YYYY-MM-DD>' }
    })

Optional args:
- `dryRun: true` — run Context+Find+Verify+Dedup and PRINT what it would file; files nothing,
  commits nothing. Use to preview before a real run.
- `lanes: ['security', ...]` — restrict to a subset of the 11 lane slugs.

Lane slugs: `swe mle ds qf clean-code agentic ml-dl-integration risk-safe-scaling
model-risk-management security observability`.

## What the pipeline does
1. **Context** — one agent builds a shared brief (architecture spec, AGENTS.md invariants, module
   inventory, open-issue titles).
2. **Find** — 11 finders (parallel). Each reads its lane section in
   `.claude/skills/readiness-review/lanes.md`, reads its KB anchor, runs WebSearch for current best
   practices, and audits the repo. Grounding contract: every finding cites `file:line` + ≥1 KB note
   + ≥1 web source; no evidence ⇒ not a finding.
3. **Verify** — one adversarial skeptic per finding tries to refute it (confirm the code defect,
   check it isn't already mitigated, check it isn't already an open issue). Only survivors proceed.
4. **Dedup** — collapse near-duplicates and drop anything matching an open issue; drops are logged.
5. **File** — ensure labels exist, then `gh issue create` per survivor (no cap).
6. **Synthesize** — write + commit (no push) `docs/superpowers/specs/<runId>-readiness-review-verdict.md`.

## After the run
Relay to the user: the returned `{ survivors, filed, dryRun }` summary, the path of the committed
verdict doc, and a note that issues were created on GitHub but nothing was pushed.
```

- [ ] **Step 2: Verify frontmatter parses and required fields exist**

Run:
```bash
python3 - <<'PY'
import re,sys
t=open('.claude/skills/readiness-review/SKILL.md').read()
m=re.match(r'^---\n(.*?)\n---\n', t, re.S)
assert m, "no frontmatter block"
fm=m.group(1)
assert re.search(r'^name:\s*readiness-review\s*$', fm, re.M), "name missing/wrong"
assert re.search(r'^description:\s*\S', fm, re.M), "description missing"
print("SKILL.md frontmatter OK")
PY
```
Expected: `SKILL.md frontmatter OK`

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/readiness-review/SKILL.md
git commit -m "feat(skill): readiness-review SKILL.md — 11-lane audit entrypoint

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VnGemnoHubNa8JRQpZucPC"
```

---

### Task 2: Lane rubrics (`lanes.md`)

**Files:**
- Create: `.claude/skills/readiness-review/lanes.md`

**Interfaces:**
- Consumes: nothing.
- Produces: 11 lane sections keyed by the canonical slugs. Each finder subagent Reads this file and
  locates its `## <slug>` section. Each section has: KB anchor(s) (absolute paths), "Audit for"
  bullets, and "North-star focus".

- [ ] **Step 1: Write `lanes.md`**

````markdown
# Readiness-review lane rubrics

Each finder reads ONLY its own `## <slug>` section below, plus the shared context brief.
Grounding contract (all lanes): cite `file:line`; read your KB anchor; run WebSearch for current
best practices and cite ≥1 URL; a finding without concrete evidence is not a finding. Every finding
must state its `north_star_link`: `ml_readiness`, `safe_scale`, `both`, or `none`.

## swe — Software engineering / architecture
KB anchor: `/home/liornisimov/KB/software-engineering/04-software-design`,
`.../07-backend-engineering`, `.../11-error-handling-resilience`,
`.../14-concurrency-async-distributed`, `.../06-database-design`
Audit for: module boundaries & coupling, error-handling/resilience, concurrency & transactional
correctness (BEGIN IMMEDIATE, TOCTOU), DB schema/migration discipline, dependency-boundary
violations (`lint-imports`), interface clarity. Also test-suite adequacy (folded in).
North-star focus: does the architecture admit an ML/DL strategy without breaking boundaries; do
concurrency/transaction patterns hold as strategy count scales?

## mle — ML engineering (infra / serving)
KB anchor: `/home/liornisimov/KB/machine-learning-engineering/`
Audit for: is there a training→artifact→serving path at all; model versioning/registry; reproducible
training; feature pipeline parity (train vs serve); inference in the decision/order path (latency,
determinism, failure modes); data provenance & reproducibility (folded in).
North-star focus: what MLE infra is MISSING for ML/DL strategies to be first-class; scaling of model
artifacts/inference.

## ds — Data science (stats / experimentation)
KB anchor: `/home/liornisimov/KB/data-science/`
Audit for: statistical validity of the gates (DSR, FDR/LORD++, deflation), experiment tracking,
metric definitions, sample-size/power floors, reproducibility of experiments, data leakage in
feature construction (folded in).
North-star focus: are the statistical gates sound enough to admit ML models (which overfit harder);
does experiment infra scale to many models?

## qf — Quant finance
KB anchor: `/home/liornisimov/KB/quant-trading/13-alpha-research`, `.../15-backtesting`,
`.../16-metrics-and-risk`, `.../09-leakage-and-bias`, `.../10-splitting-and-validation`,
`.../17-costs-slippage-execution`, `/home/liornisimov/KB/finance/`
Audit for: backtest realism (costs/slippage/fills/market impact), backtest↔live parity (folded in),
survivorship/look-ahead/PIT correctness, walk-forward/holdout discipline, position sizing & capital
allocation, corporate-action handling.
North-star focus: do cost/parity models hold for ML strategies (often higher turnover); does capital
allocation stay safe as strategies scale?

## clean-code — Clean code / maintainability
KB anchor: `/home/liornisimov/KB/software-engineering/17-code-quality-maintainability`,
`.../04-software-design`
Audit for: readability, naming, function/file size & responsibility, dead code / compat cruft /
dual paths, duplication, comment quality, cyclomatic complexity, test readability.
North-star focus: is the code maintainable enough that ML-strategy additions won't rot it; does
complexity scale sub-linearly with features?

## agentic — Agentic operation readiness
KB anchor: `/home/liornisimov/KB/agentic/`,
`/home/liornisimov/KB/software-engineering/24-agent-procedures`, `.../22-ai-coding-agent-risks`
Audit for: CLI/JSON seam completeness & stability, fail-closed defaults, agent-forgeable inputs
(`--actor` class), idempotency & crash-safety of agent operations, autonomy boundaries (the
never-go-live invariant), auditability of agent decisions (folded in), determinism of agent-facing
outputs.
North-star focus: can an autonomous operator drive ML strategies safely; do agent guardrails hold as
operation scales.

## ml-dl-integration — ML/DL integration readiness
KB anchor: `/home/liornisimov/KB/machine-learning-engineering/`,
`/home/liornisimov/KB/quant-trading/12-models`
Audit for: the concrete seams an ML/DL strategy would need — where a model would plug into the
`signal`/`construction` contract; feature availability & PIT-correctness for model inputs; how a
trained artifact would be versioned, gated, and lifecycle-managed; batch vs online inference;
GPU/CPU/latency budget in the tick loop; fallback when a model errors.
North-star focus: THIS lane's entire job is ml_readiness — enumerate the missing integration seams
concretely, each with file:line where the seam would attach.

## risk-safe-scaling — Risk & safe-scaling
KB anchor: `/home/liornisimov/KB/quant-trading/16-metrics-and-risk`,
`/home/liornisimov/KB/software-engineering/13-performance`, `.../14-concurrency-async-distributed`
Audit for: risk limits (position/sector/net/gross caps, drawdown, turnover), breach→flatten→halt
paths, allocation/reservation-pool correctness under concurrency, performance/throughput ceilings,
resource exhaustion, blast-radius containment (one strategy can't sink the book).
North-star focus: THIS lane's entire job is safe_scale — what breaks as strategies/capital/model
complexity scale; what risk wall is missing or fail-open.

## model-risk-management — Model risk management (drift / monitoring)
KB anchor: `/home/liornisimov/KB/quant-trading/12-models`, `.../10-splitting-and-validation`,
`/home/liornisimov/KB/machine-learning-engineering/` (monitoring/drift notes)
Audit for: model validation discipline (SR 11-7 style), concept/feature drift detection, live
model-performance monitoring, champion–challenger, automatic model rollback/kill on decay, model
governance & documentation, silent-decay detection.
North-star focus: an ML/DL strategy that silently decays in production is the core safe-scale +
ml_readiness failure — enumerate what monitoring/rollback machinery is missing.

## security — Security & trust boundaries
KB anchor: `/home/liornisimov/KB/software-engineering/12-security`, `.../22-ai-coding-agent-risks`
Audit for: broker API-key/secret handling, the go-live signature / trust-anchor integrity (re-verify
at trade time vs DB-as-record), injection (SQL/command/prompt), the agent↔human trust boundary &
forgeable inputs, supply-chain (deps, lockfile), authz fail-closed, audit-trail integrity.
North-star focus: real capital + autonomous operation raises the security bar as the system scales;
find the trust boundaries that don't hold.

## observability — Observability & operability
KB anchor: `/home/liornisimov/KB/software-engineering/15-observability`,
`.../11-error-handling-resilience`
Audit for: kill-switches / halt-all / flatten reachability, alerting on breach/error/decay,
structured logging & metrics, audit trail, incident-response runbooks, health checks, the
"can you SEE it and STOP it" axis, dead-letter/quarantine visibility.
North-star focus: as strategies/models scale, can an operator observe and halt unsafe behavior fast
enough.
````

- [ ] **Step 2: Verify all 11 lane slugs are present as headers**

Run:
```bash
for s in swe mle ds qf clean-code agentic ml-dl-integration risk-safe-scaling model-risk-management security observability; do
  grep -q "^## ${s} " .claude/skills/readiness-review/lanes.md || { echo "MISSING lane: $s"; exit 1; }
done; echo "all 11 lanes present"
```
Expected: `all 11 lanes present`

- [ ] **Step 3: Verify every lane cites a KB anchor and a north-star focus**

Run:
```bash
grep -c '^KB anchor:' .claude/skills/readiness-review/lanes.md
grep -c '^North-star focus:' .claude/skills/readiness-review/lanes.md
```
Expected: `11` then `11`

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/readiness-review/lanes.md
git commit -m "feat(skill): readiness-review lane rubrics (11 lanes + KB anchors)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VnGemnoHubNa8JRQpZucPC"
```

---

### Task 3: Workflow script — meta, config, schemas, Context phase

**Files:**
- Create: `.claude/skills/readiness-review/readiness-review.workflow.mjs`

**Interfaces:**
- Produces (module-scope constants later tasks append after): `DRY_RUN`, `LANE_SUBSET`, `RID`,
  `LANES_DOC`, `ALL_LANES`, `LANES`, `FINDING_SCHEMA`, `VERDICT_SCHEMA`, `DEDUP_SCHEMA`,
  `FILED_SCHEMA`, and `context` (string brief from the Context phase).

- [ ] **Step 1: Write the workflow header, config, schemas, and Context phase**

```javascript
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
```

- [ ] **Step 2: Syntax-check the script**

Run: `node --check .claude/skills/readiness-review/readiness-review.workflow.mjs`
Expected: exit 0, no output (top-level `await`/`export` are valid ESM; injected globals like
`agent`/`phase`/`args` are free identifiers and do not fail a syntax-only check).

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/readiness-review/readiness-review.workflow.mjs
git commit -m "feat(skill): readiness-review workflow — config, schemas, Context phase

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VnGemnoHubNa8JRQpZucPC"
```

---

### Task 4: Workflow script — Find + Verify pipeline, and Dedup barrier

**Files:**
- Modify: `.claude/skills/readiness-review/readiness-review.workflow.mjs` (append after the Context phase)

**Interfaces:**
- Consumes: `context`, `LANES`, `LANES_DOC`, `FINDING_SCHEMA`, `VERDICT_SCHEMA`, `DEDUP_SCHEMA`.
- Produces: `const survivors` (flat array of kept finding objects) and `let deduped` (array of
  finding objects to file).

- [ ] **Step 1: Append the Find→Verify pipeline and the Dedup barrier**

```javascript
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
```

- [ ] **Step 2: Syntax-check**

Run: `node --check .claude/skills/readiness-review/readiness-review.workflow.mjs`
Expected: exit 0, no output.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/readiness-review/readiness-review.workflow.mjs
git commit -m "feat(skill): readiness-review workflow — find/verify pipeline + dedup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VnGemnoHubNa8JRQpZucPC"
```

---

### Task 5: Workflow script — File phase, Synthesize phase, return

**Files:**
- Modify: `.claude/skills/readiness-review/readiness-review.workflow.mjs` (append after Dedup)

**Interfaces:**
- Consumes: `deduped`, `survivors`, `DRY_RUN`, `RID`, `FILED_SCHEMA`.
- Produces: `let filed` (array of `{number,title,lane,severity}`) and the workflow's return value
  `{ survivors, filed, dryRun }`.

- [ ] **Step 1: Append the File phase, Synthesize phase, and return**

```javascript
// ---- Phase 4: file GitHub issues (skipped entirely under DRY_RUN) ----
phase('File')
let filed = []
if (DRY_RUN) {
  log(`DRY_RUN: would file ${deduped.length} issue(s); nothing created, nothing committed`)
  deduped.forEach((f) => log(`WOULD FILE [${f.lane}/${f.severity}] ${f.title}`))
} else {
  const fileAgent = await agent(
    `You are the issue-filing agent (cwd = repo root). Create GitHub issues via gh.

FIRST, idempotently ensure these labels exist (use --force so an existing label is not an error):
  gh label create readiness-review --color 5319e7 --force
  For each lane slug, gh label create "lane:<slug>" --color 1d76db --force
    slugs: swe mle ds qf clean-code agentic ml-dl-integration risk-safe-scaling model-risk-management security observability
  gh label create severity:critical --color b60205 --force
  gh label create severity:high --color d93f0b --force
  gh label create severity:medium --color fbca04 --force
  gh label create severity:low --color 0e8a16 --force

THEN create ONE issue per finding below:
  gh issue create --title "[<lane>] <title>" \
    --label readiness-review --label "lane:<lane>" --label "severity:<severity>" \
    --body "<body>"
The body MUST include, as markdown sections: Severity, Lane, North-star link, Evidence (the files
list), Description, Why it matters, Recommendation, KB citation, Web citation.
Capture the issue number gh prints (trailing integer of the returned URL).

Findings JSON:
${JSON.stringify(deduped)}

Return the filed schema: { filed: [{number, title, lane, severity}] }.`,
    { label: 'file-issues', phase: 'File', schema: FILED_SCHEMA },
  )
  filed = (fileAgent && fileAgent.filed) || []
  log(`Filed ${filed.length} issue(s)`)
}

// ---- Phase 5: synthesis verdict, committed (never pushed); skipped under DRY_RUN ----
if (!DRY_RUN) {
  phase('Synthesize')
  await agent(
    `You are the synthesis agent (cwd = repo root). Write a north-star readiness verdict and COMMIT
it (do NOT push).

Write to: docs/superpowers/specs/${RID}-readiness-review-verdict.md
Content:
- Title + the run date (${RID}).
- Overall ML/DL-integration-readiness verdict AND safe-to-scale verdict (2-3 evidence-based paragraphs).
- Cross-lane themes and the BINDING LEVERS (the few changes that unblock the most).
- A table of filed issues: number | lane | severity | one-line title. Data: ${JSON.stringify(filed)}
- Per-lane one-paragraph summary (11 lanes).

Then commit ONLY that file (never git add -A), appending the standard trailers used by recent
commits (Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com> and the Claude-Session line):
  git add docs/superpowers/specs/${RID}-readiness-review-verdict.md
  git commit -m "docs(readiness): ${RID} 11-lane readiness verdict"
Do NOT git push. Return the path you wrote.`,
    { label: 'synthesize', phase: 'Synthesize', model: 'sonnet' },
  )
}

return { survivors: survivors.length, filed: filed.length, dryRun: DRY_RUN }
```

- [ ] **Step 2: Syntax-check**

Run: `node --check .claude/skills/readiness-review/readiness-review.workflow.mjs`
Expected: exit 0, no output.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/readiness-review/readiness-review.workflow.mjs
git commit -m "feat(skill): readiness-review workflow — file issues + synthesis verdict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VnGemnoHubNa8JRQpZucPC"
```

---

### Task 6: Integration dry-run + plan verification

**Files:**
- None created; this task runs the pipeline end-to-end in preview mode against ONE cheap lane.

**Interfaces:**
- Consumes: the complete workflow script + `SKILL.md` + `lanes.md`.
- Produces: evidence the pipeline wires up (Context→Find→Verify→Dedup→"would file"), with no issues
  filed and nothing committed.

- [ ] **Step 1: Confirm `gh` is authenticated (finders/dedup call it read-only)**

Run: `gh auth status`
Expected: a logged-in account is shown. If not, stop and tell the user to run `gh auth login`.

- [ ] **Step 2: Run a single-lane dry-run via the Workflow tool**

Invoke (this spawns real subagents but files nothing and commits nothing):
```
Workflow({
  scriptPath: '.claude/skills/readiness-review/readiness-review.workflow.mjs',
  args: { runId: 'dry', dryRun: true, lanes: ['clean-code'] }
})
```
Expected: the run completes and returns `{ survivors: <int>, filed: 0, dryRun: true }`. The progress
log shows `Context` → `Find` (one `find:clean-code`) → `Verify` (per finding) → `Dedup` →
`WOULD FILE ...` lines. No `Synthesize` phase runs.

- [ ] **Step 3: Confirm no side effects**

Run:
```bash
git status --porcelain docs/superpowers/specs/ | grep -i verdict || echo "no verdict file created (correct)"
gh issue list --state open --label readiness-review --limit 5
```
Expected: `no verdict file created (correct)`; and no `readiness-review`-labeled issues exist yet
(the dry-run must not have created any).

- [ ] **Step 4: Final commit of the plan doc (if not already committed by brainstorming flow)**

```bash
git add docs/superpowers/plans/2026-07-01-readiness-review-skill.md
git commit -m "docs(plan): readiness-review skill implementation plan

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VnGemnoHubNa8JRQpZucPC" || echo "plan already committed"
```

---

## Notes for the executor

- **Do not run a full (non-dry) pass as part of implementation.** A real run files GitHub issues and
  commits a verdict — that is the *user-triggered* deliverable, not a test. Task 6 verifies wiring in
  dry-run only.
- If `node --check` errors on `export`/top-level `await`, confirm the file extension is `.mjs`
  (ESM) — it must be.
- Finder token cost scales with lanes; the dry-run restricts to one lane on purpose.
- KB paths are absolute (`/home/liornisimov/KB/...`); if a KB anchor dir is missing, the finder
  should still proceed with web + repo evidence and note the missing anchor rather than fail.
