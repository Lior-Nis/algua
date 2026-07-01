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
