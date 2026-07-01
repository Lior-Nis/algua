# Design — `readiness-review` skill (11-lane readiness audit)

**Date:** 2026-07-01
**Status:** Approved (brainstorming) — ready for implementation plan
**Author:** agent (with operator)

## 1. Purpose

A committed, re-runnable **project skill** that audits the whole `algua` repo across
**11 evaluation lanes** by fanning out one subagent per lane, adversarially verifying every
finding, deduping survivors against each other and against already-open GitHub issues, then
**auto-filing** the survivors as labeled GitHub issues and committing a north-star synthesis
verdict.

The audit is oriented by two standing north-star themes that **every lane must address from its
own angle**:

1. **ML/DL-model-as-strategy integration readiness** — is the platform ready to host ML/DL models
   as first-class strategies (training→serving→gating→lifecycle) without breaking its invariants?
2. **Risk management for safe scale** — is the system built so that scaling the number of
   strategies / capital / model complexity cannot fail unsafely?

This formalizes and hardens the earlier ad-hoc 6-lens review (which produced issues #324–342) into
a repeatable artifact.

## 2. Scope & non-goals

- **In scope:** whole-repo standing audit (production code + `docs/.../*-architecture-design.md`
  spec + `AGENTS.md` invariants). Re-runnable; dedups against open issues on every run.
- **Out of scope:** current-branch-diff-only mode (rejected — a "built to scale safely" review must
  see standing/architectural gaps); fixing the findings (the skill files issues, it does not patch);
  putting anything live.
- **Non-goal:** replacing `multi-model-review` (that is adversarial review of a *specific diff*;
  this is a *whole-system readiness* audit).

## 3. The 11 lanes

Each lane is one finder subagent. Each is pinned to a **primary KB anchor** it MUST read and a
**web best-practice search** it MUST run before auditing.

| # | Lane | KB anchor (`/home/liornisimov/KB/...`) |
|---|------|-----------------------------------------|
| 1 | SWE / architecture | `software-engineering/{04-software-design,07-backend-engineering,11-error-handling-resilience,14-concurrency-async-distributed,06-database-design}` |
| 2 | MLE (infra / serving) | `machine-learning-engineering/` |
| 3 | Data Science (stats / experimentation) | `data-science/` |
| 4 | Quant Finance | `quant-trading/{13-alpha-research,15-backtesting,16-metrics-and-risk,09-leakage-and-bias,10-splitting-and-validation,17-costs-slippage-execution}`, `finance/` |
| 5 | Clean code / maintainability | `software-engineering/{17-code-quality-maintainability,04-software-design}` |
| 6 | Agentic operation readiness | `agentic/`, `software-engineering/{24-agent-procedures,22-ai-coding-agent-risks}` |
| 7 | ML/DL integration readiness | `machine-learning-engineering/`, `quant-trading/12-models` |
| 8 | Risk & safe-scaling | `quant-trading/16-metrics-and-risk`, `software-engineering/{13-performance,14-concurrency-async-distributed}` |
| 9 | Model Risk Management (drift / monitoring) | `quant-trading/{12-models,10-splitting-and-validation}`, MLE monitoring notes |
| 10 | Security & trust boundaries | `software-engineering/{12-security,22-ai-coding-agent-risks}` |
| 11 | Observability & operability | `software-engineering/{15-observability,11-error-handling-resilience}` |

**Folded-in concerns** (NOT separate lanes, to avoid overlap/noise — assigned to the lane in
parentheses): data provenance & reproducibility (DS + MLE); execution microstructure, cost realism,
backtest↔live parity (QF); test-suite adequacy (SWE + clean-code); compliance / auditability
(agentic + security).

### Lane-specific emphasis for the two north-star themes
- **MRM (9)** carries the sharpest ML-safety load: model validation, concept/feature drift
  detection, champion–challenger, model rollback, live model-performance monitoring (SR 11-7 style
  discipline applied to trading models).
- **Security (10)** carries the live-capital trust load: signing / trust-anchor integrity, broker
  key handling, injection, the agent↔human trust boundary, supply chain, the forgeable-`--actor`
  class (#329).
- **Observability (11)** carries the "see it and stop it" load: kill-switches, alerting, audit
  trail, halt/flatten paths, incident response.

## 4. Orchestration — Workflow pipeline

The skill authors and runs a **Workflow script** (deterministic; structured schemas; resumable).

```
Phase 0  Context-pack   1 agent  → shared brief: architecture-spec summary, AGENTS.md invariants,
                                   module inventory, open-issue {number,title,labels} for dedup.
Phase 1  Find           11 finders (parallel) → each: read KB anchor + WebSearch best practices +
                                   audit repo through its lens → structured FINDINGS.
Phase 2  Verify         1 skeptic per finding (pipelined) → tries to REFUTE → keep survivors only.
Phase 3  Dedup          barrier → semantic dedup survivors vs each other + vs open issues; log drops.
Phase 4  File           ensure labels exist; gh issue create per survivor (no cap); capture numbers.
Phase 5  Synthesize     1 agent → commit north-star readiness verdict doc linking filed issues.
```

- **Pipeline, not barrier, for Find→Verify:** each finding verifies as soon as its lane completes;
  lanes do not wait on each other. Dedup (Phase 3) is a genuine barrier — it needs all survivors.
- **Structured output** via schemas at each stage (finding schema, verdict schema) so the
  orchestrator gets validated objects, not prose to parse.

### Finding schema (fields)
`lane, title, severity{critical|high|medium|low}, files[] (path:line evidence),
description, why_it_matters, north_star_link{ml_readiness|safe_scale|both|none},
recommendation, kb_citation (note path), web_citation (url + one-line)`.

### Verdict schema (adversarial verify)
`finding_id, is_real{bool}, already_mitigated{bool}, duplicate_of_open_issue{number|null},
evidence_confirmed{bool}, refutation_reason{string}, keep{bool}`.

## 5. Grounding contract (every finder obeys)

1. **Cite `file:line`** for every finding — no evidence, no finding (fail toward "not a finding").
2. Cite **≥1 KB note** AND **≥1 web best-practice source**. A finder that skips its KB anchor or
   web search is non-conformant.
3. Findings must be **concrete and actionable** (a recommendation an implementer can act on), not
   vague ("improve testing"). Vague findings are dropped in verify.
4. Every finding tags `north_star_link` so the synthesis can report ML-readiness / safe-scale
   coverage explicitly.

## 6. Adversarial verification

One independent skeptic subagent per candidate finding, prompted to **refute by default**:
- Re-open the cited `file:line` and confirm the defect actually exists there.
- Check the concern is not **already mitigated** elsewhere in code (a common false-positive class:
  the reviewer didn't see the guard).
- Check it is not **already an open issue** (mark `duplicate_of_open_issue`).
- Default `keep=false` when uncertain. Only `keep=true` survivors proceed.

## 7. Dedup & filing

- **Dedup (barrier):** fetch `gh issue list --state open --json number,title,body,labels`; a
  semantic-dedup agent collapses near-duplicate survivors and drops any matching an open issue.
  **Every drop is logged** (title + reason) — never silent truncation.
- **Filing (no volume cap — operator choice):** idempotently ensure labels exist
  (`lane:<slug>`, `severity:<level>`, `readiness-review`), then `gh issue create` per survivor with
  title, structured body (evidence, why-it-matters, recommendation, KB + web citations,
  north-star tag), and labels. Capture the created issue numbers.
- Filing targets the repo's `origin` via `gh` (issues are remote by nature).

## 8. Synthesis output

A final agent writes and **commits** (does NOT push — operator choice) a north-star verdict doc to
`docs/superpowers/specs/YYYY-MM-DD-readiness-review-verdict.md`:
- Overall **ML/DL-integration-readiness** verdict and **safe-to-scale** verdict.
- Cross-lane themes and the **binding levers** (the few changes that unblock the most).
- A table of all filed issues (number, lane, severity, one-line), plus the logged drops.
- Per-lane one-paragraph summary.

The synthesis doc is committed on the current branch; **no auto-push**. GitHub issues are created
regardless (via `gh`).

## 9. Deliverables

1. `.claude/skills/readiness-review/SKILL.md` — the invocable skill: frontmatter (name,
   description with trigger phrases), the grounding contract, the lane roster + KB anchors, and the
   instruction to author/run the Workflow pipeline described here.
2. Supporting reference material under `.claude/skills/readiness-review/` as needed
   (e.g. `lanes.md` with the per-lane rubric prompts, `finding-schema.md`).
3. This spec, committed.

## 10. Open risks / decisions already made

- **No volume cap** on filed issues (operator choice) — flooding risk accepted; drops are logged so
  nothing is hidden, but the first run may create a large batch against the ~30 open issues. The
  dedup barrier is the main guard against duplicates.
- **Whole-repo every run** — higher token cost than diff mode; accepted for a readiness gate.
- **Auto-file, hands-off** — the adversarial-verify pass is the quality gate that makes hands-off
  filing safe; it must default to `keep=false` under uncertainty.
