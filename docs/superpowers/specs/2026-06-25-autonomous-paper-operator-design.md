# Autonomous Paper Operator — Design

**Status:** Draft — revised after Codex adversarial review (findings folded in; pending human review → implementation plan)
**Date:** 2026-06-25
**Author:** agent (Claude) + human operator
**Review:** Codex verified claims against the codebase (2026-06-25). It confirmed the paper-primitive
gap (§5 G1) and the live-wall enforcement (§3, §10), and corrected five claims now fixed in this
revision: breadth source (§6.3), merge ordering (§6.1–6.2), family-creation feasibility (§3.2, §5 G2),
family-audit teeth + transition legality (§6.5), and the `block-push-to-main` scope (§6.4).

## 1. Problem & motivation

Algua has the *capabilities* to run an autonomous research-and-paper-trading funnel, but
nothing **assembles them into an always-on operator**. Per the architecture roadmap (§9 of
`2026-05-29-algua-platform-architecture-design.md`):

- **SP1 (CLI), SP2 (data), SP3 (research core)** — done. The verbs exist.
- **SP4 (agent operating layer)** — delivered only as the one-shot `run-research-loop.sh`, scoped
  to `candidate`, EXPLORATION-mode (per-run DB), producing a branch a human reviews and merges.
- **SP5 (execution + paper)** — built the *primitives* (`paper trade-tick`, kill-switch,
  breach/flatten/resume, quarantine) but **no loop drives them** continuously.
- **SP6 (live hardening + cloud lift)** — the entire always-on layer (scheduling, restart, secrets,
  Docker/VM) lives here and is barely started.

So the gap is not a missing feature; it is that **no component's job is "keep the whole funnel
running every session without a human."** The build energy went deep into the statistical gates
(SP4) and execution primitives (SP5); the unglamorous assembly into a 24/7 operator (SP6) kept being
deferred. This spec closes that gap for **paper** (never live).

## 2. Goals & non-goals

**Goals**
- A 24/7 autonomous operator that runs the funnel `idea → backtested → candidate → paper →
  forward_tested` with **no human in the loop except the live wall**.
- Codex performs all judgment work (ideate, author, interpret, promote); deterministic code performs
  all broker/order work.
- Runs on the operator's current machine now; hardenable to a cloud host later as config, not a
  rewrite.

**Non-goals (YAGNI / out of scope)**
- The `forward_tested → live` transition (stays human + TOTP-signed, unchanged).
- Per-strategy timers; multi-calendar / multi-frequency scheduling (one schedule-class today).
- Cloud VM / containerized always-on host.
- A Codex "exception supervisor" that acts during breaches (model C — a future v2).

## 3. Locked decisions

1. **Autonomy boundary:** only the live transition is human. Codex autonomously creates families,
   merges its own discoveries into the authoritative registry, and allocates candidates to paper.
2. **Host:** this machine, via local systemd timers. Harden later.
3. **Operating model (A):** a deterministic spine (systemd) owns the clock; Codex is *invoked* for
   judgment; the order-submission path contains no LLM.

### 3.1 Consequence: the gates are now load-bearing

With no human review, the funnel-wide multiple-testing defenses (LORD++ FDR alpha-wealth ledger, DSR
deflation, holdout single-burn, family governance) are the **only** backstop between noise and the
paper book. This is the regime that justifies that machinery. The design must therefore preserve
their integrity exactly — never run the metered promote against anything but the authoritative DB,
never double-burn a holdout, never undercount search breadth.

### 3.2 Accepted risk: automated family creation (full automation chosen)

Making NOVEL family creation autonomous re-opens the breadth-evasion vector that #222 made human-only:
minting fresh families dodges funnel-wide breadth deflation and can inflate significance. **This is a
deliberate teardown of an existing control**, not wiring. Codex confirmed: `--new-family` is human-only
by code (`promotion.py:122,233,240`, `research_cmd.py:80`), and the `research family-audit` detector is
**read-only — it benches nothing today** (`research_cmd.py:304–357`).

The operator therefore must BUILD three new pieces (§5 G2/G3, §6.5): (a) permit agent NOVEL family
creation in `promotion.py`; (b) the family-audit **guard with teeth**; (c) a *legal* remediation that
does not violate the lifecycle (`candidate→dormant` is illegal — `lifecycle.py`). **Compensating
control:** the guard blocks the auto-allocator for a flagged candidate (a quarantine flag, no illegal
transition) and benches a flagged *paper* strategy `paper→dormant` (legal, reversible); a human then
performs the #222 family consolidation.

## 4. Architecture & control flow

Two systemd timer+service pairs on this box, sharing the authoritative `data/algua.db`,
`snapshots/`, and local git `main`. No long-lived process; each firing is a bounded single-shot run,
so a reboot just means "the next timer fires."

```
                    authoritative: data/algua.db  +  snapshots/  +  git main (local)
                                          ▲
        ┌─────────────────────────────────┼─────────────────────────────────┐
 ┌──────┴───────┐ (research cadence) ┌─────┴──────┐ (each session, post-close)
 │ RESEARCH JOB │                    │ PAPER       │
 │  (Codex)     │                    │ RUNTIME     │
 └──────────────┘                    │ (pure CLI)  │
  ideate→author→backtest→SWEEP→      └─────────────┘
  quality-gate→merge→                 ingest fresh bars → paper run-all
  promote (auto-family)→allocate      → paper promote (forward eval)
  → family-audit guard                → structured log + alert
```

- **Paper runtime** — fires once per XNYS session, ~30 min after close, calendar-gated. Deterministic.
- **Research job** — slower off-hours cadence, bounded by OS `timeout` + a hypotheses cap.
- **Analysis job** (optional) — periodic Codex `report-experiments` run; read-only.

Every job is a **single-shot command** (`--once` semantics); systemd merely repeats it. This is what
makes the system testable by invoking one command against fixtures.

## 5. Components to build

`[NEW]` = code to write; `[WIRE]` = exists, orchestrate.

### Group 1 — Paper trading primitives (close the live/paper gap)
- **[NEW] `paper allocate <name> --capital $X`** — per-strategy paper capital base, enforcing
  Σ allocations ≤ paper-account equity. Mirror of `live allocate` + the `allocations` module.
  *Without it, N concurrent strategies each size against the full account and over-leverage it.*
- **[NEW] `paper run-all`** — batch driver. Loads active paper strategies, groups by **schedule-class**
  (calendar × rebalance_freq), asks each `is_due?`, ticks the due set via the existing per-strategy
  `trade-tick` logic. A breach trips+flattens **only that strategy**; the batch continues. Mirror of
  `live run-all`.
- **[NEW] `is_due(strategy, session)`** — predicate on the execution contract. Today: `True` for
  daily/XNYS. The seam that makes multi-cadence additive later.

### Group 2 — Research cycle (Codex, made authoritative)
- **[NEW] research-cycle driver** — evolves `run-research-loop.sh`: runs Codex in an isolated
  worktree (scratch), then performs the **automated merge-back** that replaces the human (§6).
- **[NEW] agent NOVEL-family creation** — modify `promotion.py` (currently human-only at
  `promotion.py:122,233,240` + `research_cmd.py:80`) to permit the agent to mint a NOVEL family.
  This deliberately removes a #222 control; the §6.5 guard is its compensating enforcement.

### Group 3 — Autonomous safety guards
- **[NEW] `family-audit` guard (teeth)** — `research family-audit` is read-only today
  (`research_cmd.py:304–357`); build a guard that, at end of cycle, acts on a flagged family:
  **block the auto-allocator** for a flagged *candidate* (a quarantine flag — no illegal transition)
  and **bench a flagged *paper* strategy `paper→dormant`** (legal, reversible). Alert; a human then
  does the #222 consolidation. `candidate→dormant` is NOT a legal transition and must not be used.
- **[WIRE] kill-switch / breach-flatten** — per-strategy, exists; the runtime surfaces trips to the
  alert channel.

### Group 4 — Always-on driver (this box)
- **[NEW] 2 systemd timer+service pairs** — `algua-paper.timer` (~16:30 ET, calendar-gated) and
  `algua-research.timer` (off-hours). Stateless; a reboot resumes on next fire.
- **[NEW] session-idempotency guard** — the paper job checks "already ticked this session?" (from
  tick-snapshot rows) so a double-fire / restart cannot double-trade.
- **[NEW] structured run logs + alert hook** — append-only JSON logs; an alert on breach, halt,
  gate-fail, or job crash (desktop notification / log line; pluggable later).

### Group 5 — Secrets & config
- **[WIRE] `.env` + systemd `EnvironmentFile`** — Alpaca **paper** creds only. Note (Codex-confirmed):
  the live wall is **cryptographically enforced** — `live` requires a human actor, an active
  allocation, a fresh forward certificate, a recomputed identity, and a verified signature, re-checked
  at trade time (`live_gate.py:155–188`, `live_cmd.py:248–273`). Absent live creds only block broker
  construction; the wall itself does not depend on that. The operator cannot cross it regardless.

### Group 6 — Analysis job (optional)
- **[NEW] `algua-report.timer`** — periodic Codex `report-experiments` over active paper strategies
  → report into the kb vault. Read-only; no registry effect.

## 6. The research → merge-back → paper pipeline

Respects two hard codebase invariants: the holdout is **single-use** (one `research promote` attempt
per strategy per OOS interval — #192/#193/#161) and the FDR alpha-wealth ledger is **funnel-wide**
(every metered attempt draws a shared budget).

### 6.1 Pipeline (one cycle, serialized under a lock)

**Ordering note (Codex finding #2):** the metered promote mutates the shared DB to `candidate`. If
that happened *before* the code reached main, the registry would reference a candidate not importable
on main — a real (and crash-durable) inconsistency. So **merge precedes promote**, and a FAIL reverts
the merge.

| # | Step | Store | Metered? |
|---|------|-------|----------|
| 1 | Codex forms hypotheses, authors code, **runs `backtest sweep`** (the MEASURED breadth the gate reads — §6.3), in an isolated **worktree/branch** | worktree git + `search_trials` | breadth measured |
| 2 | **Quality gate** on the branch: `pytest && ruff && mypy && lint-imports` | worktree | — |
| 3 | **Merge branch → local main** (additions-only) so the strategy is importable on main | git main | — |
| 4 | **`research promote` once**, code loaded from **main**, against the **authoritative DB** | authoritative DB | **the metered event** — breadth deflation, FDR draw, holdout burn, candidate mint |
| 5a | **PASS** → family-audit guard → `paper allocate` + `candidate→paper` | authoritative DB | — |
| 5b | **FAIL** → **revert the merge commit** (main clean again), log | git main | holdout for that interval now spent (correct) |

### 6.2 Invariants the ordering guarantees
1. **The registry never references code that isn't on main.** Merge (step 3) precedes the candidate-
   minting promote (step 4); a FAIL reverts the merge (step 5b). There is no window — even across a
   crash — where the DB holds a `candidate` whose code is absent from main. `code_hash` is path-
   independent (computed from dotted module name + source text, not file path — `approvals.py:62–106`,
   Codex-confirmed), so loading from main vs worktree yields the identical hash.
2. **One metered attempt per strategy.** Steps 1–3 are free scratch; the single promote in step 4 is
   the only holdout burn and FDR draw (single-use reservation refuses overlaps — `store.py:500–532`).
   No double-burn.

### 6.3 Breadth honesty (load-bearing, given no human — Codex finding #1)
Because Codex self-approves, the gates must see the *true* search. **Correction from the first draft:**
gate breadth is driven by `search_trials` from **`backtest sweep`** (`promotion.py:337`, `gates.py:477`,
`search_breadth.py`), **not** by the idea pool — `windowed_idea_counts` is explicitly *not yet consumed*
(#126, deferred). So honesty requires the research cycle to **express each hypothesis's parameter search
as a `backtest sweep`** (step 1), so `search_trials` reflects what was actually tried; merely running a
single backtest, or registering ideas, would undercount breadth and inflate significance. Across
*separate* hypotheses, the funnel-wide LORD++ alpha-wealth ledger meters every promote (each draws
budget), and `effective_funnel_breadth` folds in the windowed funnel total + family-lifetime breadth.
(Optionally wiring #126's idea counts into the gate would tighten this further — out of scope here.)

### 6.4 Concurrency, idempotency, recovery
- A **file lock** serializes research cycles and excludes the paper runtime from mutating the
  registry mid-promote (the DB already uses `BEGIN IMMEDIATE`; the lock guards the worktree/main).
- Operates on **local main only** — the operator never runs `git push`. **Caveat (Codex):**
  `block-push-to-main` is a *Claude-process guardrail, not a git hook* (`.git/hooks` has only samples),
  so a systemd job calling `git push` would bypass it entirely. The mitigation is simply that the
  operator code contains no push; **if push is ever added, a real git `pre-push` hook must be installed
  first.**
- A crash mid-cycle leaves either a stranded worktree (cleaned on next start), a merged-but-unpromoted
  strategy on main at `backtested` (harmless — not allocated; re-promotable since the holdout wasn't
  burned), or a burned holdout with a minted candidate already on main (consistent). No state where a
  candidate lacks code on main (§6.2).

### 6.5 Family-audit guard (teeth, post-hoc — Codex findings #4/#5)
`research family-audit` is **read-only today** (writes nothing — `research_cmd.py:304–357`); the guard
is NEW code that runs it at the **end of each cycle** and acts on a flagged family (the cross-family
breadth-evasion pattern the per-promote classifier cannot see):
- a flagged **candidate** → set a **quarantine flag** the auto-allocator respects (do not allocate).
  We do **not** transition it — `candidate→dormant` is illegal (`lifecycle.py`).
- a flagged **paper** strategy → **bench `paper→dormant`** (legal, reversible — pulls it from the book).
- **alert** in both cases; a human then performs the #222 family consolidation (member reassignment).

## 7. Operating parameters (defaults — tunable config, not code)

### Schedules
| Param | Default |
|---|---|
| Paper runtime fires | ~16:30 ET (≈30 min after XNYS close), calendar-gated |
| Research cycle fires | daily, 02:00 local |
| Hypotheses per cycle | 3 |
| Research run timeout | 30 min (OS `timeout`) |
| Analysis/report job | weekly, Sun |

### Capital & allocation (paper)
| Param | Default |
|---|---|
| Paper account equity | whatever the Alpaca paper account holds (~$100k default) |
| Allocation policy | fixed $10k per strategy |
| Max concurrent paper strategies | 8 (Σ ≤ $80k → headroom) |
| Overflow behavior | stay `candidate`, queue until a slot frees |

### Risk
| Param | Default |
|---|---|
| Per-strategy kill-switch drawdown | 20% (`paper trade-tick --max-drawdown`) |
| Hard walls (gross/concentration/short) | existing #135 defaults (not weakened) |
| Forward-gate thresholds | protected defaults (≥63 obs, ≥90% coverage, Sharpe floor…); agent cannot relax |

### Data & cost
| Param | Default |
|---|---|
| Daily ingest source | yfinance (daily bars) |
| Forward-promote check | each session (cheap eligibility check) |
| Codex cost ceiling | 1 cycle/day × 3 hypotheses × 30 min |

## 8. Testing & bring-up

Every job is a single-shot command, so most tests are one invocation against fixtures.

**Unit (no real broker/LLM)**
- `paper allocate` / `run-all` / `is_due` — mirror the `live` tests against a SimBroker/fake broker:
  Σ ≤ equity; dormant rejection; `run-all` ticks the due set and skips not-due; a single breach
  trips+flattens only that strategy while the batch continues; session-idempotency.
- Family-audit guard — fixture DAG: flagged ⟹ bench to dormant + alert; clean ⟹ no-op.
- Calendar gate + idempotency guard — holiday/weekend ⟹ no-op; double-fire ⟹ single trade.

**Orchestration (merge-back driver, the risky unit)** — test around a **fake `codex`** stub (as
#210 tests fake `docker`): PASS ⟹ code on main + candidate minted + allocated, `code_hash` identical
across merge; FAIL ⟹ **merge reverted**, main clean, holdout spent; breadth honesty ⟹ each hypothesis
is promoted with a `search_trials` count matching its sweep (not 1); lock ⟹ concurrent cycles
serialize; crash ⟹ stranded worktree cleaned next start, no candidate-without-code on main.

**Integration / E2E** — full cycle on synthetic data + SimBroker with an **injected session clock**:
ingest → fake-Codex cycle → candidate → allocate → tick × N → forward-promote, accelerated to ≥63
sessions so `paper promote` fires without a 3-month wait.

**Safety negative tests** — no live creds ⟹ live path refuses; holdout single-burn holds under the
driver; promote runs against the authoritative DB (footgun guard satisfied, not tripped).

**Discipline** — TDD per unit; full gate (`pytest && ruff && mypy && lint-imports`) green before any
commit; add to the existing suite, never weaken a gate.

**Staged manual bring-up (before enabling timers)**
1. Paper runtime, manual, real Alpaca paper — one strategy, a few sessions; watch fills + ticks.
2. Research cycle, manual, real Codex — one full cycle; watch it reach candidate→paper.
3. Enable the timers — only after 1 & 2 look right.

## 9. Open questions / future

- **Multi-schedule-class** scheduling (weekly, crypto 24/7, international) — additive when a non-
  daily/non-XNYS strategy first appears.
- **Cloud lift** (SP6) — move the timers + secrets to an always-on host; design keeps this a config
  change.
- **Model C** — a Codex exception-supervisor invoked on breach/reconcile/quarantine anomalies.
- **Pushing agent-authored code** to origin / opening PRs — currently local-main only.
- **Dead-code housekeeping** — promote-FAIL reverts the merge, so main is not polluted; revisit if the
  registry accumulates `backtested` cruft from crashes between merge and promote.

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Automated family creation tears down #222's anti-gaming control (§3.2) | High | `family-audit` guard with teeth: quarantine flagged candidates, bench flagged paper→dormant, alert + human consolidation (§6.5) |
| Self-approval promotes overfit noise | High | Merge-then-promote against authoritative DB; breadth measured via `backtest sweep` + funnel-wide LORD++ ledger (§6.3); gates unweakened |
| Registry references code not on main | High→resolved | Merge precedes promote; FAIL reverts the merge; no candidate-without-code window (§6.2) |
| Concurrent paper strategies over-leverage the account | High | `paper allocate` (Σ ≤ equity) is a hard prerequisite (§5 G1) |
| Box reboots / sleeps (single point of failure) | Medium | Stateless single-shot jobs; resume on next fire; cloud lift later |
| Concurrent research cycles corrupt worktree/main | Medium | File lock serializes cycles (§6.4) |
| Double-trade on timer double-fire / restart | Medium | Session-idempotency guard (§5 Group 4) |
| A future `git push` bypasses `block-push-to-main` (Claude-only guardrail) | Medium | Operator never pushes; if added, install a real git `pre-push` hook first (§6.4) |
| Accidental live trade | Critical | Live wall is cryptographically enforced (signature/cert/allocation/identity), not merely creds-absent (§5 G5) |
