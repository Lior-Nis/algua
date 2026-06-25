# Autonomous Paper Operator — Design

**Status:** Draft (design approved in brainstorming; pending Codex review → implementation plan)
**Date:** 2026-06-25
**Author:** agent (Claude) + human operator

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

### 3.2 Accepted risk: automated family creation

Making first-family / NOVEL family creation autonomous re-opens the breadth-evasion vector that #222
made human-only: minting fresh families dodges funnel-wide breadth deflation and can inflate
significance. **Mitigation:** the `research family-audit` cross-family gaming detector runs as a
post-cycle **guard with teeth** (§6.5) — it benches a flagged strategy to `dormant` and alerts,
rather than merely advising.

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
  ideate→author→backtest→            └─────────────┘
  promote (auto-family)→              ingest fresh bars → paper run-all
  quality-gate→merge→allocate         → paper promote (forward eval)
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
- **[WIRE] family auto-creation** — Codex passes `--new-family` on a NOVEL classification (now
  permitted for the agent per §3.1).

### Group 3 — Autonomous safety guards
- **[NEW] `family-audit` guard** — runs `research family-audit` at end of cycle; a flagged family ⟹
  bench the strategy to `dormant` + alert (teeth, not advice).
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
- **[WIRE] `.env` + systemd `EnvironmentFile`** — Alpaca **paper** creds only. Live creds
  deliberately absent — the box literally cannot trade live.

### Group 6 — Analysis job (optional)
- **[NEW] `algua-report.timer`** — periodic Codex `report-experiments` over active paper strategies
  → report into the kb vault. Read-only; no registry effect.

## 6. The research → merge-back → paper pipeline

Respects two hard codebase invariants: the holdout is **single-use** (one `research promote` attempt
per strategy per OOS interval — #192/#193/#161) and the FDR alpha-wealth ledger is **funnel-wide**
(every metered attempt draws a shared budget).

### 6.1 Pipeline (one cycle, serialized under a lock)

| # | Step | Store | Metered? |
|---|------|-------|----------|
| 1 | Codex forms hypotheses, **registers each as an idea** (`research idea add`), authors code, backtests/sweeps — in an isolated **worktree/branch** | worktree git + idea pool | breadth counted |
| 2 | **Quality gate** on the branch: `pytest && ruff && mypy && lint-imports` | worktree | — |
| 3 | **`research promote` once**, code loaded from the worktree, against the **authoritative DB** | authoritative DB | **the metered event** — breadth deflation, FDR draw, holdout burn, candidate mint |
| 4a | **PASS** → merge branch → **local main** → `paper allocate` + `candidate→paper` | git main + authoritative DB | — |
| 4b | **FAIL** → **discard the branch** (nothing merged), log | — | holdout for that interval now spent (correct) |

### 6.2 Invariants the ordering guarantees
1. **Code on main ⟹ it is a candidate.** Merge happens *only* on a promote PASS. Failed hypotheses
   never reach main; passing ones are importable before they are allocated. `code_hash` at promote
   (worktree bytes) == on main after merge (additions-only files → trivial merge, same bytes).
2. **One metered attempt per strategy.** Steps 1–2 are free scratch; the single authoritative promote
   in step 3 is the only holdout burn and FDR draw. No explore-then-re-promote double-burn.

### 6.3 Breadth honesty (load-bearing, given no human)
Because Codex self-approves, the gates must see the *true* search. Step 1 registers **every**
hypothesis into the idea pool — including pre-screened-away ones — so funnel-wide breadth deflation
accounts for the whole search, not just promoted survivors. (This exercises #126's windowed-idea-count
wiring.) Cherry-picking the best K of N and promoting only those would otherwise undercount breadth
by N/K.

### 6.4 Concurrency, idempotency, recovery
- A **file lock** serializes research cycles and excludes the paper runtime from mutating the
  registry mid-promote (the DB already uses `BEGIN IMMEDIATE`; the lock guards the worktree/main).
- Operates on **local main only** — no push (the `block-push-to-main` hook governs origin; the
  runtime imports local). Push/PR of agent-authored code stays a separate human concern.
- A crash mid-cycle leaves either a stranded worktree (cleaned on next start) or a burned holdout
  with no candidate (correct — it was a real attempt). Merge is the last step and additions-only, so
  there is no partial-merge state.

### 6.5 Family-audit guard (teeth, post-hoc)
`research family-audit` runs at the **end of each cycle**: if it flags a paper/candidate strategy's
family as breadth-evasion gaming (the cross-family pattern the per-promote classifier cannot see), it
**benches that strategy to `dormant` and alerts**, reversing the autonomous allocation.

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
#210 tests fake `docker`): PASS ⟹ code on main + allocated, `code_hash` stable across merge; FAIL ⟹
branch discarded, main untouched, holdout spent; breadth honesty ⟹ every hypothesis lands in the
idea pool; lock ⟹ concurrent cycles serialize; crash ⟹ stranded worktree cleaned next start.

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
- **Dead-code housekeeping** — promote-FAIL discards the branch, so main is not polluted; revisit if
  the idea pool or registry accumulates cruft.

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Automated family creation evades breadth deflation (§3.2) | High | `family-audit` guard with teeth (§6.5) |
| Self-approval promotes overfit noise | High | Authoritative-only metered promote + breadth honesty (§6.3); gates unweakened |
| Box reboots / sleeps (single point of failure) | Medium | Stateless single-shot jobs; resume on next fire; cloud lift later |
| Concurrent research cycles corrupt worktree/main | Medium | File lock serializes cycles (§6.4) |
| Double-trade on timer double-fire / restart | Medium | Session-idempotency guard (§5 Group 4) |
| Accidental live trade | Critical | No live creds on the box; live path refuses |
