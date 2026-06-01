# Algua — Platform Architecture Design

**Date:** 2026-05-29
**Status:** Approved (architecture + roadmap); sub-project specs to follow.

## 1. Purpose

Algua is the core repository for a solo operator's algorithmic-trading operation:
research, backtesting, forward/paper testing, full strategy-lifecycle tracking, and an
**agent-first** workflow that takes a strategy from idea to production. The platform is
built to scale up over time.

**Trading thesis (philosophy, not a fixed signal):** ride institutional / "whale"
momentum — trade *with* the big participants rather than against them, and avoid being
the liquidity that gets faked out. No specific signal is committed to yet, so the
architecture must stay **data-source-agnostic and research-first**: the point is to
*discover* which whale-proxy signals actually work before locking in.

## 2. Constraints & Decisions

| Dimension | Decision |
|---|---|
| Operator profile | Capable coder; prefers leverage on mature OSS over custom plumbing. |
| Time horizon | Swing/multi-day + daily/intraday bars. **Batch + bar-polling, no tick-level/HFT.** |
| Broker | Alpaca (equities/ETFs), behind a thin broker-agnostic interface. |
| Engine architecture | **Approach A — "one signal definition + one execution contract"** (see §3). |
| Agent role | **Agent-first.** Agents autonomously operate the lifecycle through **paper**. |
| Human gate | **`paper → live` only.** Real money always requires verified human approval. |
| Runtime | Develop/backtest locally now; config-driven & containerizable for a clean cloud lift later. |

## 3. Engine Architecture — "One signal definition + one execution contract"

A strategy is defined **once** as a pure function: `features up to bar t → target weights`.
Both runners consume the same function:

- **Backtester** runs it over history (vectorbt for portfolio simulation, metrics, and
  fast parameter/walk-forward sweeps).
- **Live/paper loop** calls the same function on the latest *closed* bar and diffs target
  weights against current positions to generate orders.

**Parity is of *semantics*, not PnL.** vectorbt's vectorized fill model is not identical to
a live bar-by-bar loop, so parity is guaranteed by an explicit **`ExecutionContract`**, not
by shared code alone. The contract pins:

- Decision point: features computed on a **fully closed bar `t`**; orders fill **no earlier
  than `t+1`** open / next polling cycle. The backtest engine enforces the same `t→t+1`
  delay explicitly. *(Prevents look-ahead bias — the single most likely source of false
  edge.)*
- Rebalance cadence, rounding policy, cash handling, and when a target weight becomes an
  executable order.

Rejected alternatives: nautilus_trader unified event engine (too heavy for daily/swing,
slows research sweeps); fully custom event loop (reinvents risk/slippage/order modeling,
contradicts the leverage preference).

## 4. Design Principles

1. **One signal definition + one execution contract.** Strategies are pure functions;
   the `ExecutionContract` governs how weights become orders, identically in backtest and live.
2. **The command surface is the product.** Every lifecycle action is a typed CLI command
   with **structured (JSON) output**. Humans and agents drive the *identical* surface.
3. **Agents are first-class operators, not a bolted-on sandbox.** Agents author strategies,
   run research, advance lifecycle state, and operate paper trading — through the same
   commands a human uses. Operating playbooks live in the repo (`CLAUDE.md`, `docs/agent/`).
4. **Lifecycle is data with statistical gates.** Stage lives in the SQL registry (not in
   filesystem layout). Promotion requires an untouched holdout, recorded search breadth,
   and reproducibility stamps — not just a good-looking backtest.
5. **State is a record; the live gate is a wall.** See §6.
6. **Pure, well-bounded modules.** Small units with typed contracts — this is what lets an
   agent reliably reason about and modify the code, and what keeps backtest/live honest.

## 5. Repo Skeleton (agent-first, lean)

```
algua/
├── CLAUDE.md  +  docs/agent/        # operating playbooks — the heart of "agent-first"
├── cli/                             # THE command surface: every lifecycle action, JSON output
├── config/                          # pydantic-settings + per-strategy YAML (calendar, cadence, lookback)
├── algua/
│   ├── contracts/                   # Strategy, ExecutionContract, DataProvider, Broker, OrderIntent
│   ├── calendar/                    # market calendar / timezone / sessions (both runners depend on it)
│   ├── data/                        # providers (Alpaca, yfinance) + point-in-time store
│   │                                #   (raw+adjusted bars, universe snapshots by date, provenance, manifest)
│   ├── features/                    # pure, side-effect-free signal computation
│   ├── strategies/                  # agent-writable; one pure fn + config each; `algua strategy new` scaffolds
│   ├── backtest/                    # vectorbt engine w/ enforced t→t+1; walk-forward; slippage stress; JSON results
│   ├── tracking/                    # MLflow wrapper; mandatory code/config/data/seed stamps
│   ├── registry/                    # SQLite(WAL) — single source of truth for lifecycle stage + transitions
│   ├── execution/                   # broker iface + Alpaca PAPER adapter + lean order-intent state + reconcile
│   ├── risk/                        # exposure limits + kill-switch (validated in paper)
│   ├── live/                        # the loop (paper mode); warm-up gate; verifies live approval
│   └── audit/                       # lean append-only log: actor (human/agent/system) + reason
├── research/ tests/ docs/ data/ artifacts/ mlruns/
└── (deferred to live-hardening phase, kept as contracts/stubs):
       full reconciliation depth, restart playbook, real alerting/monitoring,
       secrets-in-prod, docker/cloud lift
```

Dependency rules (enforced via import lint / architecture tests): `strategies/` and
`features/` stay pure; `backtest/` and `live/` consume the same strategy contract but not
each other's internals; agent-facing code never imports live-execution internals directly.

## 6. Lifecycle, Registry & the Live Gate

**Stages:** `idea → backtested → shortlisted → paper → live → retired`.

- **State → SQL registry.** One row per strategy with a `stage` column plus an append-only
  transition history. Dashboards, queries, and audit derive directly from this. Agents
  advance state up through `paper` autonomously.
- **State is a record, not a wall.** Because agents write the registry, a bare `stage='live'`
  flag is forgeable by the very actor it should stop — and a trading agent ingests
  attacker-influenceable text (news, filings), making prompt-injection-to-live a real threat,
  on top of plain bugs/hallucination.
- **The `paper → live` transition requires a verified human approval.** A human-only command
  (gated by a credential/confirmation absent from the agent's runtime) mints an `approvals`
  row bound to the exact **code + config hash** being approved.
- **The live runner trusts the approval, not the flag.** It loads a strategy for live trading
  only if `stage='live'` **and** there is a valid, current approval whose hash matches the
  code/config it is about to run. No matching approval → it refuses to trade live.

This keeps the SQL-first, dashboard-friendly model **and** a gate that survives a buggy or
injected agent.

## 7. Correctness Essentials (in from day one)

These poison research if wrong, so they are not deferrable:

- **`t→t+1` execution-contract enforcement** in the backtester (no same-bar fills);
  test fixtures fail if a feature uses contemporaneous bar data for a same-bar fill.
- **Point-in-time data:** raw + explicit adjustment metadata, universe membership snapshots
  by date, source provenance; every backtest records exactly which dataset snapshot it used.
- **Market-calendar / timezone** module depended on by both runners (sessions, half-days,
  DST, data-availability lag).
- **Reproducibility stamps:** every run records code commit/artifact hash, config hash, data
  snapshot version, dependency lock, and RNG seed. Promotion is impossible without a
  reproducible rerun.
- **Statistical promotion gates:** untouched out-of-sample/walk-forward holdout, recorded
  search breadth (trials per idea family), and robustness metrics (parameter stability,
  turnover sensitivity, train→validation degradation) — to counter the multiple-testing risk
  inherent in agent-driven grid search.

## 8. Deferred to the live-hardening phase (contracts/stubs now)

Built as the operation approaches real money, not before an edge is found:
full economic-state reconciliation depth (cash/buying-power/corp-action edge cases),
crash/restart playbook, real alerting/monitoring infrastructure, production secrets
(keyring/mounted vs `.env`), Docker + cloud-VM deployment, kill-switch hardening.

## 9. Decomposition & Build Order

Six sub-projects; each gets its own spec → plan → implementation cycle. Build order
`1 → 2 → 3 → 4 → 5 → 6`. Sub-projects **1–3 form the "research MVP an agent can operate."**

1. **Foundation & command surface** — uv/pyproject, config, `contracts/`, `calendar/`,
   `cli/` skeleton with structured output, `registry/` schema, `CLAUDE.md` + agent docs.
   *→ An agent can discover and drive the system; lifecycle states exist.*
2. **Data layer (point-in-time)** — providers (Alpaca + yfinance), parquet store with
   provenance + universe snapshots + manifest, `algua data ingest/inspect`.
   *→ Reproducible historical datasets.*
3. **Research core** ⭐ — strategy template + scaffolding, `features/`, vectorbt engine with
   enforced `t→t+1`, walk-forward, slippage stress, `tracking/` stamps, registry transitions
   (idea→backtested→shortlisted). *→ Agent authors a strategy, backtests it, gets structured
   results. First real value.*
4. **Agent operating layer** — playbooks/skills for the autonomous research loop
   (ideate→author→backtest→interpret→shortlist) + statistical promotion gates.
   *→ Agent runs research end-to-end to a shortlist.*
5. **Execution + paper trading** — Alpaca paper adapter, lean order-intent state +
   reconciliation, risk limits + kill-switch, paper loop + warm-up gate, status/health
   commands, audit log. *→ Agent promotes to paper and operates it.*
6. **Live hardening + cloud lift** (human-gated) — verified-approval live gate, full
   reconciliation, restart playbook, alerting, secrets, Docker/VM. The verified approval is a
   **TOTP (authenticator-app) code** on `registry approve` (single-use, bound to code/config
   hash); the autonomous agent must never hold the TOTP seed, and the live transition runs vetted
   code in a trusted context the agent isn't in.
   *→ Human-gated path to real money.*

## 10. Open Questions / To Revisit Later

- Concrete whale-proxy data sources & budget (13F, options flow, insider/Form 4, dark-pool) —
  deferred until research surfaces what's worth paying for; the data layer stays agnostic.
- MLflow local-file backend → real backend migration trigger (before agents launch unbounded
  run counts).
- vectorbt operating envelope (universe size × history × grid breadth) before introducing a
  chunked/alternate backtest backend.
